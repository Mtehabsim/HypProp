"""Ground-truth-tree distortion probe — WHAT makes activations hierarchical,
and WHERE hyperbolic space helps (PREREGISTER3.md).

Replaces cloud four-point delta as the headline instrument. Instead of asking
whether an undifferentiated token cloud is globally tree-like (ceiling-bounded,
clustering-confounded — see v2), we ask a sharper question: given the KNOWN
ground-truth is-a tree of each prompt (retained by ``data/prontoqa_tree.py``),
can a capacity- and conditioning-matched decoder recover the tree's pairwise
distances from concept-token representations — and does negative curvature
recover it at LOWER dimension than flat space?

Design
------
Each prompt has its own small ontology (~15 concepts), so we train ONE SHARED
linear decoder ``g`` whose stress loss is block-diagonal: only within-prompt
concept pairs contribute (cross-prompt tree distance is undefined). The
train/val split is over PROMPTS. This tests "is there a single subspace in which
the model's concept representations encode the is-a tree" — i.e. a usable
read-out head, not a per-prompt curve fit.

Two geometries, IDENTICAL conditioning (LayerNorm + spectral-norm + bounded
scaling + MDR; reused from ``matched_probe.MatchedProbe``), identical params,
epochs, optimiser, init; NO learnable curvature (capacity match):
  cond_euclidean (c=0)   vs   hyperbolic (Poincare, c=0.5)
Advantage  Delta(layer, role, m) = rho_hyperbolic - rho_cond_euclidean, scored on
held-out prompts, whitened (train-only fit), >=5 seeds, Wilcoxon.

Secondary: the distortion-vs-dimension curve (curvature is "used" iff Delta rises
as m shrinks toward 2-4), the radial-norm<->depth correlation (training-free
fingerprint that separates hierarchy from angular clustering), and a
shuffled-tree permutation null.

Decision rules and thresholds are pre-registered in PREREGISTER3.md §5.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ..io import ensure_dir, iter_samples, log_line, save_csv, save_json
from ..manifest import write_manifest
from . import poincare
from .concept_align import concept_matrix
from .matched_probe import MatchedProbe, _whiten_fit, hypll_distance_check
from .structural_probe import _spearman

# Pre-registered thresholds (PREREGISTER3.md §5).
THRESHOLDS = dict(
    delta_margin=0.05,
    slope_margin=0.03,
    shuffle_ceiling=0.10,
    radial_depth_rho_min=0.30,
    positive_control_radial_min=0.50,
    n_seeds=6,   # >=6 so the one-sided signed-rank floor (1/2**n) can clear 0.05
    wilcoxon_alpha=0.05,
    decode_dims=(2, 3, 5, 8, 16),
    branching_levels=(1, 2, 3),
    curvature=0.5,
)


# --------------------------------------------------------------------------- #
# Core: shared-decoder tree distortion fit
# --------------------------------------------------------------------------- #
def _batch(prompts):
    """Pad a list of (X, D) prompts to a ``(P, maxn, ...)`` batch.

    Returns ``(Z, tgt, pair_mask, weight, sizes)``: only WITHIN-prompt
    off-diagonal pairs are active in ``pair_mask``; ``weight`` normalises each
    prompt block by its own sum-of-squares and by P, so the masked weighted loss
    equals the mean over prompts of the stress-normalised loss (equal weight per
    prompt). Batched over prompts, so one forward computes only the P*(n*n)
    within-prompt distances — not the (sum n)^2 cross-prompt matrix.
    """
    Xs = [np.asarray(X, dtype=np.float64) for X, _ in prompts]
    Ds = [np.asarray(D, dtype=np.float64) for _, D in prompts]
    sizes = [x.shape[0] for x in Xs]
    P, mn, d = len(Xs), max(sizes), Xs[0].shape[1]
    Z = np.zeros((P, mn, d), dtype=np.float64)
    tgt = np.zeros((P, mn, mn), dtype=np.float64)
    pair_mask = np.zeros((P, mn, mn), dtype=bool)
    weight = np.zeros((P, mn, mn), dtype=np.float64)
    for p, (x, D, n) in enumerate(zip(Xs, Ds, sizes)):
        Z[p, :n] = x
        tgt[p, :n, :n] = D
        m = ~np.eye(n, dtype=bool)
        pair_mask[p, :n, :n] = m
        denom = max((D[m] ** 2).sum(), 1e-9)
        weight[p, :n, :n] = m / (denom * P)
    return (torch.as_tensor(Z, dtype=torch.float32),
            torch.as_tensor(tgt, dtype=torch.float32),
            torch.as_tensor(pair_mask),
            torch.as_tensor(weight, dtype=torch.float32),
            sizes)


def _batched_dist(model, Z):
    """Geodesic distances within each prompt of a batch. ``Z`` (P, mn, d) -> (P, mn, mn)."""
    zt = model.transformed(Z)                      # (P, mn, m) — pointwise transform
    P, mn, m = zt.shape
    c = model.curvature if model.arm == "hyperbolic" else 0.0
    zi = zt.unsqueeze(2).expand(P, mn, mn, m).reshape(-1, m)
    zj = zt.unsqueeze(1).expand(P, mn, mn, m).reshape(-1, m)
    return poincare.dist(zi, zj, c).reshape(P, mn, mn)


def _batch_rho(dpred, tgt, sizes, shuffle_rng=None):
    """Mean over prompts of Spearman(decoded, true) on each prompt's off-diagonal."""
    dpred = np.asarray(dpred)
    tgt = np.asarray(tgt)
    rhos = []
    for p, n in enumerate(sizes):
        if n < 4:
            continue
        sub = dpred[p, :n, :n]
        Dt = tgt[p, :n, :n]
        if shuffle_rng is not None:
            perm = shuffle_rng.permutation(n)
            Dt = Dt[np.ix_(perm, perm)]
        m = ~np.eye(n, dtype=bool)
        rhos.append(_spearman(sub[m], Dt[m]))
    return float(np.mean(rhos)) if rhos else float("nan")


def _per_prompt_rho(model, prompts, shuffle_rng=None):
    """Mean over prompts of Spearman(decoded within-prompt dist, true tree dist)."""
    Z, tgt, _, _, sizes = _batch(prompts)
    with torch.no_grad():
        dpred = _batched_dist(model, Z).numpy()
    return _batch_rho(dpred, tgt.numpy(), sizes, shuffle_rng=shuffle_rng)


def fit_tree_arm(arm, train_prompts, val_prompts, proj_dim=5, seed=0,
                 curvature=0.5, max_epochs=1000, patience=6, check_every=50,
                 lr=1e-2):
    """Train the shared decoder to VAL-RHO convergence; return held-out scores.

    ``train_prompts``/``val_prompts`` are lists of ``(X, D)`` where ``X`` is
    ``(n_concepts, hidden)`` whitened concept reps for one prompt and ``D`` is
    that prompt's ``(n_concepts, n_concepts)`` ground-truth tree-distance matrix.
    Block-diagonal stress (within-prompt pairs only), batched over prompts into
    one padded forward pass. Early stopping on mean val rho (absolute-tolerance
    criterion — the v2 fix for the under-training trap).
    """
    in_dim = train_prompts[0][0].shape[1]
    model = MatchedProbe(in_dim, proj_dim, arm, seed=seed, curvature=curvature)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Ztr, tgt_tr, mask_tr, w_tr, _ = _batch(train_prompts)
    Zva, tgt_va, _, _, sizes_va = _batch(val_prompts)
    tgt_va_np = tgt_va.numpy()

    best_rho, best_epoch, stale = -2.0, 0, 0
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        dp = _batched_dist(model, Ztr)
        loss = (w_tr[mask_tr] * (dp[mask_tr] - tgt_tr[mask_tr]) ** 2).sum()
        loss.backward()
        opt.step()
        if epoch % check_every == 0:
            with torch.no_grad():
                rho = _batch_rho(_batched_dist(model, Zva).numpy(), tgt_va_np, sizes_va)
            if rho > best_rho + 1e-3:
                best_rho, best_epoch, stale = rho, epoch, 0
            else:
                stale += 1
                if stale >= patience:
                    break
    with torch.no_grad():
        val_rho = _batch_rho(_batched_dist(model, Zva).numpy(), tgt_va_np, sizes_va)
    return dict(model=model, val_rho=val_rho, best_val_rho=best_rho,
                epochs_trained=epoch, best_epoch=best_epoch)


# --------------------------------------------------------------------------- #
# Training-free fingerprint: radial norm <-> node generality (depth)
# --------------------------------------------------------------------------- #
def radial_depth_correlation(prompt_arrays):
    """Per-prompt Spearman(||concept rep||, node depth), averaged over prompts.

    Class clustering is angular and does NOT predict a radial ordering by
    generality, so a norm<->depth correlation is evidence of genuine hierarchy
    (root/general concepts nearer the origin, leaves farther) beyond clustering.
    Returns (signed_mean, abs_mean, n_prompts). ``prompt_arrays`` is a list of
    ``(X, depths)``.

    IMPORTANT: feed RAW (or mean-centered) features, NOT whitened ones —
    PCA-whitening equalises per-component variance and re-centers, which erases
    the norm-encodes-depth signal (verified on the CPU positive control: raw
    rho=+0.66, whitened rho=-0.03). This is why the tree-probe run computes the
    radial fingerprint before the whitening step.
    """
    signed = []
    for X, depths in prompt_arrays:
        X = np.asarray(X)
        if X.shape[0] < 4 or np.ptp(depths) == 0:
            continue
        norms = np.linalg.norm(X, axis=1)
        signed.append(_spearman(norms, depths))
    if not signed:
        return float("nan"), float("nan"), 0
    signed = np.asarray(signed)
    return float(signed.mean()), float(np.abs(signed).mean()), len(signed)


def dist0_depth_correlation(model, prompt_arrays):
    """Per-prompt Spearman(poincare.dist0(hyperbolic embedding), depth), averaged.

    The usage-relevant radial measure: after the trained hyperbolic decoder maps
    a concept to the Poincare ball, does its geodesic distance from the ORIGIN
    order concepts by taxonomic depth (general near center, specific near
    boundary)? Robust to raw-activation-norm artifacts because the decoder learns
    the reference frame. ``prompt_arrays`` is a list of ``(X, depths)`` with the
    SAME (whitened) features the model was trained on.
    """
    signed = []
    with torch.no_grad():
        for X, depths in prompt_arrays:
            if np.asarray(X).shape[0] < 4 or np.ptp(depths) == 0:
                continue
            z = model.transformed(torch.as_tensor(np.asarray(X), dtype=torch.float32))
            c = model.curvature if model.arm == "hyperbolic" else 0.0
            d0 = poincare.dist0(z, c).numpy()
            signed.append(_spearman(d0, depths))
    if not signed:
        return float("nan"), 0
    return float(np.mean(signed)), len(signed)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _arm_key(sample):
    tm = sample.get("tree_meta") or {}
    return (tm.get("naming", "?"), int(tm.get("branching", -1)))


def _gather(samples, layer, role, pool="last"):
    """Collect per-prompt (X, node_ids, D, depths) for samples at (layer, role)."""
    out = []
    for s in samples:
        cm = concept_matrix(s, layer, role=role, pool=pool)
        if cm is not None:
            out.append(cm)
    return out


def run(activations_dir, out_dir, dataset="prontoqa_tree", roles=("premise", "query", "last"),
        dims=THRESHOLDS["decode_dims"], seeds=tuple(range(THRESHOLDS["n_seeds"])),
        curvature=THRESHOLDS["curvature"], layer_stride=4, layers=None,
        max_epochs=1000, max_prompts=80, pool="last"):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                           "logs", "tree_probe.log")

    check = hypll_distance_check()
    # HARD GATE: our Poincare distance must match the dependency-free textbook
    # arcosh closed form. This is the real correctness proof (no shared code with
    # dist()), and it does NOT depend on HypLL being installed.
    if not check["ok"]:
        raise RuntimeError(
            f"Poincare distance disagrees with the textbook arcosh closed form "
            f"(max err {check['closed_form_max_abs_err']:.2e}) — our geometry is "
            f"wrong; fix before running")
    log_line(logfile, f"Poincare distance matches the textbook closed form "
                      f"(max err {check['closed_form_max_abs_err']:.2e})")
    # SOFT cross-check: HypLL, if installed. A convention (curvature-scale)
    # difference is expected and fine; only a mismatch under ALL conventions
    # would be alarming — and the closed-form gate above already rules that out.
    hy = check.get("hypll")
    if hy == "not installed":
        log_line(logfile, "HypLL not installed -> skipping the library cross-check "
                          "(closed-form gate already passed)")
    elif check.get("hypll_ok"):
        log_line(logfile, f"HypLL cross-check matches under the "
                          f"'{check['hypll_best_convention']}' curvature convention "
                          f"(max err {check['hypll_max_abs_err']:.2e})")
    else:
        log_line(logfile, f"NOTE: HypLL differs under all tested conventions "
                          f"(best {check.get('hypll_best_convention')}: "
                          f"{check.get('hypll_max_abs_err', float('nan')):.2e}); "
                          f"proceeding on the closed-form gate (our dist is proven "
                          f"correct vs the textbook form).")

    rows = []          # one row per (model, arm, role, layer, dim, seed, geometry)
    radial_rows = []   # one row per (model, arm, role, layer)
    models = sorted({s["model"] for s in iter_samples(activations_dir, dataset=dataset)})
    for model in models:
        all_samples = [s for s in iter_samples(activations_dir, model, dataset)
                       if s.get("tree_meta")]
        # split into preregistered arms
        arms: dict = {}
        for s in all_samples:
            arms.setdefault(_arm_key(s), []).append(s)
        if not arms:
            log_line(logfile, f"{model}: no tree_meta samples; skipping")
            continue
        n_layers = int(np.asarray(all_samples[0]["hidden"]).shape[0])
        use_layers = layers if layers else list(range(0, n_layers, layer_stride))
        if (n_layers - 1) not in use_layers:
            use_layers.append(n_layers - 1)

        for (naming, branching), samples in sorted(arms.items()):
            samples = samples[:max_prompts]
            arm_label = f"{naming}_b{branching}"
            for role in roles:
                for layer in use_layers:
                    gathered = _gather(samples, layer, role, pool=pool)
                    if len(gathered) < 12:
                        continue
                    Xs = [g[0] for g in gathered]
                    Ds = [g[2] for g in gathered]
                    depths = [g[3] for g in gathered]

                    for seed in seeds:
                        rng = np.random.default_rng(seed)
                        perm = rng.permutation(len(Xs))
                        ntr = int(0.7 * len(Xs))
                        tr, va = perm[:ntr], perm[ntr:]
                        # whiten on TRAIN concept reps only (leakage fix)
                        Xtr_cat = np.concatenate([Xs[i] for i in tr], axis=0)
                        wf = _whiten_fit(Xtr_cat)
                        train_p = [(wf(Xs[i]), Ds[i]) for i in tr]
                        val_p = [(wf(Xs[i]), Ds[i]) for i in va]

                        # radial fingerprint (training-free): on RAW (unwhitened)
                        # val features — whitening erases the norm<->depth signal.
                        if seed == seeds[0]:
                            rp = [(Xs[i], depths[i]) for i in va]
                            r_signed, r_abs, r_n = radial_depth_correlation(rp)
                            radial_rows.append(dict(
                                model=model, arm=arm_label, naming=naming,
                                branching=branching, role=role, layer=layer,
                                radial_depth_rho=round(r_signed, 4),
                                radial_depth_absrho=round(r_abs, 4), n_prompts=r_n))

                        for m in dims:
                            fits = {}
                            for arm in ("cond_euclidean", "hyperbolic"):
                                res = fit_tree_arm(arm, train_p, val_p, proj_dim=m,
                                                   seed=seed, curvature=curvature,
                                                   max_epochs=max_epochs)
                                fits[arm] = res
                            # shuffled-tree null at this dim (permutation test on the
                            # trained hyperbolic decoder)
                            shuf = _per_prompt_rho(
                                fits["hyperbolic"]["model"], val_p,
                                shuffle_rng=np.random.default_rng(1000 + seed))
                            rows.append(dict(
                                model=model, arm=arm_label, naming=naming,
                                branching=branching, role=role, layer=layer,
                                dim=m, seed=seed,
                                rho_cond_euc=round(fits["cond_euclidean"]["val_rho"], 4),
                                rho_hyp=round(fits["hyperbolic"]["val_rho"], 4),
                                delta=round(fits["hyperbolic"]["val_rho"]
                                            - fits["cond_euclidean"]["val_rho"], 4),
                                shuffle_rho_hyp=round(shuf, 4),
                                n_train=len(train_p), n_val=len(val_p)))
                    log_line(logfile, f"{model} [{arm_label}] {role} L{layer}: "
                                      f"{len(seeds)}s x {len(dims)}d done "
                                      f"({len(gathered)} prompts w/ >=4 concepts)")

    save_csv(os.path.join(out_dir, "tree_probe.csv"), rows)
    save_csv(os.path.join(out_dir, "tree_probe_radial.csv"), radial_rows)
    verdict = _verdict(rows, radial_rows, logfile)
    save_json(os.path.join(out_dir, "tree_probe_verdict.json"), verdict)
    _write_verdict_md(os.path.join(out_dir, "tree_probe_verdict.md"), verdict, check)
    write_manifest(out_dir, "tree_probe",
                   args=dict(activations=activations_dir, dataset=dataset,
                             roles=list(roles), dims=list(dims),
                             seeds=list(seeds), curvature=curvature,
                             layer_stride=layer_stride, max_epochs=max_epochs),
                   extra=dict(hypll_check=check, thresholds=_json_thresholds()))
    return rows, verdict


# --------------------------------------------------------------------------- #
# Verdict — the pre-registered 4 gates + dose-response
# --------------------------------------------------------------------------- #
def _wilcoxon_p(diffs):
    """One-sided (greater) Wilcoxon p on the per-seed hyperbolic-advantage diffs.

    ONE-SIDED because the pre-registered hypothesis is directional (hyperbolic
    beats matched Euclidean), which pre-registration licenses and which is more
    powerful. CRITICAL: the two-sided signed-rank floor with n seeds is 2/2**n,
    so with n=5 the smallest achievable two-sided p is 0.0625 > 0.05 -> G1 could
    NEVER fire (this is the trap that made v2's +0.089..+0.145 conditioning gaps
    read as "p~0.062, just short"). We therefore require n_seeds>=6 and score
    one-sided (n=6 -> one-sided floor 1/64=0.016, two-sided 2/64=0.031).
    """
    from scipy.stats import wilcoxon
    nz = np.asarray(diffs)[np.asarray(diffs) != 0]
    if len(nz) < 6:
        return float("nan")
    try:
        return float(wilcoxon(nz, alternative="greater")[1])
    except ValueError:
        return float("nan")


def _cell(rows, model, arm, role, layer, dim):
    return [r for r in rows if r["model"] == model and r["arm"] == arm
            and r["role"] == role and r["layer"] == layer and r["dim"] == dim]


def _verdict(rows, radial_rows, logfile):
    T = THRESHOLDS
    models = sorted({r["model"] for r in rows})
    roles = sorted({r["role"] for r in rows})
    dims = sorted({r["dim"] for r in rows})
    low_dim = min(dims)
    high_dim = max(dims)
    # A cell is "complete" when it has all the seeds that were actually run (not a
    # hardcoded 5) — Wilcoxon's own >=5 guard still enforces significance, so a
    # short run can never spuriously pass G1. This keeps validation runs honest.
    n_seeds_run = len({r["seed"] for r in rows}) or T["n_seeds"]
    out = {"thresholds": _json_thresholds(), "positions": [], "dose_response": [],
           "radial": [], "by_model": {}}

    for model in models:
        arms = sorted({r["arm"] for r in rows if r["model"] == model})
        layers = sorted({r["layer"] for r in rows if r["model"] == model})
        out["by_model"][model] = {"suitable_positions": 0}

        # --- WHERE: 4-gate scan over (arm, role, layer), pick best low dim ---
        for arm in arms:
            for role in roles:
                for layer in layers:
                    # best low dim = the dim (<=5) with the largest mean delta
                    best = None
                    for m in [d for d in dims if d <= 5] or [low_dim]:
                        cell = _cell(rows, model, arm, role, layer, m)
                        if len(cell) < n_seeds_run:
                            continue
                        md = float(np.mean([c["delta"] for c in cell]))
                        if best is None or md > best[1]:
                            best = (m, md, cell)
                    if best is None:
                        continue
                    m_star, mean_delta, cell = best
                    p = _wilcoxon_p([c["delta"] for c in cell])
                    # G2 slope: delta(low) - delta(high)
                    hi = _cell(rows, model, arm, role, layer, high_dim)
                    slope = (mean_delta - float(np.mean([c["delta"] for c in hi]))
                             if hi else float("nan"))
                    # G3 controls: shuffled-tree ceiling at m_star (hyp)
                    shuf = float(np.mean([c["shuffle_rho_hyp"] for c in cell]))
                    g1 = (mean_delta >= T["delta_margin"]) and (p < T["wilcoxon_alpha"])
                    g2 = np.isfinite(slope) and (slope >= T["slope_margin"])
                    g3 = shuf < T["shuffle_ceiling"]
                    rec = dict(model=model, arm=arm, role=role, layer=layer,
                               dim=m_star, mean_delta=round(mean_delta, 4),
                               wilcoxon_p=round(p, 4) if np.isfinite(p) else None,
                               slope=round(slope, 4) if np.isfinite(slope) else None,
                               shuffle_rho=round(shuf, 4),
                               G1_advantage=bool(g1), G2_curvature=bool(g2),
                               G3_controls=bool(g3),
                               suitable=bool(g1 and g2 and g3))
                    if g1 and g2 and g3:
                        out["positions"].append(rec)
                        out["by_model"][model]["suitable_positions"] += 1
                    elif mean_delta >= T["delta_margin"] / 2:
                        out.setdefault("near_miss", []).append(rec)

        # --- WHAT: branching dose-response (fictional arms), best role/layer/dim ---
        for role in roles:
            curve = []
            for b in T["branching_levels"]:
                arm = f"fictional_b{b}"
                # max mean-delta over layers at low dims for this arm/role
                best_md = float("-inf")
                for layer in layers:
                    for m in [d for d in dims if d <= 5]:
                        cell = _cell(rows, model, arm, role, layer, m)
                        if len(cell) >= n_seeds_run:
                            best_md = max(best_md, float(np.mean([c["delta"] for c in cell])))
                curve.append(None if best_md == float("-inf") else round(best_md, 4))
            if any(v is not None for v in curve):
                vals = [v for v in curve if v is not None]
                monotone = all(curve[i] is None or curve[i + 1] is None
                               or curve[i + 1] >= curve[i] - 1e-6
                               for i in range(len(curve) - 1))
                b1 = curve[0]
                b3 = curve[-1]
                dose_ok = (b1 is not None and b3 is not None
                           and b1 <= T["delta_margin"] / 2
                           and b3 >= T["delta_margin"] and monotone)
                out["dose_response"].append(dict(
                    model=model, role=role, branching=list(T["branching_levels"]),
                    max_delta_by_branching=curve, monotone_nondecreasing=bool(monotone),
                    negative_control_b1_ok=bool(b1 is not None and b1 <= T["delta_margin"] / 2),
                    dose_response_positive=bool(dose_ok)))

    # --- radial fingerprint summary (best |rho| per model/role) ---
    for model in sorted({r["model"] for r in radial_rows}):
        for role in sorted({r["role"] for r in radial_rows if r["model"] == model}):
            sub = [r for r in radial_rows if r["model"] == model and r["role"] == role]
            if not sub:
                continue
            best = max(sub, key=lambda r: abs(r["radial_depth_rho"]))
            out["radial"].append(dict(
                model=model, role=role, best_layer=best["layer"],
                radial_depth_rho=best["radial_depth_rho"],
                passes=bool(abs(best["radial_depth_rho"]) >= T["radial_depth_rho_min"])))

    n_pos = len(out["positions"])
    log_line(logfile, f"VERDICT: {n_pos} suitable position(s) passed G1-G3; "
                      f"{sum(1 for d in out['dose_response'] if d['dose_response_positive'])}"
                      f"/{len(out['dose_response'])} dose-response curves positive; "
                      f"{sum(1 for r in out['radial'] if r['passes'])}"
                      f"/{len(out['radial'])} radial fingerprints pass")
    return out


def _json_thresholds():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in THRESHOLDS.items()}


def _write_verdict_md(path, verdict, check):
    L = []
    L.append("# Tree-probe verdict (PREREGISTER3)\n")
    L.append(f"HypLL cross-check: {check}\n")
    L.append("\n## WHERE — suitable positions for hyperbolic space (G1&G2&G3)\n")
    if verdict["positions"]:
        L.append("| model | arm | role | layer | dim | mean Δ | p | slope | shuffle ρ |\n")
        L.append("|---|---|---|---|---|---|---|---|---|\n")
        for r in verdict["positions"]:
            L.append(f"| {r['model']} | {r['arm']} | {r['role']} | {r['layer']} | "
                     f"{r['dim']} | {r['mean_delta']:+.3f} | {r['wilcoxon_p']} | "
                     f"{r['slope']} | {r['shuffle_rho']:.3f} |\n")
    else:
        L.append("_No position passed all three gates — hyperbolic geometry does not "
                 "beat matched-capacity Euclidean on the ground-truth tree here._\n")
        if verdict.get("near_miss"):
            L.append(f"\n_{len(verdict['near_miss'])} near-miss cell(s) with "
                     f"mean Δ ≥ margin/2 (see JSON)._\n")
    L.append("\n## WHAT — branching dose-response (fictional arms)\n")
    L.append("| model | role | Δ(b=1) | Δ(b=2) | Δ(b=3) | monotone | b1 control | positive |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for d in verdict["dose_response"]:
        c = d["max_delta_by_branching"]
        fmt = lambda v: "—" if v is None else f"{v:+.3f}"
        L.append(f"| {d['model']} | {d['role']} | {fmt(c[0])} | {fmt(c[1])} | {fmt(c[2])} | "
                 f"{d['monotone_nondecreasing']} | {d['negative_control_b1_ok']} | "
                 f"**{d['dose_response_positive']}** |\n")
    L.append("\n## Radial-norm ↔ generality (training-free fingerprint)\n")
    L.append("| model | role | best layer | ρ(norm, depth) | passes |\n")
    L.append("|---|---|---|---|---|\n")
    for r in verdict["radial"]:
        L.append(f"| {r['model']} | {r['role']} | {r['best_layer']} | "
                 f"{r['radial_depth_rho']:+.3f} | {r['passes']} |\n")
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        fh.write("".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Ground-truth-tree distortion probe: what makes activations "
                    "hierarchical + where hyperbolic space helps (PREREGISTER3).")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/tree_probe")
    ap.add_argument("--dataset", default="prontoqa_tree")
    ap.add_argument("--roles", nargs="+", default=["premise", "query", "last"])
    ap.add_argument("--dims", type=int, nargs="+", default=list(THRESHOLDS["decode_dims"]))
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(THRESHOLDS["n_seeds"])),
                    help="need >=6 for the one-sided signed-rank floor to clear 0.05")
    ap.add_argument("--curvature", type=float, default=THRESHOLDS["curvature"])
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--max-prompts", type=int, default=80)
    ap.add_argument("--pool", default="last", choices=["last", "mean"])
    args = ap.parse_args(argv)
    run(args.activations, args.out, dataset=args.dataset, roles=tuple(args.roles),
        dims=tuple(args.dims), seeds=tuple(args.seeds), curvature=args.curvature,
        layer_stride=args.layer_stride, layers=args.layers,
        max_epochs=args.max_epochs, max_prompts=args.max_prompts, pool=args.pool)


if __name__ == "__main__":
    main()
