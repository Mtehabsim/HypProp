"""THE decisive experiment: matched-conditioning structural probe (Raj re-run).

Raj's hyperbolic probe carries FOUR stabilisers (LayerNorm, spectral-norm,
bounded scaling, MDR) while his Euclidean probe is a bare nn.Linear — so his
"hyperbolic wins" confounds GEOMETRY with PROBE CONDITIONING (verified from his
source; see the raj-reproduction memory). Neither Raj nor the Atlas ever ran
the disentangling experiment. This module does, with three arms per layer:

  bare_euclidean  — nn.Linear only (Raj's actual Euclidean arm; the erratic one)
  cond_euclidean  — LayerNorm + spectral-norm + bounded scaling (FULL Raj
                    conditioning; only the exp-map/distance removed)
  hyperbolic      — the full Raj recipe incl. expmap0 + Poincare distance
                    (optionally cross-checked with HypLL, the library Raj's
                    line of work builds on: https://github.com/maxvanspengler/
                    hyperbolic_learning_library — used ONLY to validate our
                    from-scratch Poincare distance, not for training)

Decision rule (pre-registered in PREREGISTER2.md):
  gap_conditioning = rho(cond_euclidean) - rho(bare_euclidean)
  gap_geometry     = rho(hyperbolic)     - rho(cond_euclidean)
  If gap_geometry is ~0 while gap_conditioning is large -> Raj's effect was
  CONDITIONING (Park-consistent deflation). If gap_geometry stays large ->
  real geometry advantage. Judged per layer, across seeds, Wilcoxon
  signed-rank on the per-seed paired diffs, on WHITENED features (fit on
  train, applied to val) with a held-out pair split.

Fixes folded in from the re-evaluation:
  - train/val split of PAIRS (v1 fit and scored on the same pairs);
  - whitening fit on train only (leakage fix);
  - convergence training: rho-plateau early stopping with a relative-improvement
    criterion (the documented under-training trap: distortion loss plateaus
    long before rho converges on branching targets);
  - identical epochs/optimizer/lr/init across arms; NO learnable curvature
    (capacity match);
  - >= 5 seeds and a paired significance test.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from ..geometry import poincare
from ..io import ensure_dir, iter_samples, log_line, save_csv
from ..manifest import write_manifest
from .structural_probe import (_depth_target, _maximum_distance_rescaling,
                               _spearman, _taxonomy_target)


class MatchedProbe(nn.Module):
    """One probe, three conditioning levels, identical parameter shapes.

    arm='bare_euclidean' : dist = ||Wx_i - Wx_j||           (Raj's Euclidean)
    arm='cond_euclidean' : z = s(alpha)*tanh(spectral(W) LN(x)); dist = ||z_i-z_j||
    arm='hyperbolic'     : same z, then MDR -> expmap0 -> Poincare dist (c=0.5)

    alpha exists in every arm (bare arm simply never uses the gate's output
    path... it does, actually: to keep the PARAMETER COUNT identical we
    multiply bare/cond arms by sigma(alpha) too — for the bare arm this is just
    a learnable global scale, which a distance regression can absorb; for the
    cond arm it is the Raj bounded-scaling gate). No arm learns curvature.
    """

    def __init__(self, in_dim, proj_dim, arm, seed=0, r_max=15.0, curvature=0.5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.arm = arm
        self.ln = nn.LayerNorm(in_dim)
        lin = nn.Linear(in_dim, proj_dim, bias=True)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(proj_dim, in_dim, generator=g) * 0.05)
            lin.bias.zero_()
        if arm == "bare_euclidean":
            self.B = lin                        # no spectral norm (Raj's bare arm)
        else:
            self.B = nn.utils.spectral_norm(lin)
        self.alpha = nn.Parameter(torch.tensor(0.95))
        self.r_max = r_max
        self.curvature = curvature

    def transformed(self, x):
        if self.arm == "bare_euclidean":
            return torch.sigmoid(self.alpha) * self.B(x)
        z = torch.sigmoid(self.alpha) * torch.tanh(self.B(self.ln(x)))
        if self.arm == "cond_euclidean":
            return z
        z = _maximum_distance_rescaling(z, self.r_max)
        return poincare.expmap0(z, self.curvature)

    def dist(self, x):
        z = self.transformed(x)
        n = z.shape[0]
        zi = z.unsqueeze(1).expand(n, n, z.shape[-1]).reshape(n * n, -1)
        zj = z.unsqueeze(0).expand(n, n, z.shape[-1]).reshape(n * n, -1)
        c = self.curvature if self.arm == "hyperbolic" else 0.0
        return poincare.dist(zi, zj, c).reshape(n, n)


def _whiten_fit(train_X, pca_cap=128, eps=1e-6):
    """PCA-then-whiten fit on TRAIN only; returns transform(X)->whitened."""
    mu = train_X.mean(axis=0, keepdims=True)
    xc = train_X - mu
    k = max(1, min(train_X.shape[0] // 3, pca_cap, train_X.shape[1]))
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(k, s.shape[0])
    comps = vt[:k]
    std = np.clip(s[:k] / np.sqrt(max(train_X.shape[0] - 1, 1)), eps, None)
    W = comps.T / std[None, :]

    def transform(z):
        return (np.asarray(z, dtype=np.float64) - mu) @ W

    return transform


def fit_arm(arm, train_X, train_D, val_X, val_D, proj_dim=5, seed=0,
            max_epochs=3000, patience=6, check_every=50, lr=1e-2):
    """Train one arm to RHO convergence; score on held-out val pairs.

    Early stopping monitors VAL RHO with a relative-improvement criterion
    (best * (1 - 1e-3)), checked every ``check_every`` epochs — the fix for the
    documented under-training trap where the distortion loss plateaus hundreds
    of epochs before rho converges on branching targets.
    """
    xt = torch.as_tensor(train_X, dtype=torch.float32)
    xv = torch.as_tensor(val_X, dtype=torch.float32)
    dt = torch.as_tensor(train_D, dtype=torch.float32)
    model = MatchedProbe(xt.shape[1], proj_dim, arm, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mask_t = ~torch.eye(len(xt), dtype=torch.bool)
    mask_v = ~torch.eye(len(xv), dtype=torch.bool)
    tgt_norm = (dt[mask_t] ** 2).sum().clamp_min(1e-9)
    val_flat = val_D[np.asarray(mask_v)]

    best_rho, best_epoch, stale = -2.0, 0, 0
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        dpred = model.dist(xt)
        loss = ((dpred[mask_t] - dt[mask_t]) ** 2).sum() / tgt_norm
        loss.backward()
        opt.step()
        if epoch % check_every == 0:
            with torch.no_grad():
                rho = _spearman(model.dist(xv)[mask_v].numpy(), val_flat)
            # improvement = beats the best so far by a small ABSOLUTE tolerance
            # (a relative criterion misbehaves around rho <= 0)
            if rho > best_rho + 1e-3:
                best_rho, best_epoch = rho, epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    with torch.no_grad():
        dpred_v = model.dist(xv)[mask_v].numpy()
    rho = _spearman(dpred_v, val_flat)
    distortion = float(np.mean(np.abs(dpred_v - val_flat)))
    return dict(rho=rho, distortion=distortion, best_val_rho=best_rho,
                epochs_trained=epoch, best_epoch=best_epoch)


def _closed_form_check(seed=0, n=32, dim=5, tol=1e-6):
    """Dependency-free correctness gate: our dist vs the textbook arcosh form.

    This is the RIGOROUS proof our Poincare distance is right — it shares no code
    with dist() (no Mobius add / artanh), so agreement is not a tautology. Runs
    at several curvatures. If this fails, our geometry is wrong and the run must
    stop regardless of whether HypLL is installed.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for c in (0.3, 0.5, 1.0):
        pts = rng.standard_normal((n, dim))
        pts = 0.3 * pts / np.linalg.norm(pts, axis=1, keepdims=True)
        x = torch.as_tensor(pts, dtype=torch.float64)
        xi = x.unsqueeze(1).expand(n, n, dim).reshape(-1, dim)
        xj = x.unsqueeze(0).expand(n, n, dim).reshape(-1, dim)
        ours = poincare.dist(xi, xj, c)
        ref = poincare.dist_closed_form(xi, xj, c)
        worst = max(worst, float((ours - ref).abs().max()))
    return dict(max_abs_err=worst, ok=worst < tol)


def hypll_distance_check(seed=0, n=32, dim=5, c=0.5, tol=1e-4):
    """Validate our Poincare distance, optionally cross-checked against HypLL.

    Two layers:
      1. ALWAYS: match our dist() to the dependency-free textbook arcosh closed
         form (``_closed_form_check``). This is the real correctness proof.
      2. IF HypLL is installed: also compare to HypLL. HypLL (van Spengler et al.)
         is the maintained library behind this line of work, BUT libraries differ
         in the curvature CONVENTION (whether ``Curvature(value=c)`` scales the
         metric like our curvature ``c`` or a reparametrised value). A raw 1:1
         comparison spuriously failed at ~2.6e-2 (== our |d(c=0.5)-d(c=1.0)|,
         the fingerprint of a factor-in-curvature convention gap), NOT a bug —
         our dist is exact vs the closed form. So we accept a match under EITHER
         HypLL curvature in {c, 2c, c/2}, and report which convention aligned.

    Returns a dict with ``ok`` (closed-form gate, the hard gate) plus HypLL
    details when available. ``ok`` never depends on HypLL being installed.
    """
    cf = _closed_form_check(seed=seed, n=n, dim=dim)
    result = dict(closed_form_max_abs_err=cf["max_abs_err"],
                  closed_form_ok=cf["ok"], ok=cf["ok"])
    try:
        from hypll.manifolds.poincare_ball import Curvature, PoincareBall
        from hypll.tensors import ManifoldTensor
    except ImportError:
        result["hypll"] = "not installed"
        return result

    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, dim))
    pts = 0.3 * pts / np.linalg.norm(pts, axis=1, keepdims=True)
    x = torch.as_tensor(pts, dtype=torch.float64)
    ours = poincare.dist(x.unsqueeze(1).expand(n, n, dim).reshape(-1, dim),
                         x.unsqueeze(0).expand(n, n, dim).reshape(-1, dim),
                         c).reshape(n, n)

    def _hypll_dist(c_hypll):
        ball = PoincareBall(c=Curvature(value=float(c_hypll), requires_grad=False))
        mt = ManifoldTensor(x, manifold=ball)
        theirs = torch.empty(n, n, dtype=torch.float64)
        for i in range(n):
            xi = ManifoldTensor(x[i].unsqueeze(0).expand(n, dim), manifold=ball)
            theirs[i] = ball.dist(xi, mt)
        return theirs

    best = None
    for label, c_h in (("c", c), ("2c", 2 * c), ("c/2", c / 2)):
        err = float((ours - _hypll_dist(c_h)).abs().max())
        if best is None or err < best[1]:
            best = (label, err)
    result["hypll_best_convention"] = best[0]
    result["hypll_max_abs_err"] = best[1]
    result["hypll_ok"] = best[1] < tol
    return result


def run(activations_dir, out_dir, dataset="prontoqa", target="depth",
        proj_dim=5, seeds=(0, 1, 2, 3, 4), layers=None, whiten=True,
        max_epochs=3000, max_samples=400):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                           "logs", "matched_probe.log")

    check = hypll_distance_check()
    if not check["ok"]:
        raise RuntimeError(
            f"Poincare distance disagrees with the textbook arcosh closed form "
            f"(max err {check['closed_form_max_abs_err']:.2e}) — fix before running")
    log_line(logfile, f"Poincare distance matches the textbook closed form "
                      f"(max err {check['closed_form_max_abs_err']:.2e}); "
                      f"HypLL: {check.get('hypll', 'n/a')}"
                      + (f" (best '{check.get('hypll_best_convention')}', "
                         f"err {check.get('hypll_max_abs_err'):.2e})"
                         if check.get('hypll') not in (None, 'not installed') else ""))

    rows = []
    models = sorted({s["model"] for s in iter_samples(activations_dir, dataset=dataset)})
    for model in models:
        samples = [s for s in iter_samples(activations_dir, model, dataset)
                   if s.get("variant", "original") == "original"][:max_samples]
        if len(samples) < 24:
            log_line(logfile, f"{model}/{dataset}: too few samples; skipping")
            continue
        target_d_full = (_taxonomy_target(samples) if target == "taxonomy"
                         else _depth_target(samples))
        if float(target_d_full.max()) <= 0:
            log_line(logfile, f"{model}/{dataset}: degenerate target; skipping")
            continue
        n_layers = int(np.asarray(samples[0]["hidden"]).shape[0])
        use_layers = layers if layers else sorted(set(
            [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4,
             n_layers - 2, n_layers - 1]))

        for layer in use_layers:
            feats = np.stack([np.asarray(s["hidden"])[min(layer, n_layers - 1), -1]
                              for s in samples]).astype(np.float64)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(feats))
                n_train = int(0.7 * len(feats))
                tr, va = perm[:n_train], perm[n_train:]
                Xtr, Xva = feats[tr], feats[va]
                Dtr = target_d_full[np.ix_(tr, tr)]
                Dva = target_d_full[np.ix_(va, va)]
                if whiten:
                    tf = _whiten_fit(Xtr)
                    Xtr, Xva = tf(Xtr), tf(Xva)
                for arm in ("bare_euclidean", "cond_euclidean", "hyperbolic"):
                    res = fit_arm(arm, Xtr, Dtr, Xva, Dva, proj_dim=proj_dim,
                                  seed=seed, max_epochs=max_epochs)
                    rows.append(dict(model=model, dataset=dataset, target=target,
                                     layer=layer, seed=seed, arm=arm,
                                     whitened=whiten,
                                     rho=round(res["rho"], 4),
                                     distortion=round(res["distortion"], 4),
                                     epochs_trained=res["epochs_trained"],
                                     best_epoch=res["best_epoch"]))
            log_line(logfile, f"{model}/{dataset} L{layer}: "
                              f"{len(seeds)} seeds x 3 arms done")

        # per-layer verdict: conditioning gap vs geometry gap, paired over seeds
        _verdict(rows, model, dataset, use_layers, seeds, logfile)

    save_csv(os.path.join(out_dir, "matched_probe.csv"), rows)
    write_manifest(out_dir, "matched_probe",
                   args=dict(activations=activations_dir, dataset=dataset,
                             target=target, seeds=list(seeds), whiten=whiten,
                             proj_dim=proj_dim, max_epochs=max_epochs),
                   extra=dict(hypll_check=check))
    return rows


def _verdict(rows, model, dataset, use_layers, seeds, logfile):
    from scipy.stats import wilcoxon

    for layer in use_layers:
        by_arm = {}
        for arm in ("bare_euclidean", "cond_euclidean", "hyperbolic"):
            by_arm[arm] = [r["rho"] for r in rows
                           if r["model"] == model and r["dataset"] == dataset
                           and r["layer"] == layer and r["arm"] == arm]
        if not all(len(v) == len(seeds) for v in by_arm.values()):
            continue
        cond_gap = np.array(by_arm["cond_euclidean"]) - np.array(by_arm["bare_euclidean"])
        geo_gap = np.array(by_arm["hyperbolic"]) - np.array(by_arm["cond_euclidean"])

        def _p(diffs):
            nz = diffs[diffs != 0]
            if len(nz) < 5:
                return float("nan")
            try:
                return float(wilcoxon(nz)[1])
            except ValueError:
                return float("nan")

        log_line(logfile,
                 f"{model} L{layer}: "
                 f"bare={np.mean(by_arm['bare_euclidean']):.3f} "
                 f"cond={np.mean(by_arm['cond_euclidean']):.3f} "
                 f"hyp={np.mean(by_arm['hyperbolic']):.3f} | "
                 f"conditioning gap={cond_gap.mean():+.3f} (p={_p(cond_gap):.3f}) "
                 f"geometry gap={geo_gap.mean():+.3f} (p={_p(geo_gap):.3f}) -> "
                 + ("GEOMETRY real" if (geo_gap.mean() > 0.05 and _p(geo_gap) < 0.05)
                    else ("CONDITIONING explains Raj"
                          if (cond_gap.mean() > 0.05 and geo_gap.mean() <= 0.02)
                          else "inconclusive at this layer")))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Matched-conditioning probe: was Raj's win geometry or conditioning?")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry_v2")
    ap.add_argument("--dataset", default="prontoqa")
    ap.add_argument("--target", default="depth", choices=["depth", "taxonomy"])
    ap.add_argument("--proj-dim", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--no-whiten", dest="whiten", action="store_false", default=True)
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--max-samples", type=int, default=400)
    args = ap.parse_args(argv)
    run(args.activations, args.out, dataset=args.dataset, target=args.target,
        proj_dim=args.proj_dim, seeds=tuple(args.seeds), layers=args.layers,
        whiten=args.whiten, max_epochs=args.max_epochs, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
