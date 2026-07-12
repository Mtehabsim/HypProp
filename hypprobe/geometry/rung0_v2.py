"""Rung 0 v2 — the corrected adjudication (answers the re-evaluation gaps).

What changed vs rung0.py, mapped to the confirmed findings:

  GATE 0 (new, HARD STOP)  — the pre-registered Atlas parity gate is now CODE:
    if the raw within-prompt Atlas-object plateau is far below the theoretical
    ceiling regime the Atlas reports, the verdict is EXTRACTION/OBJECT MISMATCH
    and Gate B is not issued at all. (Finding: "parity gate never implemented;
    verdict issued despite 10x failure".)

  GATE A (fixed)           — controls (flat Gaussian, embedded tree, cluster
    null) now run through the SAME metric family at the SAME (N, k) as each
    data cell, via calibration.calibrated_delta. (Findings: "controls calibrated
    at d=64 while verdicts scored at k=256"; "Gate A ran one metric per
    control".)

  BACKGROUND METRIC (fixed) — the background transform is now fit PER LAYER
    from a token pool BALANCED across token sources and sampled across the
    whole store (reservoir), not final-layer-only / first-13-files-only /
    generated-dominated. (Findings: "background fit on final layer applied
    everywhere manufactures the drop"; "pool is alphabetically-first depth-1
    prompts"; "generated-dominated pool manufactures H1".)

  GATE B (fixed)            — margins are SPAN-RELATIVE (fractions of the
    measured flat<->tree span at the cell's own regime) and the label set now
    distinguishes NO_RAW_EFFECT (the final layer is not more tree-like than the
    plateau even in raw coordinates -> nothing to explain) from
    ANISOTROPY_ARTIFACT (raw effect exists, vanishes under correction) and
    REAL_SIGNAL (survives). A hierarchy reading additionally requires beating
    the cluster null. (Findings: "wrong verdict label when raw drop <= 0";
    "low delta_rel does not mean hierarchy".)

  H1 (fixed)                — scored as a PAIRED per-prompt statistic: for each
    prompt, delta_rel of its own prompt-token cloud vs its own generated-token
    cloud (within-prompt Atlas object, matched token counts by subsampling),
    Wilcoxon signed-rank across prompts. No threshold games: the verdict is a
    sign + p-value + span-relative effect. (Findings: "H1 unpaired, absolute
    margins on a compressed scale"; "different token counts alone shift delta";
    "background pooling manufactures H1".)

Everything writes to results/geometry_v2/ so v1 artifacts remain as the record
of what the first run produced.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..io import TOKEN_SOURCES, _sink_mask, ensure_dir, iter_samples, log_line, save_csv
from ..manifest import write_manifest
from .calibration import calibrated_delta, cluster_null_cloud, embedded_tree_cloud
from .delta import delta_hyperbolicity, fit_background

# Span-relative thresholds (PREREGISTER2.md is the source of truth; these are
# the code defaults that file commits to).
DEFAULTS = {
    "parity_min_plateau": 0.20,      # Gate 0: raw Atlas-object plateau must be at
                                     # least this (theoretical ceiling is 0.29; the
                                     # Atlas's printed ~0.99 is unattainable under
                                     # its own formula — see atlas_forensics.py)
    "span_effect_frac": 0.15,        # an effect is real if > this fraction of the
                                     # cell's own flat<->tree span
    "min_effect_over_floor": 3.0,    # ... AND > this multiple of the noise floor
    "h1_alpha": 0.05,                # Wilcoxon significance for paired H1
    "gate_a_flat_frac": 0.5,         # matched flat anchor must stay in the upper
                                     # half of its own span under every metric
    "gate_a_tree_frac": 0.5,         # matched tree anchor must stay in the lower half
}

METRICS = ("raw", "pca_only", "per_cloud", "background")


def _load_thresholds(project_root):
    path = os.path.join(project_root, "PREREGISTER2.md")
    if os.path.exists(path):
        txt = open(path).read()
        if "```json" in txt:
            block = txt.split("```json", 1)[1].split("```", 1)[0]
            try:
                th = json.loads(block).get("rung0_v2", {})
                merged = dict(DEFAULTS)
                merged.update(th)
                return merged, "PREREGISTER2.md"
            except Exception:
                pass
    return dict(DEFAULTS), "code defaults (write PREREGISTER2.md!)"


# ---------------------------------------------------------------------------
# Balanced background fitting (per layer)
# ---------------------------------------------------------------------------

def fit_balanced_backgrounds(activations_dir, model, dataset, n_layers,
                             per_source_tokens=2000, pca_cap=256, seed=0):
    """Per-layer background transforms from a source-balanced reservoir sample.

    v1 failure modes fixed here:
      - fit per LAYER (anisotropy rotates across layers; a final-layer fit
        deflates the final layer and inflates the others -> manufactured drop);
      - equal token budget for prompt vs generated tokens (a generated-dominated
        pool preferentially flattens the generated cloud -> manufactured H1);
      - reservoir sampling over the WHOLE store (sorted-order filling took only
        the alphabetically-first files = all depth-1 prompts).
    Returns {layer: transform} (missing layers -> None).
    """
    rng = np.random.default_rng(seed)
    # reservoirs[layer][source] -> list of token vectors
    reservoirs = {L: {"input": [], "generated": []} for L in range(n_layers)}
    seen_counts = {L: {"input": 0, "generated": 0} for L in range(n_layers)}

    for s in iter_samples(activations_dir, model, dataset):
        if s.get("variant", "original") != "original":
            continue
        hidden = np.asarray(s["hidden"], dtype=np.float64)
        is_gen = np.asarray(s.get("is_generated"))
        for L in range(min(n_layers, hidden.shape[0])):
            h = hidden[L]
            keep = _sink_mask(h)
            for src, mask in (("input", (~is_gen) & keep), ("generated", is_gen & keep)):
                vecs = h[mask]
                res = reservoirs[L][src]
                for v in vecs:
                    seen_counts[L][src] += 1
                    if len(res) < per_source_tokens:
                        res.append(v)
                    else:
                        j = rng.integers(0, seen_counts[L][src])
                        if j < per_source_tokens:
                            res[j] = v

    transforms = {}
    for L in range(n_layers):
        n_in, n_gen = len(reservoirs[L]["input"]), len(reservoirs[L]["generated"])
        m = min(n_in, n_gen)
        if m < 16:  # can't balance -> fall back to whatever exists
            pool = reservoirs[L]["input"] + reservoirs[L]["generated"]
        else:
            pool = reservoirs[L]["input"][:m] + reservoirs[L]["generated"][:m]
        if len(pool) < 32:
            transforms[L] = None
            continue
        transforms[L] = fit_background(np.stack(pool), pca_cap=pca_cap)
    return transforms


# ---------------------------------------------------------------------------
# Gate 0: Atlas-object parity (raw, within-prompt token clouds)
# ---------------------------------------------------------------------------

def _atlas_object_plateau(activations_dir, model, dataset, max_prompts=100, seed=0):
    """Median raw within-prompt delta_rel over the mid-depth band (the Atlas object)."""
    from .atlas_parity import _atlas_delta_components

    rng = np.random.default_rng(seed)
    per_layer: dict[int, list] = {}
    used = 0
    for s in iter_samples(activations_dir, model, dataset):
        if s.get("variant", "original") != "original":
            continue
        hidden = np.asarray(s["hidden"], dtype=np.float64)
        for L in range(hidden.shape[0]):
            h = hidden[L][_sink_mask(hidden[L])]
            if h.shape[0] < 4:
                continue
            dr_max, _, _, _, _ = _atlas_delta_components(h, 1000, rng)
            per_layer.setdefault(L, []).append(dr_max)
        used += 1
        if used >= max_prompts:
            break
    if not per_layer:
        return float("nan")
    n_layers = max(per_layer) + 1
    band = [np.median(v) for L, v in per_layer.items()
            if 0.35 <= L / max(n_layers - 1, 1) <= 0.70]
    return float(np.median(band)) if band else float("nan")


# ---------------------------------------------------------------------------
# Per-prompt paired H1 (the design fix: pairing beats bigger margins)
# ---------------------------------------------------------------------------

def paired_h1(activations_dir, model, dataset, layer, min_tokens=8,
              n_quadruples=800, seed=0, max_prompts=300):
    """Per-prompt paired delta_rel: prompt-token cloud vs generated-token cloud.

    Both clouds come from the SAME prompt at the SAME layer, in RAW coordinates
    (no cross-cloud transform that could favor one side), with token counts
    MATCHED by subsampling the larger side down to the smaller (token-count
    concentration alone shifts delta_rel — the v1 confound). Returns the list of
    per-prompt (delta_prompt, delta_generated) pairs.
    """
    from .atlas_parity import _atlas_delta_components

    rng = np.random.default_rng(seed)
    pairs = []
    used = 0
    for s in iter_samples(activations_dir, model, dataset):
        if s.get("variant", "original") != "original":
            continue
        hidden = np.asarray(s["hidden"], dtype=np.float64)
        L = min(layer, hidden.shape[0] - 1)
        h = hidden[L]
        is_gen = np.asarray(s.get("is_generated"))
        keep = _sink_mask(h)
        hp = h[(~is_gen) & keep]
        hg = h[is_gen & keep]
        m = min(hp.shape[0], hg.shape[0])
        if m < min_tokens:
            continue
        # matched token counts (subsample the larger cloud)
        if hp.shape[0] > m:
            hp = hp[rng.choice(hp.shape[0], m, replace=False)]
        if hg.shape[0] > m:
            hg = hg[rng.choice(hg.shape[0], m, replace=False)]
        dp, _, _, _, _ = _atlas_delta_components(hp, n_quadruples, rng)
        dg, _, _, _, _ = _atlas_delta_components(hg, n_quadruples, rng)
        if np.isfinite(dp) and np.isfinite(dg):
            pairs.append((dp, dg))
        used += 1
        if used >= max_prompts:
            break
    return pairs


def score_paired_h1(pairs, alpha=0.05):
    """Wilcoxon signed-rank on per-prompt (delta_prompt - delta_generated).

    Positive median diff = generated is MORE tree-like (H1's direction).
    Returns a dict with the effect, test, and a verdict that is a property of
    the data (sign + significance), not of an absolute margin.
    """
    from scipy.stats import wilcoxon

    if len(pairs) < 10:
        return dict(n=len(pairs), verdict="UNDERPOWERED(<10 pairs)")
    diffs = np.array([p - g for p, g in pairs])
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 10:
        return dict(n=len(pairs), verdict="DEGENERATE(all-zero diffs)")
    stat, p = wilcoxon(nonzero)
    med = float(np.median(diffs))
    frac_pos = float((diffs > 0).mean())
    if p < alpha and med > 0:
        verdict = "PASS(generated more tree-like)"
    elif p < alpha and med < 0:
        verdict = "REVERSED(prompt more tree-like)"
    else:
        verdict = "NULL(no paired difference)"
    return dict(n=len(pairs), median_diff=round(med, 4),
                frac_positive=round(frac_pos, 3), wilcoxon_p=float(p),
                verdict=verdict)


# ---------------------------------------------------------------------------
# The v2 run
# ---------------------------------------------------------------------------

def run(activations_dir, out_dir, project_root=".", seed=0, n_bootstrap=25,
        pca_cap=256, max_h1_prompts=300):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                           "logs", "rung0_v2.log")
    th, th_source = _load_thresholds(project_root)
    log_line(logfile, f"rung0_v2 thresholds from {th_source}: {th}")

    rows, h1_rows, verdict_lines = [], [], []
    verdict_lines += ["# Rung 0 v2 verdict", "",
                      f"Thresholds source: **{th_source}**", f"`{json.dumps(th)}`", ""]

    models_datasets = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    if not models_datasets:
        log_line(logfile, "no activations found")
        return []

    for model, dataset in models_datasets:
        sample = next(iter_samples(activations_dir, model, dataset), None)
        n_layers = int(np.asarray(sample["hidden"]).shape[0])

        # ---- GATE 0: Atlas-object parity (hard stop for Gate B) ----
        plateau = _atlas_object_plateau(activations_dir, model, dataset, seed=seed)
        parity_ok = np.isfinite(plateau) and plateau >= th["parity_min_plateau"]
        verdict_lines += [f"## {model} / {dataset}", "",
                          f"**Gate 0 (Atlas-object raw plateau)**: {plateau:.3f} "
                          f"(threshold {th['parity_min_plateau']}; Euclidean-formula ceiling "
                          f"is 0.293 — see atlas_forensics) -> "
                          f"**{'PASS' if parity_ok else 'OBJECT/EXTRACTION MISMATCH'}**", ""]

        # ---- balanced per-layer backgrounds ----
        log_line(logfile, f"{model}/{dataset}: fitting per-layer balanced backgrounds")
        bgs = fit_balanced_backgrounds(activations_dir, model, dataset, n_layers,
                                       pca_cap=pca_cap, seed=seed)

        # ---- pooled per-prompt clouds per (layer, source), single pass ----
        from ..io import pool_features
        feats = {(L, src): [] for L in range(n_layers) for src in TOKEN_SOURCES}
        labels = {(L, src): [] for L in range(n_layers) for src in TOKEN_SOURCES}
        for s in iter_samples(activations_dir, model, dataset):
            if s.get("variant", "original") != "original":
                continue
            for L in range(n_layers):
                for src in TOKEN_SOURCES:
                    v = pool_features(s, L, src)
                    if v is not None:
                        feats[(L, src)].append(v)
                        labels[(L, src)].append(int(s.get("label", 0)))

        # ---- calibrated deltas for every cell (data + matched controls) ----
        gate_a_fail = []
        for L in range(n_layers):
            for src in TOKEN_SOURCES:
                xs = feats[(L, src)]
                if len(xs) < 16:
                    continue
                X = np.stack(xs)
                y = np.asarray(labels[(L, src)])
                for metric in METRICS:
                    bg = bgs.get(L) if metric == "background" else None
                    if metric == "background" and bg is None:
                        continue
                    try:
                        cal = calibrated_delta(X, y, metric=metric, bg_transform=bg,
                                               seed=seed, n_bootstrap=n_bootstrap,
                                               pca_cap=pca_cap)
                    except Exception as exc:
                        log_line(logfile, f"  cell ({L},{src},{metric}) failed: {exc}")
                        continue
                    rows.append(dict(
                        model=model, dataset=dataset, layer=L, token_source=src,
                        metric=metric,
                        delta_data=round(cal.delta_data, 4),
                        delta_flat=round(cal.delta_flat, 4),
                        delta_tree=round(cal.delta_tree, 4),
                        delta_cluster=(round(cal.delta_cluster, 4)
                                       if np.isfinite(cal.delta_cluster) else ""),
                        span=round(cal.span, 4),
                        score=round(cal.score, 4) if np.isfinite(cal.score) else "",
                        excess_over_cluster=(round(cal.excess_over_cluster, 4)
                                             if np.isfinite(cal.excess_over_cluster) else ""),
                        noise_floor=round(cal.noise_floor, 4),
                        n=cal.n_points, dim=cal.ambient_dim,
                    ))
                    # Gate A, regime-matched: the matched anchors must keep the
                    # span open under EVERY metric — flat in the upper half,
                    # tree in the lower half of the span.
                    if cal.span <= 0:
                        gate_a_fail.append((L, src, metric, "span collapsed (flat <= tree)"))
            if L % 5 == 0:
                log_line(logfile, f"  {model}/{dataset}: layer {L}/{n_layers} calibrated")

        ga_ok = not gate_a_fail
        verdict_lines += [f"**Gate A (regime-matched anchors keep the span open under every "
                          f"metric)**: **{'PASS' if ga_ok else 'FAIL'}**"]
        for (L, src, metric, why) in gate_a_fail[:8]:
            verdict_lines.append(f"  - L{L}/{src}/{metric}: {why}")
        verdict_lines.append("")

        # ---- Gate B, span-relative, with the corrected label set ----
        sub_rows = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
        verdict_lines += ["**Gate B (span-relative drop, per source x metric)**", ""]
        for src in TOKEN_SOURCES:
            raw_sig = False  # set by the 'raw' pass; False if raw cells missing
            for metric in ("raw", "background", "per_cloud"):
                cells = sorted([r for r in sub_rows if r["token_source"] == src
                                and r["metric"] == metric and r["score"] != ""],
                               key=lambda r: r["layer"])
                if len(cells) < 3:
                    continue
                lo, hi = cells[0]["layer"], cells[-1]["layer"]
                span_layers = max(hi - lo, 1)
                band = [r for r in cells if 0.35 <= (r["layer"] - lo) / span_layers <= 0.70]
                if not band:
                    band = [cells[len(cells) // 2]]
                plateau_score = float(np.median([r["score"] for r in band]))
                final = cells[-1]
                drop = final["score"] - plateau_score   # score UP = more tree-like
                floor = max(final["noise_floor"] / max(final["span"], 1e-9), 1e-6)
                significant = (drop > th["span_effect_frac"]
                               and drop > th["min_effect_over_floor"] * floor)
                beats_cluster = (final["excess_over_cluster"] != ""
                                 and float(final["excess_over_cluster"])
                                 > th["min_effect_over_floor"] * final["noise_floor"])
                if metric == "raw":
                    raw_sig = significant  # remember for the label logic
                    label = ("RAW_EFFECT" if significant else "NO_RAW_EFFECT")
                else:
                    if not raw_sig:
                        label = "NO_RAW_EFFECT(nothing to adjudicate)"
                    elif significant and beats_cluster:
                        label = "REAL_SIGNAL(beats cluster null)"
                    elif significant:
                        label = "CLUSTER_STRUCTURE(not hierarchy)"
                    else:
                        label = "ANISOTROPY_ARTIFACT(vanished under correction)"
                verdict_lines.append(
                    f"- {src}/{metric}: plateau_score={plateau_score:+.3f} "
                    f"final_score={final['score']:+.3f} drop={drop:+.3f} "
                    f"(floor {floor:.3f}, beats_cluster={beats_cluster}) -> **{label}**")
        verdict_lines.append("")

        # ---- paired per-prompt H1 (final layer) ----
        h1 = score_paired_h1(paired_h1(activations_dir, model, dataset,
                                       layer=n_layers - 1, seed=seed,
                                       max_prompts=max_h1_prompts),
                             alpha=th["h1_alpha"])
        h1["model"] = model
        h1["dataset"] = dataset
        h1_rows.append(h1)
        verdict_lines += [f"**H1 paired (per-prompt, matched token counts, raw, "
                          f"final layer)**: {h1}", ""]

    # ---- H2 v2: direction consistency of the paired effect across model pairs ----
    verdict_lines += ["## H2 v2 (direction consistency across models)", ""]
    for h in h1_rows:
        verdict_lines.append(f"- {h['model']}: {h.get('verdict','?')} "
                             f"(median_diff={h.get('median_diff','-')}, "
                             f"p={h.get('wilcoxon_p','-')})")
    verdict_lines += ["", "H2 requires >= 2 reasoning/base model PAIRS showing the same",
                      "ordering of median_diff; with a single pair report direction only,",
                      "make NO tuning-causes-it claim (the v1 design error).", ""]

    save_csv(os.path.join(out_dir, "rung0_v2.csv"), rows)
    save_csv(os.path.join(out_dir, "h1_paired_v2.csv"), h1_rows)
    with open(os.path.join(out_dir, "rung0_v2_verdict.md"), "w") as fh:
        fh.write("\n".join(verdict_lines) + "\n")
    write_manifest(out_dir, "rung0_v2",
                   args=dict(activations=activations_dir, seed=seed,
                             n_bootstrap=n_bootstrap, pca_cap=pca_cap),
                   extra=dict(thresholds=th, thresholds_source=th_source))
    log_line(logfile, "rung0_v2 verdict written")
    print("\n".join(verdict_lines))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rung 0 v2: corrected adjudication.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry_v2")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=25)
    ap.add_argument("--pca-cap", type=int, default=256)
    ap.add_argument("--max-h1-prompts", type=int, default=300)
    args = ap.parse_args(argv)
    run(args.activations, args.out, project_root=args.project_root, seed=args.seed,
        n_bootstrap=args.n_bootstrap, pca_cap=args.pca_cap,
        max_h1_prompts=args.max_h1_prompts)


if __name__ == "__main__":
    main()
