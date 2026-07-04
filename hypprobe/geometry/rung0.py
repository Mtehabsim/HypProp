"""Rung 0: the whitening adjudication -- the go/no-go for the whole project.

The question: is the final-layer tree-likeness (Atlas) REAL hierarchy, or just
anisotropy/compression (Park et al. 2025)? We answer it not with a single
whiten-vs-raw number (a strawman, and knob-dependent) but with a METRIC FAMILY
read against CALIBRATION CONTROLS:

  metric family (per layer x token_source):
    raw        -- reproduces the Atlas (no correction)
    pca_only   -- drop empty dims, no rescale (dims vs rescaling probe)
    per_cloud  -- PCA-then-whiten this cloud (strong; biased against real hierarchy)
    background -- remove GENERIC anisotropy fitted from a big pooled sample
                  (the honest middle: strips the shared cone, keeps cloud structure)

  calibration controls (run through the SAME metrics):
    tree_control     -- synthetic tree; MUST stay low-delta under every metric,
                        else whitening is broken and "vanished" is uninterpretable
    gaussian_control -- isotropic Gaussian; MUST stay high-delta,
                        else whitening is MANUFACTURING tree-likeness
    sphere_control   -- positively curved; MUST stay high-delta

Verdict logic (Gate A + Gate B), with thresholds READ FROM PREREGISTER.md so the
call is pre-committed, not chosen after seeing the numbers.

This module is deliberately data-source-agnostic: it runs on the mock store (to
prove the harness recovers a planted answer) and on real DGX activations
(the actual experiment) with no code change.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..data import synthetic
from ..io import (TOKEN_SOURCES, build_token_matrix, ensure_dir, iter_samples,
                  log_line, pool_features, save_csv, save_json)
from .delta import (delta_from_distance_matrix, delta_hyperbolicity, fit_background,
                    _pairwise_euclidean)

METRICS = ("raw", "pca_only", "per_cloud", "background")

# Fallback thresholds; overridden by PREREGISTER.md if present (see _load_thresholds).
DEFAULT_THRESHOLDS = {
    "tree_control_max": 0.15,       # tree must stay below this under every metric (Gate A)
    "gaussian_control_min": 0.15,   # gaussian must stay above this (Gate A, no manufacturing)
    "survive_margin": 0.10,         # data final-layer delta must be this far below its own
                                    # middle-layer delta, under background, to count as real
    "min_effect_over_boot": 2.0,    # effect must exceed this * bootstrap_std to be trusted
}


def _load_thresholds(project_root):
    """Read pre-registered thresholds from PREREGISTER.md JSON block if present."""
    path = os.path.join(project_root, "PREREGISTER.md")
    if not os.path.exists(path):
        return dict(DEFAULT_THRESHOLDS), "defaults (no PREREGISTER.md)"
    txt = open(path).read()
    # thresholds live in a fenced ```json ... ``` block
    if "```json" in txt:
        block = txt.split("```json", 1)[1].split("```", 1)[0]
        try:
            th = json.loads(block)
            merged = dict(DEFAULT_THRESHOLDS)
            merged.update(th.get("rung0", {}))
            return merged, "PREREGISTER.md"
        except Exception:
            pass
    return dict(DEFAULT_THRESHOLDS), "defaults (PREREGISTER.md unparsable)"


def _control_deltas(dim, seed, n_bootstrap):
    """Calibrate the delta ESTIMATOR, not re-ask the research question.

    Key subtlety (and the project's whole premise): a tree does NOT embed in
    Euclidean COORDINATES with low delta -- that is exactly why hyperbolic space
    exists. So a "tree-shaped point cloud" is not a valid low-delta control.
    The honest instrument check is:
      - tree_control  = the true tree DISTANCE MATRIX -> estimator must give ~0
      - gaussian/sphere = isotropic point clouds       -> estimator must give high
    We report tree_control once (metric='distance_matrix'); gaussian/sphere run
    through the point-based estimator (metric='raw') since that is how data is fed.
    """
    rng = np.random.default_rng(seed)
    out = []
    # Tree: exact tree metric -> delta ~ 0 (validates the estimator's low end).
    _, tree_d, _ = synthetic.balanced_tree(depth=5, branching=2, seed=seed)
    tr = delta_from_distance_matrix(tree_d, 2000, 3, np.random.default_rng(seed))
    out.append(("tree_control", "distance_matrix", tr.delta_rel, tr.std_rel))
    # Gaussian + sphere: isotropic -> delta high (validates the estimator's high end,
    # and that whitening does not manufacture tree-likeness).
    for cname, cloud in (("gaussian_control", synthetic.random_gaussian(300, dim, seed)),
                         ("sphere_control", synthetic.sphere_points(300, dim, seed))):
        r = delta_hyperbolicity(cloud, metric="raw", seed=seed, n_repeats=3,
                                n_bootstrap=n_bootstrap)
        out.append((cname, "raw", r.delta_rel, max(r.std_rel, r.bootstrap_std)))
    return out


def _delta_all_metrics(cloud, bg_transform, seed, n_bootstrap, pca_cap):
    out = {}
    for m in METRICS:
        kw = dict(metric=m, seed=seed, n_bootstrap=n_bootstrap, pca_cap=pca_cap)
        if m == "background":
            if bg_transform is None:
                continue
            kw["bg_transform"] = bg_transform
        try:
            r = delta_hyperbolicity(cloud, **kw)
            out[m] = (r.delta_rel, max(r.std_rel, r.bootstrap_std))
        except Exception:
            continue
    return out


def run(activations_dir, out_dir, seed=0, n_bootstrap=25, pca_cap=256,
        project_root="."):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "rung0.log")
    thresholds, th_source = _load_thresholds(project_root)
    log_line(logfile, f"Rung 0 thresholds from: {th_source} -> {thresholds}")

    rows = []
    # --- calibration controls (metric-parity; dim inferred from first data cloud) ---
    models_datasets = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    if not models_datasets:
        log_line(logfile, "no activations found; nothing to adjudicate")
        return []

    for model, dataset in models_datasets:
        sample = next(iter_samples(activations_dir, model, dataset), None)
        if sample is None:
            continue
        n_layers = int(np.asarray(sample["hidden"]).shape[0])
        hidden_dim = int(np.asarray(sample["hidden"]).shape[-1])

        # Fit the GENERIC background transform ONCE per (model) from a big pooled
        # token sample across everything available for this model.
        bg_pool = build_token_matrix(activations_dir, model, dataset,
                                     layer=n_layers - 1, max_tokens=4000)
        bg_transform = fit_background(bg_pool, pca_cap=pca_cap) if bg_pool.shape[0] >= 32 else None

        # Controls: calibrate the estimator (tree distance-matrix ~0; gaussian/
        # sphere high). Not re-run through whitening -- see _control_deltas.
        for cname, cmetric, dr, floor in _control_deltas(min(hidden_dim, 64), seed, n_bootstrap):
            rows.append(dict(model=model, dataset=dataset, layer=-1,
                             token_source="control", metric=cmetric,
                             delta_rel=round(dr, 4), noise_floor=round(floor, 4),
                             cloud_kind=cname))

        # Data: sweep layers x token_source through the metric family.
        for layer in range(n_layers):
            for src in TOKEN_SOURCES:
                xs = []
                for s in iter_samples(activations_dir, model, dataset):
                    v = pool_features(s, layer, src)
                    if v is not None:
                        xs.append(v)
                if len(xs) < 16:
                    continue
                X = np.stack(xs)
                dm = _delta_all_metrics(X, bg_transform, seed, n_bootstrap, pca_cap)
                for metric, (dr, floor) in dm.items():
                    rows.append(dict(model=model, dataset=dataset, layer=layer,
                                     token_source=src, metric=metric,
                                     delta_rel=round(dr, 4), noise_floor=round(floor, 4),
                                     cloud_kind="data"))
        log_line(logfile, f"{model}/{dataset}: adjudicated {n_layers} layers x "
                          f"{len(TOKEN_SOURCES)} sources x {len(METRICS)} metrics")

    save_csv(os.path.join(out_dir, "rung0.csv"), rows,
             columns=["model", "dataset", "layer", "token_source", "metric",
                      "delta_rel", "noise_floor", "cloud_kind"])
    verdict = _verdict(rows, thresholds, th_source)
    with open(os.path.join(out_dir, "rung0_verdict.md"), "w") as fh:
        fh.write(verdict)
    log_line(logfile, "Rung 0 verdict written to rung0_verdict.md")
    print("\n" + verdict)
    return rows


def _plateau_and_final(sub):
    """Return (plateau_delta, final_row, plateau_noise_floor) for a set of
    per-layer rows (one token_source, one metric).

    The plateau is the MEDIAN delta over relative depth 0.35-0.70 (the Atlas's
    plateau definition), not a single midpoint index -- so the "drop" matches the
    Atlas construction on deep (e.g. 28-layer) models. Falls back to the middle
    element if too few layers land in the band.
    """
    import numpy as _np

    layers = sorted(sub, key=lambda r: r["layer"])
    idxs = [r["layer"] for r in layers]
    lo, hi = min(idxs), max(idxs)
    span = max(hi - lo, 1)
    band = [r for r in layers if 0.35 <= (r["layer"] - lo) / span <= 0.70]
    if not band:
        band = [layers[len(layers) // 2]]
    plateau = float(_np.median([r["delta_rel"] for r in band]))
    floor = max([r["noise_floor"] for r in band] + [1e-6])
    return plateau, layers[-1], floor


def _verdict(rows, th, th_source):
    """Gate A (controls behave) + Gate B (data survives) -> plain-English verdict."""
    lines = ["# Rung 0 verdict", "",
             f"Thresholds source: **{th_source}**  ", f"`{json.dumps(th)}`", ""]

    # Gate A: tree control low under every metric; gaussian high under every metric.
    tree = [r for r in rows if r["cloud_kind"] == "tree_control"]
    gauss = [r for r in rows if r["cloud_kind"] == "gaussian_control"]
    tree_ok = tree and all(r["delta_rel"] <= th["tree_control_max"] for r in tree)
    gauss_ok = gauss and all(r["delta_rel"] >= th["gaussian_control_min"] for r in gauss)
    lines += ["## Gate A -- calibration (is whitening trustworthy?)",
              f"- tree_control stays low (<= {th['tree_control_max']}) under every metric: "
              f"**{'PASS' if tree_ok else 'FAIL'}**"
              + ("" if tree_ok else "  -> whitening is broken; 'vanished' is uninterpretable"),
              f"- gaussian_control stays high (>= {th['gaussian_control_min']}): "
              f"**{'PASS' if gauss_ok else 'FAIL'}**"
              + ("" if gauss_ok else "  -> whitening is manufacturing tree-likeness"), ""]

    if not (tree_ok and gauss_ok):
        lines += ["## VERDICT: **CANNOT ADJUDICATE** -- fix whitening/controls first.", ""]
        return "\n".join(lines)

    # Gate B: per (model, token_source), does the final-layer delta under `background`
    # drop below the middle-layer delta by survive_margin AND beat the noise floor?
    lines += ["## Gate B -- does the data's tree-likeness survive `background` whitening?", ""]
    verdicts = []
    keys = sorted({(r["model"], r["token_source"]) for r in rows
                   if r["cloud_kind"] == "data" and r["metric"] == "background"})
    for model, src in keys:
        sub = [r for r in rows if r["cloud_kind"] == "data" and r["metric"] == "background"
               and r["model"] == model and r["token_source"] == src]
        if len(sub) < 3:
            continue
        plateau, final, floor = _plateau_and_final(sub)
        drop = plateau - final["delta_rel"]                 # positive = final more tree-like
        floor = max(floor, final["noise_floor"], 1e-6)
        survives = drop > th["survive_margin"] and drop > th["min_effect_over_boot"] * floor
        # compare to raw: did whitening REMOVE the drop? (raw drop vs background drop)
        raw_sub = [r for r in rows if r["cloud_kind"] == "data" and r["metric"] == "raw"
                   and r["model"] == model and r["token_source"] == src]
        raw_drop = None
        if len(raw_sub) >= 3:
            rp, rf, _ = _plateau_and_final(raw_sub)
            raw_drop = rp - rf["delta_rel"]
        tag = "REAL_HIERARCHY" if survives else "ANISOTROPY_ARTIFACT"
        verdicts.append(tag)
        lines.append(f"- **{model} / {src}**: raw drop={raw_drop if raw_drop is None else round(raw_drop,3)}, "
                     f"background drop={round(drop,3)} (floor {round(floor,3)}) -> **{tag}**")

    lines.append("")
    if verdicts and all(v == "REAL_HIERARCHY" for v in verdicts):
        overall = "REAL_HIERARCHY -> proceed to WHEN/WHY; build the probe."
    elif verdicts and all(v == "ANISOTROPY_ARTIFACT" for v in verdicts):
        overall = ("ANISOTROPY_ARTIFACT -> PIVOT to the deflationary paper "
                   "(Park wins; publish as an Atlas correction). Do NOT build a probe expecting a win.")
    else:
        overall = "MIXED -> proceed only on the (model, source) cells tagged REAL_HIERARCHY."
    lines += [f"## VERDICT: **{overall}**", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rung 0: whitening adjudication (go/no-go).")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=25)
    ap.add_argument("--pca-cap", type=int, default=256)
    args = ap.parse_args(argv)
    run(args.activations, args.out, seed=args.seed, n_bootstrap=args.n_bootstrap,
        pca_cap=args.pca_cap, project_root=args.project_root)


if __name__ == "__main__":
    main()
