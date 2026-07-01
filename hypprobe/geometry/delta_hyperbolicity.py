"""Phase 1 CLI: map delta_rel + label alignment across (model, layer, token-source).

Reads the activation store, pools features for every (layer, token-source), and
computes:
  - delta_rel (whitened, normalised, with variance) -- is the point cloud tree-like?
  - label alignment -- does the TAXONOMY embed as a tree here (Euclidean vs
    hyperbolic prototype-distance correlation) and does norm encode depth?

Together these implement the plan's JOINT selection criterion: the best setting
has BOTH low delta_rel AND high hyperbolic label alignment. Writes
``geometry/delta_rel.csv`` and a per-layer plot, and logs the joint winner.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import (TOKEN_SOURCES, build_feature_matrix, ensure_dir, iter_samples,
                  log_line, save_csv)
from .delta import delta_hyperbolicity
from .label_alignment import label_alignment


def _models_datasets(activations_dir):
    seen = set()
    for s in iter_samples(activations_dir):
        seen.add((s["model"], s["dataset"]))
    return sorted(seen)


def run(activations_dir, out_dir, whiten=True, n_layers_hint=None, seed=0):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "delta.log")
    rows = []
    for model, dataset in _models_datasets(activations_dir):
        # Determine layer count from one sample.
        sample = next(iter_samples(activations_dir, model, dataset), None)
        if sample is None:
            continue
        n_layers = int(np.asarray(sample["hidden"]).shape[0])
        layers = range(n_layers) if n_layers_hint is None else n_layers_hint
        for layer in layers:
            for src in TOKEN_SOURCES:
                X, y, paths = build_feature_matrix(activations_dir, model, dataset, layer, src)
                if X.shape[0] < 8:
                    continue
                res = delta_hyperbolicity(X, n_quadruples=1500, n_repeats=5,
                                          do_whiten=whiten, seed=seed)
                # Label alignment: does the taxonomy embed as a tree here?
                align = label_alignment(X, paths, curvature=1.0, do_whiten=whiten)
                # Joint score: low delta_rel AND high hyperbolic alignment. We
                # combine as (1 - delta_rel) + hyperbolic prototype correlation,
                # so higher = better on both axes (used only for logging/ranking).
                joint = round((1.0 - res.delta_rel) + align.proto_corr_hyperbolic, 4)
                rows.append(dict(model=model, dataset=dataset, layer=layer,
                                 token_source=src, delta_rel=round(res.delta_rel, 4),
                                 std_rel=round(res.std_rel, 4), diam=round(res.diam, 3),
                                 n_points=res.n_points,
                                 align_euc=round(align.proto_corr_euclidean, 4),
                                 align_hyp=round(align.proto_corr_hyperbolic, 4),
                                 norm_depth_corr=round(align.norm_depth_corr, 4),
                                 joint_score=joint))
        log_line(logfile, f"{model}/{dataset}: mapped {n_layers} layers x {len(TOKEN_SOURCES)} sources")

    csv_path = os.path.join(out_dir, "delta_rel.csv")
    save_csv(csv_path, rows, columns=["model", "dataset", "layer", "token_source",
                                      "delta_rel", "std_rel", "diam", "n_points",
                                      "align_euc", "align_hyp", "norm_depth_corr",
                                      "joint_score"])
    if rows:
        best_delta = min(rows, key=lambda r: r["delta_rel"])
        log_line(logfile, f"most hyperbolic (delta_rel) setting: {best_delta['model']} "
                          f"L{best_delta['layer']} {best_delta['token_source']} "
                          f"delta_rel={best_delta['delta_rel']}")
        best_joint = max(rows, key=lambda r: r["joint_score"])
        log_line(logfile, f"best JOINT setting (low delta_rel + taxonomy fit): "
                          f"{best_joint['model']} L{best_joint['layer']} "
                          f"{best_joint['token_source']} "
                          f"(delta_rel={best_joint['delta_rel']}, "
                          f"align_hyp={best_joint['align_hyp']}, "
                          f"align_euc={best_joint['align_euc']})")
    _maybe_plot(rows, out_dir)
    return rows


def _maybe_plot(rows, out_dir):
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    keys = sorted({(r["model"], r["token_source"]) for r in rows})
    for model, src in keys:
        pts = sorted([(r["layer"], r["delta_rel"]) for r in rows
                      if r["model"] == model and r["token_source"] == src])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=f"{model.split('/')[-1]}:{src}")
    ax.set_xlabel("layer"); ax.set_ylabel("delta_rel (0=tree, ~1=flat)")
    ax.set_title("Hyperbolicity by layer and token source"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "delta_by_layer.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1: delta-hyperbolicity map.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry")
    ap.add_argument("--whiten", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, whiten=args.whiten, seed=args.seed)


if __name__ == "__main__":
    main()
