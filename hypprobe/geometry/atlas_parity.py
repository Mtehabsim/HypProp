"""Diagnostic (b): measure delta the ATLAS way and check the ~0.99 plateau.

Purpose: confirm that the invalid rung0 verdict is caused by measuring the WRONG
geometric object (pooled per-prompt vectors) rather than anything else.

The Atlas (Eq. 7) measures delta over the token cloud WITHIN a single prompt --
i.e. for one prompt at one layer, the points are that prompt's own token hidden
states -- then aggregates (median) across prompts. That is completely different
from rung0's current object (one pooled vector per prompt, delta over the
cross-prompt cloud).

This script computes the Atlas object, RAW, with Atlas-faithful normalization
(delta / max-diameter), per layer, median across prompts. If the middle layers
show a high plateau (~0.9-0.99) dropping at the final layer, we have reproduced
the Atlas -> object mismatch confirmed as the whole story.

It is a DIAGNOSTIC -- it does not modify rung0 or the verdicts. Run it on the
saved activations (no re-extraction).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import _sink_mask, ensure_dir, iter_samples, save_csv


def _atlas_delta_rel(points: np.ndarray, n_quadruples: int, rng) -> float:
    """Atlas-faithful delta_rel for ONE cloud: sum-form four-point defect,
    normalized by the MAX pairwise distance (their Eq. 6). No whitening."""
    n = points.shape[0]
    if n < 4:
        return float("nan")
    # pairwise Euclidean distance matrix
    sq = np.sum(points * points, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    np.maximum(d2, 0.0, out=d2)
    dmat = np.sqrt(d2)
    diam = float(dmat.max())            # Atlas: max, not percentile
    if diam <= 0:
        return 0.0
    idx = rng.integers(0, n, size=(n_quadruples, 4))
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    s1 = dmat[a, b] + dmat[c, d]
    s2 = dmat[a, c] + dmat[b, d]
    s3 = dmat[a, d] + dmat[b, c]
    S = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)   # ascending
    defect = 0.5 * (S[:, 2] - S[:, 1])                    # 1/2 (s3 - s2)
    delta = float(defect.max())                           # Atlas uses max defect
    return delta / diam


def run(activations_dir, out_dir, n_quadruples=2000, max_prompts=200, seed=0,
        drop_sink=True):
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        # collect per-prompt delta_rel at each layer (Atlas object)
        per_layer: dict[int, list] = {}
        n_used = 0
        for s in iter_samples(activations_dir, model, dataset):
            hidden = np.asarray(s["hidden"], dtype=np.float64)   # (n_layers, n_tok, h)
            n_layers, n_tok, _ = hidden.shape
            if n_tok < 4:
                continue
            for L in range(n_layers):
                hL = hidden[L]
                if drop_sink:
                    hL = hL[_sink_mask(hL)]          # strip attention-sink tokens
                dr = _atlas_delta_rel(hL, n_quadruples, rng)
                per_layer.setdefault(L, []).append(dr)
            n_used += 1
            if n_used >= max_prompts:
                break
        n_layers = max(per_layer) + 1 if per_layer else 0
        for L in range(n_layers):
            vals = np.array([v for v in per_layer.get(L, []) if not np.isnan(v)])
            if vals.size == 0:
                continue
            rows.append(dict(model=model, dataset=dataset, layer=L,
                             rel_depth=round(L / max(n_layers - 1, 1), 3),
                             median_delta_rel=round(float(np.median(vals)), 4),
                             n_prompts=int(vals.size)))
        # quick plateau/final summary
        drs = {r["layer"]: r["median_delta_rel"] for r in rows if r["model"] == model}
        if drs:
            layers = sorted(drs)
            band = [drs[L] for L in layers if 0.35 <= L / max(layers) <= 0.70]
            plateau = float(np.median(band)) if band else float("nan")
            final = drs[layers[-1]]
            print(f"\n[{model}] Atlas-object delta_rel (within-prompt token clouds):")
            print(f"   middle-plateau median = {plateau:.3f}   final-layer = {final:.3f}")
            print(f"   -> Atlas reports ~0.99 plateau dropping to ~0.5. "
                  f"{'REPRODUCED' if plateau > 0.8 else 'NOT reproduced'}")
    save_csv(os.path.join(out_dir, "atlas_parity.csv"), rows,
             columns=["model", "dataset", "layer", "rel_depth",
                      "median_delta_rel", "n_prompts"])
    print(f"\nwrote {out_dir}/atlas_parity.csv")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnostic: Atlas-object delta parity check.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry")
    ap.add_argument("--n-quadruples", type=int, default=2000)
    ap.add_argument("--max-prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-sink", action="store_true",
                    help="do NOT strip attention-sink tokens (shows the degeneracy)")
    args = ap.parse_args(argv)
    run(args.activations, args.out, n_quadruples=args.n_quadruples,
        max_prompts=args.max_prompts, seed=args.seed, drop_sink=not args.keep_sink)


if __name__ == "__main__":
    main()
