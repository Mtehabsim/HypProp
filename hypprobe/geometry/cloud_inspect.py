"""Diagnostic (c): WHY do within-prompt token clouds score delta ~ 0 in the
middle layers when the Atlas gets ~0.99? Localize the cause per prompt.

We proved (atlas_parity.py) that fixing the object AND using the exact Atlas
sum-form formula still gives an inverted, ~1000x-deflated middle-layer delta.
So the cause is neither the pooling object nor the defect formula. This script
inspects ONE prompt's token cloud at a few layers and tests the two remaining
hypotheses:

  H-SINK  (attention-sink / BOS outlier): one token (usually position 0) has a
          massive-norm hidden state, strongest in the MIDDLE layers, relaxing at
          the final layer. The Atlas normalizes by the MAX pairwise distance, so
          a single huge-norm token inflates the diameter -> delta_rel collapses
          toward 0. This EXACTLY predicts the inverted shape we see (mid~0,
          final higher). TEST: does delta jump up when we (a) drop the top-norm
          token, or (b) normalize by the 99th-percentile diameter instead of max?

  H-1D    (near-1-D cloud): a short prompt's token states lie near a line/curve
          in the middle layers (positional drift), and a path is a tree -> delta~0
          for a boring reason. TEST: is the top singular value's energy fraction
          near 1 / the participation ratio (effective rank) near 1 in the middle
          layers?

Output: per (model, layer) medians across the first K prompts of:
  n_tok, max/median token-norm ratio, top-1 SV energy fraction, participation
  ratio (effective rank), and delta_rel under 4 settings
  {all|drop-topnorm} x {max-diam|p99-diam}.

Reads the saved .pt activations. Modifies nothing. Run on the DGX:
  python -m hypprobe.geometry.cloud_inspect --activations ./results/activations \
    --out ./results/geometry --max-prompts 100
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import ensure_dir, iter_samples, save_csv


def _pairwise(points: np.ndarray) -> np.ndarray:
    sq = np.sum(points * points, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _atlas_delta_from_dmat(dmat: np.ndarray, diam: float, n_quadruples: int, rng) -> float:
    """Atlas sum-form four-point defect (max defect), normalized by the given diam."""
    n = dmat.shape[0]
    if n < 4 or diam <= 0:
        return float("nan")
    idx = rng.integers(0, n, size=(n_quadruples, 4))
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    s1 = dmat[a, b] + dmat[c, d]
    s2 = dmat[a, c] + dmat[b, d]
    s3 = dmat[a, d] + dmat[b, c]
    S = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)   # ascending
    defect = 0.5 * (S[:, 2] - S[:, 1])                    # 1/2 (s3 - s2)
    return float(defect.max()) / diam


def _svd_stats(X: np.ndarray):
    """Return (top1_energy_fraction, participation_ratio) for centred X.

    participation_ratio = (sum s^2)^2 / sum s^4  -- an 'effective rank': ~1 means
    the cloud is essentially 1-D, ~n means isotropic.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    e = s ** 2
    tot = float(e.sum())
    if tot <= 0:
        return float("nan"), float("nan")
    top1 = float(e[0] / tot)
    pr = float((e.sum() ** 2) / (e ** 2).sum())
    return top1, pr


def _delta_variants(X: np.ndarray, n_quadruples: int, rng):
    """delta_rel under {all|drop-topnorm} x {max-diam|p99-diam}."""
    norms = np.linalg.norm(X - X.mean(axis=0, keepdims=True), axis=1)
    # drop the single highest-norm token (the sink candidate)
    keep = np.ones(X.shape[0], bool)
    keep[int(np.argmax(np.linalg.norm(X, axis=1)))] = False   # highest RAW norm (sink)
    out = {}
    for label, pts in (("all", X), ("drop_topnorm", X[keep])):
        if pts.shape[0] < 4:
            out[f"{label}_maxdiam"] = float("nan")
            out[f"{label}_p99diam"] = float("nan")
            continue
        dmat = _pairwise(pts)
        iu = np.triu_indices(pts.shape[0], k=1)
        diam_max = float(dmat.max())
        diam_p99 = float(np.percentile(dmat[iu], 99.0))
        out[f"{label}_maxdiam"] = _atlas_delta_from_dmat(dmat, diam_max, n_quadruples, rng)
        out[f"{label}_p99diam"] = _atlas_delta_from_dmat(dmat, diam_p99, n_quadruples, rng)
    return out, norms


def run(activations_dir, out_dir, n_quadruples=2000, max_prompts=100, seed=0):
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        # figure out layer count + which layers to inspect
        first = next(iter_samples(activations_dir, model, dataset), None)
        if first is None:
            continue
        n_layers = int(np.asarray(first["hidden"]).shape[0])
        # embedding, ~40/50/60% (middle band), and final
        probe_layers = sorted(set([0,
                                   int(round(0.40 * (n_layers - 1))),
                                   int(round(0.50 * (n_layers - 1))),
                                   int(round(0.60 * (n_layers - 1))),
                                   n_layers - 1]))

        acc = {L: {"n_tok": [], "norm_ratio": [], "top1": [], "pr": [],
                   "all_maxdiam": [], "all_p99diam": [],
                   "drop_topnorm_maxdiam": [], "drop_topnorm_p99diam": [],
                   "topnorm_pos": [], "topnorm_is_first": []} for L in probe_layers}
        n_used = 0
        for s in iter_samples(activations_dir, model, dataset):
            hidden = np.asarray(s["hidden"], dtype=np.float64)   # (n_layers, n_tok, h)
            n_tok = hidden.shape[1]
            if n_tok < 4:
                continue
            for L in probe_layers:
                X = hidden[L]
                raw_norms = np.linalg.norm(X, axis=1)
                top_pos = int(np.argmax(raw_norms))
                med = float(np.median(raw_norms))
                acc[L]["n_tok"].append(n_tok)
                acc[L]["norm_ratio"].append(float(raw_norms.max() / med) if med > 0 else float("nan"))
                top1, pr = _svd_stats(X)
                acc[L]["top1"].append(top1)
                acc[L]["pr"].append(pr)
                dv, _ = _delta_variants(X, n_quadruples, rng)
                for k, v in dv.items():
                    acc[L][k].append(v)
                acc[L]["topnorm_pos"].append(top_pos)
                acc[L]["topnorm_is_first"].append(1.0 if top_pos == 0 else 0.0)
            n_used += 1
            if n_used >= max_prompts:
                break

        def med(xs):
            xs = [v for v in xs if v is not None and not (isinstance(v, float) and np.isnan(v))]
            return round(float(np.median(xs)), 4) if xs else float("nan")

        for L in probe_layers:
            a = acc[L]
            rows.append(dict(
                model=model, dataset=dataset, layer=L,
                rel_depth=round(L / max(n_layers - 1, 1), 3),
                n_tok=med(a["n_tok"]),
                norm_ratio_maxovermed=med(a["norm_ratio"]),
                top1_sv_energy=med(a["top1"]),
                participation_ratio=med(a["pr"]),
                topnorm_is_first_frac=med(a["topnorm_is_first"]),
                delta_all_maxdiam=med(a["all_maxdiam"]),
                delta_all_p99diam=med(a["all_p99diam"]),
                delta_droptop_maxdiam=med(a["drop_topnorm_maxdiam"]),
                delta_droptop_p99diam=med(a["drop_topnorm_p99diam"]),
                n_prompts=len([v for v in a["n_tok"]]),
            ))

        # readable per-model verdict
        mid = [r for r in rows if r["model"] == model and 0.35 <= r["rel_depth"] <= 0.70]
        if mid:
            m = mid[len(mid) // 2]
            print(f"\n[{model}]  (median over {m['n_prompts']} prompts, middle layer L{m['layer']})")
            print(f"   n_tok={m['n_tok']}  max/median token-norm ratio={m['norm_ratio_maxovermed']}  "
                  f"(top-norm token is position 0 in {m['topnorm_is_first_frac']*100:.0f}% of prompts)")
            print(f"   top-1 SV energy={m['top1_sv_energy']}  participation_ratio(eff.rank)={m['participation_ratio']}")
            print(f"   delta_rel:  all/maxdiam={m['delta_all_maxdiam']}   all/p99diam={m['delta_all_p99diam']}")
            print(f"               drop-topnorm/maxdiam={m['delta_droptop_maxdiam']}   "
                  f"drop-topnorm/p99diam={m['delta_droptop_p99diam']}")
            # interpret
            msgs = []
            if m["norm_ratio_maxovermed"] and m["norm_ratio_maxovermed"] > 5:
                msgs.append(f"SINK: one token has {m['norm_ratio_maxovermed']:.0f}x the median norm")
            if m["delta_droptop_maxdiam"] and m["delta_all_maxdiam"] and \
               m["delta_droptop_maxdiam"] > m["delta_all_maxdiam"] + 0.2:
                msgs.append("-> dropping it RAISES delta a lot => H-SINK confirmed (max-diam driven by outlier)")
            if m["delta_all_p99diam"] and m["delta_all_maxdiam"] and \
               m["delta_all_p99diam"] > m["delta_all_maxdiam"] + 0.2:
                msgs.append("-> p99-diameter RAISES delta a lot => H-SINK confirmed (max-diam is outlier-inflated)")
            if m["participation_ratio"] and m["participation_ratio"] < 2.5:
                msgs.append(f"1-D: effective rank ~{m['participation_ratio']:.1f} => H-1D also in play")
            print("   VERDICT: " + ("; ".join(msgs) if msgs else "no clear sink/1-D signature -- investigate further"))

    save_csv(os.path.join(out_dir, "cloud_inspect.csv"), rows,
             columns=["model", "dataset", "layer", "rel_depth", "n_tok",
                      "norm_ratio_maxovermed", "top1_sv_energy", "participation_ratio",
                      "topnorm_is_first_frac", "delta_all_maxdiam", "delta_all_p99diam",
                      "delta_droptop_maxdiam", "delta_droptop_p99diam", "n_prompts"])
    print(f"\nwrote {out_dir}/cloud_inspect.csv")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Diagnostic: why do within-prompt token clouds score delta~0? "
                    "(attention-sink/max-diameter vs near-1-D cloud)")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry")
    ap.add_argument("--n-quadruples", type=int, default=2000)
    ap.add_argument("--max-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, n_quadruples=args.n_quadruples,
        max_prompts=args.max_prompts, seed=args.seed)


if __name__ == "__main__":
    main()
