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

PARITY-MODE DIAGNOSTICS (added after the sink fix still did NOT reproduce the
plateau): the sink strip removed the delta~0.001 degeneracy but middle delta
came out ~0.08 (Atlas ~0.99). So there is STILL a gap. To localize it, this now
reports, per layer, ALONGSIDE delta:
  - participation_ratio (effective rank) of the stripped cloud -- is it actually
    a clean multi-D cloud (~10-50), or still near-1-D (residual massive tokens)?
  - n_tok_after_strip and n_stripped -- did we over/under-strip?
  - the defect vs diameter SCALE (median), so we can see WHETHER delta is low
    because the defect is small (genuinely tree-like) or the diameter is still
    inflated (residual outlier). delta_rel = defect / diameter.
  - delta under BOTH max-diam (Atlas-exact) and p99-diam, to confirm the choice
    of diameter is NOT what separates us from the Atlas.

It is a DIAGNOSTIC -- it does not modify rung0 or the verdicts. Run it on the
saved activations (no re-extraction).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import _sink_mask, ensure_dir, iter_samples, save_csv


def _participation_ratio(points: np.ndarray) -> float:
    """Effective rank of the (centred) cloud: (sum s^2)^2 / sum s^4.

    ~1 means the cloud is essentially 1-D (a single direction dominates -- the
    degenerate / massive-activation regime); ~n means isotropic. After a correct
    sink strip this should leave 1.0 and sit in the tens.
    """
    if points.shape[0] < 2:
        return float("nan")
    xc = points - points.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    e = s ** 2
    tot = float(e.sum())
    if tot <= 0:
        return float("nan")
    return float((e.sum() ** 2) / (e ** 2).sum())


def _atlas_delta_components(points: np.ndarray, n_quadruples: int, rng):
    """Return (delta_rel_maxdiam, delta_rel_p99diam, defect_max, diam_max, diam_p99).

    Atlas-faithful: sum-form four-point defect delta = 1/2 (s3 - s2), aggregated
    as the MAX over sampled quadruples, normalized by the MAX pairwise distance
    (their Eq. 6). We ALSO return the p99-diameter variant and the raw scales so
    the caller can see why delta_rel is what it is.
    """
    n = points.shape[0]
    if n < 4:
        return (float("nan"),) * 5
    sq = np.sum(points * points, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    np.maximum(d2, 0.0, out=d2)
    dmat = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    diam_max = float(dmat.max())                       # Atlas: max
    diam_p99 = float(np.percentile(dmat[iu], 99.0))
    if diam_max <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_quadruples, 4))
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    s1 = dmat[a, b] + dmat[c, d]
    s2 = dmat[a, c] + dmat[b, d]
    s3 = dmat[a, d] + dmat[b, c]
    S = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)   # ascending
    defect = 0.5 * (S[:, 2] - S[:, 1])                    # 1/2 (s3 - s2)
    defect_max = float(defect.max())                      # Atlas uses max defect
    dr_max = defect_max / diam_max
    dr_p99 = defect_max / diam_p99 if diam_p99 > 0 else float("nan")
    return dr_max, dr_p99, defect_max, diam_max, diam_p99


def run(activations_dir, out_dir, n_quadruples=2000, max_prompts=200, seed=0,
        drop_sink=True):
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        # collect per-prompt, per-layer diagnostics (Atlas object)
        acc: dict[int, dict] = {}
        n_used = 0
        for s in iter_samples(activations_dir, model, dataset):
            hidden = np.asarray(s["hidden"], dtype=np.float64)   # (n_layers, n_tok, h)
            n_layers, n_tok, _ = hidden.shape
            if n_tok < 4:
                continue
            for L in range(n_layers):
                hL_full = hidden[L]
                keep = _sink_mask(hL_full) if drop_sink else np.ones(hL_full.shape[0], bool)
                hL = hL_full[keep]
                n_stripped = int((~keep).sum())
                dr_max, dr_p99, defect_max, diam_max, diam_p99 = _atlas_delta_components(
                    hL, n_quadruples, rng)
                pr = _participation_ratio(hL)
                a = acc.setdefault(L, {k: [] for k in
                                       ("dr_max", "dr_p99", "pr", "n_tok", "n_strip",
                                        "defect", "diam_max", "diam_p99")})
                a["dr_max"].append(dr_max); a["dr_p99"].append(dr_p99); a["pr"].append(pr)
                a["n_tok"].append(hL.shape[0]); a["n_strip"].append(n_stripped)
                a["defect"].append(defect_max); a["diam_max"].append(diam_max)
                a["diam_p99"].append(diam_p99)
            n_used += 1
            if n_used >= max_prompts:
                break
        n_layers = max(acc) + 1 if acc else 0

        def med(xs):
            xs = [v for v in xs if v is not None and not (isinstance(v, float) and np.isnan(v))]
            return round(float(np.median(xs)), 4) if xs else float("nan")

        for L in range(n_layers):
            a = acc.get(L)
            if not a or not a["dr_max"]:
                continue
            rows.append(dict(
                model=model, dataset=dataset, layer=L,
                rel_depth=round(L / max(n_layers - 1, 1), 3),
                median_delta_rel=med(a["dr_max"]),          # Atlas-exact (max diam)
                delta_rel_p99diam=med(a["dr_p99"]),
                participation_ratio=med(a["pr"]),
                n_tok_after_strip=med(a["n_tok"]),
                n_stripped=med(a["n_strip"]),
                defect_max=med(a["defect"]),
                diam_max=med(a["diam_max"]),
                diam_p99=med(a["diam_p99"]),
                n_prompts=len(a["dr_max"]),
            ))

        # readable plateau/final + parity-diagnostic summary
        sub = [r for r in rows if r["model"] == model]
        if sub:
            layers = sorted(r["layer"] for r in sub)
            drs = {r["layer"]: r for r in sub}
            band = [drs[L]["median_delta_rel"] for L in layers if 0.35 <= L / max(layers) <= 0.70]
            plateau = float(np.median(band)) if band else float("nan")
            fin = drs[layers[-1]]
            midrow = drs[[L for L in layers if 0.35 <= L / max(layers) <= 0.70][len(band)//2]] if band else fin
            print(f"\n[{model}] Atlas-object delta_rel (within-prompt token clouds, sink-stripped):")
            print(f"   middle-plateau median = {plateau:.3f}   final-layer = {fin['median_delta_rel']:.3f}")
            print(f"   -> Atlas reports ~0.99 plateau dropping to ~0.5. "
                  f"{'REPRODUCED' if plateau > 0.8 else 'NOT reproduced'}")
            print(f"   [middle-layer L{midrow['layer']} diagnostics]")
            print(f"     participation_ratio (eff. rank) = {midrow['participation_ratio']}  "
                  f"(~1 => STILL degenerate/1-D; tens => clean multi-D cloud)")
            print(f"     n_tok_after_strip = {midrow['n_tok_after_strip']}   "
                  f"n_stripped = {midrow['n_stripped']}")
            print(f"     defect_max = {midrow['defect_max']}   diam_max = {midrow['diam_max']}   "
                  f"diam_p99 = {midrow['diam_p99']}")
            print(f"     delta_rel: max-diam = {midrow['median_delta_rel']}   "
                  f"p99-diam = {midrow['delta_rel_p99diam']}  "
                  f"(if these differ a lot, a residual outlier still inflates max-diam)")
            # interpretation
            msgs = []
            if midrow['participation_ratio'] and midrow['participation_ratio'] < 2.5:
                msgs.append("cloud STILL ~1-D after strip => residual massive-activation tokens; "
                            "lower the sink multiplier or strip more")
            if (midrow['diam_max'] and midrow['diam_p99']
                    and midrow['diam_max'] > 3 * midrow['diam_p99']):
                msgs.append("max-diam >> p99-diam => a residual outlier still sets the diameter")
            if not msgs and plateau < 0.8:
                msgs.append("cloud looks CLEAN yet delta is low => the gap to the Atlas is NOT a "
                            "residual-sink artifact; likely a genuine object/dataset difference "
                            "(PrOntoQA vs their data) or a real finding -- investigate, don't assume")
            print("     PARITY DIAGNOSIS: " + ("; ".join(msgs) if msgs else "clean; plateau reproduced"))

    save_csv(os.path.join(out_dir, "atlas_parity.csv"), rows,
             columns=["model", "dataset", "layer", "rel_depth", "median_delta_rel",
                      "delta_rel_p99diam", "participation_ratio", "n_tok_after_strip",
                      "n_stripped", "defect_max", "diam_max", "diam_p99", "n_prompts"])
    print(f"\nwrote {out_dir}/atlas_parity.csv")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnostic: Atlas-object delta parity check + "
                                             "parity-mode diagnostics (participation ratio, "
                                             "strip counts, defect/diam scale).")
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
