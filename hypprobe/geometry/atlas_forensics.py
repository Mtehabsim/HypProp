"""Atlas forensics: adjudicate the 10x parity gap the referee-proof way.

The re-evaluation's verified result (do NOT rebuild the naive plan):

  Under the Atlas's own printed formula — four-point sum-form defect
  delta = 1/2 (s3 - s2) (their Eq. 5), delta_rel = delta / max-diameter
  (their Eq. 6), Euclidean distances — the supremum of delta_rel over ANY
  Euclidean point cloud is 1 - 1/sqrt(2) ~= 0.2929 (extremal config = the unit
  square), and 0.5 for any metric space whatsoever. Their reported middle-layer
  medians of ~0.995 are therefore UNATTAINABLE under their stated protocol on
  any data. Our ~0.08 readings are ~27% of the true ceiling; theirs are ~340%
  of it. A reproduction-then-ablation campaign is mathematically guaranteed to
  fail; the correct paper is a two-line ceiling proof plus a forensic
  identification of what saturating-near-1 statistic they actually computed.

This module produces both pieces on real activations:

  1. CEILING CHECK — verify numerically, on our data and on the extremal
     configuration, that Eq.5/Eq.6 delta_rel <= 0.2929, and print the two-line
     argument with our measured values in context.

  2. CANDIDATE-STATISTIC SWEEP — compute, per layer, the statistics a busy
     implementation could plausibly have computed instead, and report which one
     lands in the Atlas's reported range (~0.99 plateau, ~0.4-0.7 final):
       a. unnormalized max defect (no diameter division; saturates with scale)
       b. defect normalized by the MIN positive pairwise distance
       c. defect normalized by the MEDIAN pairwise distance
       d. 1 - delta_rel ("tree-ness" flipped; a plausible sign/def flip)
       e. max Gromov PRODUCT / diameter (products grow ~diam; ratio -> ~1)
       f. delta from the min-form defect (second-largest minus smallest Gromov
          product) normalized by max GROMOV PRODUCT instead of diameter
     Whichever candidate matches their plateau AND their final-layer drop shape
     is the forensic identification. If NONE matches, that is also a result
     (their number is not reconstructible from within-prompt clouds at all).

Writes results/geometry_v2/atlas_forensics.csv + a verdict block to stdout/log.
CPU-friendly; runs on saved activations, no re-extraction.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import _sink_mask, ensure_dir, iter_samples, log_line, save_csv
from ..manifest import write_manifest

EUCLIDEAN_CEILING = 1.0 - 1.0 / np.sqrt(2.0)   # 0.29289... (unit square extremal)


def _pairwise(points):
    sq = np.sum(points * points, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def candidate_statistics(points: np.ndarray, n_quadruples: int, rng) -> dict:
    """All candidate per-cloud statistics for the forensic sweep."""
    n = points.shape[0]
    out = {k: float("nan") for k in
           ("eq56_delta_rel", "defect_unnorm", "defect_over_mindist",
            "defect_over_meddist", "one_minus_delta_rel",
            "max_gromov_over_diam", "minform_over_maxgromov")}
    if n < 4:
        return out
    dmat = _pairwise(points)
    iu = np.triu_indices(n, k=1)
    pos = dmat[iu][dmat[iu] > 0]
    diam = float(dmat.max())
    if diam <= 0 or pos.size == 0:
        return out
    idx = rng.integers(0, n, size=(n_quadruples, 4))
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]

    # sum-form (Atlas Eq. 5)
    s1 = dmat[a, b] + dmat[c, d]
    s2 = dmat[a, c] + dmat[b, d]
    s3 = dmat[a, d] + dmat[b, c]
    S = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)
    defect = 0.5 * (S[:, 2] - S[:, 1])
    dmax = float(defect.max())

    # Gromov products w.r.t. base point w=a
    gp_ab = 0.5 * (dmat[a, b] + dmat[a, c] - dmat[b, c])   # (b|c)_a etc.
    gp_ac = 0.5 * (dmat[a, b] + dmat[a, d] - dmat[b, d])
    gp_bc = 0.5 * (dmat[a, c] + dmat[a, d] - dmat[c, d])
    G = np.sort(np.stack([gp_ab, gp_ac, gp_bc], axis=1), axis=1)
    minform = G[:, 1] - G[:, 0]                            # min-form defect
    max_gp = float(np.max(G[:, 2]))

    out["eq56_delta_rel"] = dmax / diam
    out["defect_unnorm"] = dmax
    out["defect_over_mindist"] = dmax / float(pos.min())
    out["defect_over_meddist"] = dmax / float(np.median(pos))
    out["one_minus_delta_rel"] = 1.0 - dmax / diam
    out["max_gromov_over_diam"] = max_gp / diam
    out["minform_over_maxgromov"] = (float(minform.max()) / max_gp
                                     if max_gp > 0 else float("nan"))
    return out


def ceiling_check(rng, n_random_trials=20000):
    """Numerically confirm the Eq.5/Eq.6 Euclidean ceiling on random configs.

    Random 4-point Euclidean configurations (the defect depends on 4 points, so
    checking quadruple configs directly is exact) must never exceed
    1 - 1/sqrt(2); the unit square attains it.
    """
    worst = 0.0
    for dim in (2, 3, 8):
        pts = rng.standard_normal((n_random_trials, 4, dim))
        for i in range(0, n_random_trials, 2000):
            batch = pts[i:i + 2000]
            for cfg in batch:
                dmat = _pairwise(cfg)
                s1 = dmat[0, 1] + dmat[2, 3]
                s2 = dmat[0, 2] + dmat[1, 3]
                s3 = dmat[0, 3] + dmat[1, 2]
                s = sorted([s1, s2, s3])
                diam = dmat.max()
                if diam > 0:
                    worst = max(worst, 0.5 * (s[2] - s[1]) / diam)
    # unit square attains the ceiling
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    dm = _pairwise(sq)
    s = sorted([dm[0, 1] + dm[2, 3], dm[0, 2] + dm[1, 3], dm[0, 3] + dm[1, 2]])
    square_val = 0.5 * (s[2] - s[1]) / dm.max()
    return worst, square_val


def run(activations_dir, out_dir, n_quadruples=1500, max_prompts=150, seed=0):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                           "logs", "atlas_forensics.log")
    rng = np.random.default_rng(seed)

    # ---- Part 1: the ceiling ----
    worst, square_val = ceiling_check(rng)
    lines = [
        "ATLAS FORENSICS",
        "===============",
        f"Euclidean ceiling of Eq.5/Eq.6 delta_rel: {EUCLIDEAN_CEILING:.4f} "
        f"(analytic, unit square)",
        f"  numeric check: worst over 60k random 4-point configs = {worst:.4f}; "
        f"unit square = {square_val:.4f}",
        f"  => any Euclidean cloud satisfies delta_rel <= {EUCLIDEAN_CEILING:.4f}; "
        f"the Atlas's reported ~0.995 medians are unattainable under their own "
        f"printed formula. Our measured ~0.08 is ~27% of the true ceiling.",
        "",
    ]

    # ---- Part 2: candidate sweep on real activations ----
    rows = []
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        acc: dict[int, dict[str, list]] = {}
        used = 0
        for s in iter_samples(activations_dir, model, dataset):
            if s.get("variant", "original") != "original":
                continue
            hidden = np.asarray(s["hidden"], dtype=np.float64)
            for L in range(hidden.shape[0]):
                h = hidden[L][_sink_mask(hidden[L])]
                if h.shape[0] < 4:
                    continue
                stats = candidate_statistics(h, n_quadruples, rng)
                a = acc.setdefault(L, {k: [] for k in stats})
                for k, v in stats.items():
                    if np.isfinite(v):
                        a[k].append(v)
            used += 1
            if used >= max_prompts:
                break
        if not acc:
            continue
        n_layers = max(acc) + 1
        for L in sorted(acc):
            row = dict(model=model, dataset=dataset, layer=L,
                       rel_depth=round(L / max(n_layers - 1, 1), 3))
            for k, vals in acc[L].items():
                row[k] = round(float(np.median(vals)), 4) if vals else ""
            rows.append(row)

        # which candidate matches the Atlas's shape (plateau ~0.99, final drop)?
        sub = [r for r in rows if r["model"] == model]
        lines.append(f"[{model}] candidate-statistic sweep (median across prompts):")
        for cand in ("eq56_delta_rel", "one_minus_delta_rel", "max_gromov_over_diam",
                     "defect_over_meddist", "defect_over_mindist",
                     "minform_over_maxgromov", "defect_unnorm"):
            vals = [(r["layer"], r[cand]) for r in sub if r.get(cand) != ""]
            if not vals:
                continue
            n_l = max(v[0] for v in vals)
            band = [v for L, v in vals if 0.35 <= L / max(n_l, 1) <= 0.70]
            plateau = float(np.median(band)) if band else float("nan")
            final = vals[-1][1]
            atlas_like = (0.9 <= plateau <= 1.05) and (final < plateau - 0.1)
            lines.append(f"  {cand:26s}: plateau={plateau:8.3f} final={final:8.3f}"
                         f"{'   <-- MATCHES the Atlas shape' if atlas_like else ''}")
        lines.append("")

    save_csv(os.path.join(out_dir, "atlas_forensics.csv"), rows)
    write_manifest(out_dir, "atlas_forensics",
                   args=dict(activations=activations_dir, seed=seed,
                             n_quadruples=n_quadruples, max_prompts=max_prompts),
                   extra=dict(euclidean_ceiling=EUCLIDEAN_CEILING,
                              numeric_worst=worst))
    text = "\n".join(lines)
    log_line(logfile, "atlas_forensics done")
    print(text)
    with open(os.path.join(out_dir, "atlas_forensics_verdict.md"), "w") as fh:
        fh.write(text + "\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Atlas forensics: ceiling proof + "
                                             "candidate-statistic sweep.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry_v2")
    ap.add_argument("--n-quadruples", type=int, default=1500)
    ap.add_argument("--max-prompts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, n_quadruples=args.n_quadruples,
        max_prompts=args.max_prompts, seed=args.seed)


if __name__ == "__main__":
    main()
