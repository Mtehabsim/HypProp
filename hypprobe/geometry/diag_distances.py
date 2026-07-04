"""Read-only diagnostic: WHY is within-prompt delta_rel ~0.001 in middle layers?

Hypothesis to test: extreme anisotropy (a rogue high-magnitude direction) makes
diam=max(pairwise distance) explode relative to the four-point defect, so
delta_rel = defect/diam is crushed toward 0 -- a NORMALIZATION degeneracy, not a
geometry fact.

For a few prompts at a few layers we print, over the prompt's OWN token cloud:
  - max / median / p99 pairwise distance   (if max >> median -> heavy outlier/anisotropy)
  - the raw four-point defect scale (median, max)   (numerator size)
  - delta_rel under diam=max vs diam=p99            (does robust diameter rescue it?)
  - top-1 / top-3 singular-value share of the cloud (how 1-D / rogue-dominated it is)

No changes to any pipeline. Runs on saved activations.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..io import iter_samples


def _pairwise(x):
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _defects(dmat, n_quad, rng):
    n = dmat.shape[0]
    idx = rng.integers(0, n, size=(n_quad, 4))
    a, b, c, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    s1 = dmat[a, b] + dmat[c, d]
    s2 = dmat[a, c] + dmat[b, d]
    s3 = dmat[a, d] + dmat[b, c]
    S = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)
    return 0.5 * (S[:, 2] - S[:, 1])


def run(activations_dir, layers=(2, 14, 28), n_prompts=3, seed=0):
    rng = np.random.default_rng(seed)
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        print(f"\n================ {model} ================")
        shown = 0
        for s in iter_samples(activations_dir, model, dataset):
            hidden = np.asarray(s["hidden"], dtype=np.float64)
            n_layers, n_tok, dim = hidden.shape
            if n_tok < 8:
                continue
            print(f"\n  prompt {s.get('sample_id')}  (n_tok={n_tok}, dim={dim})")
            for L in layers:
                if L >= n_layers:
                    continue
                x = hidden[L]
                dmat = _pairwise(x)
                iu = np.triu_indices(n_tok, k=1)
                dd = dmat[iu]
                mx, med, p99 = dd.max(), np.median(dd), np.percentile(dd, 99)
                defect = _defects(dmat, 2000, rng)
                # singular-value concentration of the centered cloud
                xc = x - x.mean(0, keepdims=True)
                sv = np.linalg.svd(xc, compute_uv=False)
                sv2 = sv ** 2
                top1 = sv2[0] / sv2.sum()
                top3 = sv2[:3].sum() / sv2.sum()
                dr_max = defect.max() / mx if mx > 0 else 0.0
                dr_p99 = defect.max() / p99 if p99 > 0 else 0.0
                # also the per-token NORM spread (rogue-dim signature)
                norms = np.linalg.norm(x, axis=1)
                print(f"    L{L:2d}: dist max={mx:8.1f} med={med:8.1f} p99={p99:8.1f} "
                      f"max/med={mx/max(med,1e-9):6.1f} | defect med={np.median(defect):7.2f} "
                      f"max={defect.max():7.2f} | dr(max)={dr_max:.3f} dr(p99)={dr_p99:.3f} | "
                      f"sv top1={top1:.2f} top3={top3:.2f} | norm med={np.median(norms):.0f} "
                      f"max={norms.max():.0f}")
            shown += 1
            if shown >= n_prompts:
                break


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnose delta_rel degeneracy (read-only).")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 14, 28])
    ap.add_argument("--n-prompts", type=int, default=3)
    args = ap.parse_args(argv)
    run(args.activations, layers=tuple(args.layers), n_prompts=args.n_prompts)


if __name__ == "__main__":
    main()
