"""Phase 1 sanity: structural probe (distance regression) -- reproduces Raj.

A structural probe (Hewitt & Manning 2019; Poincare version Chen et al. 2021)
learns a linear map B so that distances in the transformed space match a target
tree/graph distance. We fit both a Euclidean and a hyperbolic variant and report
Spearman rho + distortion per layer. On DeepSeek + PrOntoQA this should recover
Raj's pattern (Euclidean collapses at the final layer, hyperbolic stays high),
which validates our extractor + geometry before any novel claims.

This is distance regression, NOT logistic regression -- a different tool used
only to measure geometry.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from ..geometry import poincare
from ..io import ensure_dir, iter_samples, log_line, save_csv


class StructuralProbe(nn.Module):
    """Learn B (proj) so transformed distances match target distances."""

    def __init__(self, in_dim, proj_dim, curvature=0.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.B = nn.Linear(in_dim, proj_dim, bias=False)
        with torch.no_grad():
            self.B.weight.copy_(torch.randn(proj_dim, in_dim, generator=g) * 0.05)
        self.curvature = curvature

    def transformed(self, x):
        z = self.B(x)
        if self.curvature > 0:
            return poincare.expmap0(z, self.curvature)
        return z

    def dist(self, x):
        z = self.transformed(x)
        n = z.shape[0]
        zi = z.unsqueeze(1).expand(n, n, z.shape[-1]).reshape(n * n, -1)
        zj = z.unsqueeze(0).expand(n, n, z.shape[-1]).reshape(n * n, -1)
        d = poincare.dist(zi, zj, self.curvature)
        return d.reshape(n, n)


def _spearman(a, b):
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    ar = ar - ar.mean(); br = br - br.mean()
    denom = np.sqrt((ar ** 2).sum() * (br ** 2).sum())
    return float((ar * br).sum() / denom) if denom > 0 else 0.0


def fit_structural(feats, target_d, curvature, proj_dim=5, epochs=300, seed=0):
    x = torch.as_tensor(feats, dtype=torch.float32)
    tgt = torch.as_tensor(target_d, dtype=torch.float32)
    model = StructuralProbe(x.shape[1], proj_dim, curvature, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    mask = ~torch.eye(len(x), dtype=torch.bool)
    tgt_norm = (tgt[mask] ** 2).sum().clamp_min(1e-9)
    for _ in range(epochs):
        opt.zero_grad()
        dpred = model.dist(x)
        loss = ((dpred[mask] - tgt[mask]) ** 2).sum() / tgt_norm  # stress-normalised
        loss.backward(); opt.step()
    with torch.no_grad():
        dpred = model.dist(x)[mask].numpy()
    rho = _spearman(dpred, tgt[mask].numpy())
    distortion = float(np.mean(np.abs(dpred - tgt[mask].numpy())))
    return rho, distortion


def run(activations_dir, out_dir, dataset="prontoqa", proj_dim=5, seed=0):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "struct.log")
    rows = []
    models = sorted({s["model"] for s in iter_samples(activations_dir, dataset=dataset)})
    for model in models:
        samples = [s for s in iter_samples(activations_dir, model, dataset)]
        if len(samples) < 8:
            log_line(logfile, f"{model}/{dataset}: too few samples for structural probe")
            continue
        # Target distance = |depth_i - depth_j| using label_path length (reasoning depth).
        depths = np.array([len(s.get("label_path", [0])) for s in samples], dtype=float)
        target_d = np.abs(depths[:, None] - depths[None, :])
        n_layers = int(np.asarray(samples[0]["hidden"]).shape[0])
        for layer in range(n_layers):
            feats = np.stack([np.asarray(s["hidden"])[layer, -1] for s in samples]).astype(np.float64)
            rho_e, dist_e = fit_structural(feats, target_d, 0.0, proj_dim, seed=seed)
            rho_h, dist_h = fit_structural(feats, target_d, 1.0, proj_dim, seed=seed)
            rows.append(dict(model=model, dataset=dataset, layer=layer,
                             rho_euclidean=round(rho_e, 3), rho_hyperbolic=round(rho_h, 3),
                             dist_euclidean=round(dist_e, 3), dist_hyperbolic=round(dist_h, 3)))
        log_line(logfile, f"{model}/{dataset}: structural probe over {n_layers} layers "
                          f"(final-layer Euc rho={rows[-1]['rho_euclidean']}, "
                          f"Hyp rho={rows[-1]['rho_hyperbolic']}) -- Raj target ~0.488/0.967")
    save_csv(os.path.join(out_dir, "structural_probe.csv"), rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1: structural probe (reproduces Raj).")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry")
    ap.add_argument("--dataset", default="prontoqa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, dataset=args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()
