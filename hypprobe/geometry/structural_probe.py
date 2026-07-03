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


def _depth_target(samples):
    """Reasoning-depth target |depth_i - depth_j| (Raj-style). (N, N)."""
    depths = np.array([len(s.get("label_path", [0])) for s in samples], dtype=float)
    return np.abs(depths[:, None] - depths[None, :])


def _taxonomy_target(samples):
    """Safety-taxonomy tree-distance target from each sample's label_path.

    Uses the same shared-prefix tree metric as label_alignment, so the notion of
    "tree distance" is identical across the codebase. Two samples in sibling
    categories are close; cross-branch categories are far. Returns (N, N).
    """
    from .label_alignment import tree_distance_from_paths

    paths = [list(s.get("label_path", [s.get("label", 0)])) for s in samples]
    tree_d, uniq = tree_distance_from_paths(paths)
    index = {tuple(p): k for k, p in enumerate(uniq)}
    idx = np.array([index[tuple(p)] for p in paths])
    # Expand the per-class tree distances to a per-sample (N, N) matrix.
    return tree_d[np.ix_(idx, idx)]


def run(activations_dir, out_dir, dataset="prontoqa", proj_dim=5, seed=0,
        target="depth"):
    """Structural probe per layer, flat vs hyperbolic.

    target='depth'    -> reasoning-depth tree (reproduces Raj on PrOntoQA).
    target='taxonomy' -> safety-taxonomy tree distance from label_path (the harm
                         hierarchy; use this on AILuminate/Aegis/WOS).
    """
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "struct.log")
    rows = []
    models = sorted({s["model"] for s in iter_samples(activations_dir, dataset=dataset)})
    for model in models:
        samples = [s for s in iter_samples(activations_dir, model, dataset)]
        if len(samples) < 8:
            log_line(logfile, f"{model}/{dataset}: too few samples for structural probe")
            continue
        if target == "taxonomy":
            target_d = _taxonomy_target(samples)
        else:
            target_d = _depth_target(samples)
        if float(target_d.max()) <= 0:
            log_line(logfile, f"{model}/{dataset}: target '{target}' is degenerate "
                              f"(all distances 0 -- need >1 class/depth); skipping")
            continue
        n_layers = int(np.asarray(samples[0]["hidden"]).shape[0])
        for layer in range(n_layers):
            feats = np.stack([np.asarray(s["hidden"])[layer, -1] for s in samples]).astype(np.float64)
            rho_e, dist_e = fit_structural(feats, target_d, 0.0, proj_dim, seed=seed)
            rho_h, dist_h = fit_structural(feats, target_d, 1.0, proj_dim, seed=seed)
            rows.append(dict(model=model, dataset=dataset, target=target, layer=layer,
                             rho_euclidean=round(rho_e, 3), rho_hyperbolic=round(rho_h, 3),
                             dist_euclidean=round(dist_e, 3), dist_hyperbolic=round(dist_h, 3)))
        raj_note = " -- Raj target ~0.488/0.967" if target == "depth" else ""
        log_line(logfile, f"{model}/{dataset} [target={target}]: structural probe over "
                          f"{n_layers} layers (final-layer Euc rho={rows[-1]['rho_euclidean']}, "
                          f"Hyp rho={rows[-1]['rho_hyperbolic']}){raj_note}")
    suffix = "" if target == "depth" else f"_{target}"
    save_csv(os.path.join(out_dir, f"structural_probe{suffix}.csv"), rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase 1: structural probe. target=depth reproduces Raj; "
                    "target=taxonomy tests the safety hierarchy.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry")
    ap.add_argument("--dataset", default="prontoqa")
    ap.add_argument("--target", default="depth", choices=["depth", "taxonomy"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, dataset=args.dataset, seed=args.seed,
        target=args.target)


if __name__ == "__main__":
    main()
