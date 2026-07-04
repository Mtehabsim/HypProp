"""Does the LABEL taxonomy embed as a tree in the representation?

delta_rel tells us whether the point *cloud* is tree-like. That is necessary but
not sufficient: the cloud can be perfectly hyperbolic while the safety labels are
scattered across it. This module measures whether the *taxonomy* itself is
recovered by the geometry, which is the other half of the plan's joint selection
criterion (pick a setting with BOTH low delta_rel AND good label alignment).

Two signals, each computed for the Euclidean and the hyperbolic (Poincare)
geometry so we can see if curvature helps:

  1. prototype tree-distance correlation: build a per-class prototype (mean of
     that class's points, mapped to the ball for the hyperbolic case), then
     Spearman-correlate pairwise prototype distances with the taxonomy tree
     distances derived from label_path. Higher = the taxonomy's shape is present.

  2. norm-depth correlation: in a Poincare embedding, general concepts sit near
     the origin and specific ones near the boundary, so a point's norm should
     grow with its depth in the taxonomy. We correlate distance-from-origin with
     label depth. Positive = depth is encoded radially (a hyperbolic signature).

Everything works on pooled feature matrices + label_paths; no LLM needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import poincare
from .delta import whiten as zca_whiten


@dataclass
class AlignmentResult:
    proto_corr_euclidean: float   # Spearman(proto dist, tree dist), flat
    proto_corr_hyperbolic: float  # Spearman(proto dist, tree dist), Poincare
    norm_depth_corr: float        # Spearman(dist-from-origin, depth), hyperbolic
    n_classes: int
    curvature: float


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean()
    denom = np.sqrt((ar ** 2).sum() * (br ** 2).sum())
    return float((ar * br).sum() / denom) if denom > 0 else 0.0


def tree_distance_from_paths(paths: list[list[int]]) -> tuple[np.ndarray, list[list[int]]]:
    """Tree distance between the unique label paths (2*(depth - shared prefix))."""
    uniq: list[list[int]] = []
    seen = set()
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            uniq.append(list(p))
    n = len(uniq)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            shared = 0
            for a, b in zip(uniq[i], uniq[j]):
                if a == b:
                    shared += 1
                else:
                    break
            depth_i, depth_j = len(uniq[i]), len(uniq[j])
            d[i, j] = (depth_i - shared) + (depth_j - shared)
    return d, uniq


def label_alignment(
    X: np.ndarray,
    label_paths: list[list[int]],
    curvature: float = 1.0,
    do_whiten: bool = True,
) -> AlignmentResult:
    """Compute prototype tree-distance and norm-depth correlations.

    ``X`` is (N, d) pooled features; ``label_paths`` gives each sample's taxonomy
    path from root to leaf. Prototypes are per-unique-path means.
    """
    X = np.asarray(X, dtype=np.float64)
    if do_whiten:
        # whiten() is now sound in the N<<d regime (PCA-then-whiten), so we no
        # longer need the old "only if N>d" guard.
        X = zca_whiten(X)

    tree_d, uniq_paths = tree_distance_from_paths(label_paths)
    path_index = {tuple(p): k for k, p in enumerate(uniq_paths)}
    n_classes = len(uniq_paths)
    if n_classes < 3:
        return AlignmentResult(0.0, 0.0, 0.0, n_classes, curvature)

    # Per-class prototypes (mean feature vector).
    dim = X.shape[1]
    protos = np.zeros((n_classes, dim))
    counts = np.zeros(n_classes)
    for x, p in zip(X, label_paths):
        k = path_index[tuple(p)]
        protos[k] += x
        counts[k] += 1
    counts = np.clip(counts, 1, None)
    protos /= counts[:, None]

    # Off-diagonal tree distances (the target ordering).
    iu = np.triu_indices(n_classes, k=1)
    tree_flat = tree_d[iu]

    # Euclidean prototype distances.
    pe = torch.as_tensor(protos, dtype=torch.float32)
    n = n_classes
    ei = pe.unsqueeze(1).expand(n, n, dim).reshape(n * n, dim)
    ej = pe.unsqueeze(0).expand(n, n, dim).reshape(n * n, dim)
    d_euc = (ei - ej).norm(dim=-1).reshape(n, n).numpy()[iu]

    # Hyperbolic prototype distances: scale into the ball, exp-map, geodesic dist.
    scale = 0.9 / (np.abs(protos).max() + 1e-9)
    pb = poincare.expmap0(torch.as_tensor(protos * scale, dtype=torch.float32), curvature)
    bi = pb.unsqueeze(1).expand(n, n, dim).reshape(n * n, dim)
    bj = pb.unsqueeze(0).expand(n, n, dim).reshape(n * n, dim)
    d_hyp = poincare.dist(bi, bj, curvature).reshape(n, n).numpy()[iu]

    corr_e = _spearman(d_euc, tree_flat)
    corr_h = _spearman(d_hyp, tree_flat)

    # Norm-depth: distance-from-origin of each prototype vs its depth. Depth is
    # the LAST path element (the fine level), not len(path): builders that store
    # a fixed-length path (e.g. [coarse, depth]) have constant len -> a constant
    # depth vector -> a meaningless norm-depth correlation. Fall back to len only
    # when the path is a variable-length root->leaf chain.
    _lens = {len(p) for p in uniq_paths}
    if len(_lens) == 1 and uniq_paths and len(uniq_paths[0]) > 0:
        depths = np.array([float(p[-1]) for p in uniq_paths], dtype=float)
    else:
        depths = np.array([len(p) for p in uniq_paths], dtype=float)
    norms = poincare.dist0(pb, curvature).numpy()
    norm_depth = _spearman(norms, depths)

    return AlignmentResult(corr_e, corr_h, norm_depth, n_classes, curvature)
