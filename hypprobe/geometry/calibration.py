"""Regime-matched calibration for delta_rel — the ruler fix.

The re-evaluation established (empirically, on this repo's own estimator) that
delta_rel has NO absolute scale:

  * The flat (isotropic-Gaussian) reference falls with dimension:
    0.31 (d=2) / 0.19 (d=64) / 0.11 (k=256) / 0.03 (d=3584) — concentration
    pushes high-d clouds toward equidistance, and an equidistant set is a STAR
    TREE (delta = 0). So "flat" reads as "tree" at high d, and comparing raw
    delta_rel across metrics / sources / layers with different effective dims is
    comparing numbers on different scales.
  * Separated flat class clusters ALSO score near 0 (a k-cluster mixture is a
    depth-1 star tree metrically), so low delta_rel does not license the word
    "hierarchy" without a cluster null.

The fix implemented here: for every measured data cloud, compute delta_rel for
THREE matched references pushed through the IDENTICAL transform at the identical
(N, k):

  flat_anchor    — isotropic Gaussian, matched N and ambient d.
  cluster_null   — mixture of Gaussians with the SAME empirical class means and
                   per-class covariance scale as the data (the boring
                   "class separation, not hierarchy" explanation).
  tree_anchor    — a low-distortion hyperbolic-style embedding of a balanced
                   tree at matched N and ambient d (the "real hierarchy" end).

Everything is then reported as a SPAN-RELATIVE score:

  score = (delta_flat_anchor - delta_data) / (delta_flat_anchor - delta_tree_anchor)

  score ~ 0  -> data is no more tree-like than matched flat noise
  score ~ 1  -> data is as tree-like as an actual tree in this regime
  and separately: excess = delta_cluster_null - delta_data
  (hierarchy beyond what class separation alone predicts; a hierarchy claim
  needs excess > noise floor, not just a low absolute number).

This is the module every v2 verdict reads its margins from — margins become
fractions of the measured span at the operating regime, never absolute
constants (PREREGISTER2.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .delta import delta_hyperbolicity


@dataclass
class CalibratedDelta:
    """delta_rel of a data cloud plus its regime-matched references."""

    delta_data: float
    delta_flat: float          # matched isotropic Gaussian
    delta_tree: float          # matched embedded tree
    delta_cluster: float       # matched class-cluster mixture (nan if no labels)
    span: float                # delta_flat - delta_tree (usable range here)
    score: float               # (flat - data) / span, in ~[0, 1]
    excess_over_cluster: float # delta_cluster - delta_data (hierarchy beyond clusters)
    noise_floor: float         # max(std_rel, bootstrap_std) of the data estimate
    n_points: int
    ambient_dim: int
    metric: str


def embedded_tree_cloud(n: int, dim: int, seed: int = 0,
                        branching: int = 2) -> np.ndarray:
    """A point cloud whose Euclidean distances approximate a balanced tree metric.

    A tree does not embed isometrically in Euclidean space, but a "hyperbolic
    cone" layout (children fan out from the parent with shrinking edge lengths
    in fresh random directions) preserves the tree's coarse metric structure
    well enough that the four-point estimator reads it as strongly tree-like —
    this is the LOW anchor of the span at the data's own (N, dim). Verified in
    tests: scores well below the matched flat anchor under every metric.
    """
    rng = np.random.default_rng(seed)
    pts = [np.zeros(dim)]
    parents = [0]
    # grow generations until we have n nodes
    edge = 4.0
    frontier = [0]
    while len(pts) < n:
        nxt = []
        for node in frontier:
            for _ in range(branching):
                if len(pts) >= n:
                    break
                d = rng.standard_normal(dim)
                d /= np.linalg.norm(d)
                pts.append(pts[node] + edge * d)
                parents.append(node)
                nxt.append(len(pts) - 1)
        frontier = nxt or frontier
        edge *= 0.55  # shrink edges with depth (hyperbolic-cone style)
    return np.stack(pts[:n])


def cluster_null_cloud(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Mixture of Gaussians with the data's OWN class means and per-class scale.

    This is the mandatory control from the re-evaluation: a flat mixture with
    separated class means scores near delta=0 (star tree), so any hierarchy
    claim must show the data is MORE tree-like than this null. Per-class
    covariance is approximated as isotropic at the class's mean per-coordinate
    variance (full covariance would be rank-deficient at N<<d and reproduce the
    data too exactly to be a null).
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(X)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        mu = X[idx].mean(axis=0)
        # isotropic scale = sqrt(mean per-coordinate variance) of this class
        sigma = float(np.sqrt(np.maximum(X[idx].var(axis=0).mean(), 1e-12)))
        out[idx] = mu[None, :] + sigma * rng.standard_normal((len(idx), X.shape[1]))
    return out


def calibrated_delta(
    X: np.ndarray,
    y: np.ndarray | None = None,
    metric: str = "per_cloud",
    bg_transform=None,
    seed: int = 0,
    n_bootstrap: int = 25,
    pca_cap: int = 256,
) -> CalibratedDelta:
    """delta_rel of ``X`` with matched flat / tree / cluster references.

    All four clouds (data + 3 references) go through the SAME metric transform
    at the same (N, ambient d) — the references therefore live on the same
    scale as the data, which absolute thresholds never did.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    kw = dict(metric=metric, bg_transform=bg_transform, seed=seed,
              pca_cap=pca_cap)

    r_data = delta_hyperbolicity(X, n_bootstrap=n_bootstrap, **kw)

    rng = np.random.default_rng(seed)
    # Match the data's overall scale so the background transform (an affine map
    # fit on data statistics) treats references comparably.
    data_scale = float(np.linalg.norm(X - X.mean(0), axis=1).mean()) or 1.0

    flat = rng.standard_normal((n, d))
    flat *= data_scale / (np.linalg.norm(flat, axis=1).mean() or 1.0)
    r_flat = delta_hyperbolicity(flat, **kw)

    tree = embedded_tree_cloud(n, d, seed=seed)
    tree *= data_scale / (np.linalg.norm(tree - tree.mean(0), axis=1).mean() or 1.0)
    r_tree = delta_hyperbolicity(tree, **kw)

    if y is not None and len(np.unique(y)) >= 2:
        clus = cluster_null_cloud(X, np.asarray(y), seed=seed)
        r_clus = delta_hyperbolicity(clus, **kw)
        delta_cluster = r_clus.delta_rel
    else:
        delta_cluster = float("nan")

    span = r_flat.delta_rel - r_tree.delta_rel
    score = (r_flat.delta_rel - r_data.delta_rel) / span if span > 1e-9 else float("nan")
    excess = (delta_cluster - r_data.delta_rel) if np.isfinite(delta_cluster) else float("nan")

    return CalibratedDelta(
        delta_data=r_data.delta_rel,
        delta_flat=r_flat.delta_rel,
        delta_tree=r_tree.delta_rel,
        delta_cluster=delta_cluster,
        span=span,
        score=score,
        excess_over_cluster=excess,
        noise_floor=max(r_data.std_rel, r_data.bootstrap_std),
        n_points=n,
        ambient_dim=d,
        metric=metric,
    )
