"""Gromov delta-hyperbolicity of a point cloud.

delta-hyperbolicity measures how "tree-like" a metric space is. delta = 0 means a
perfect tree; a flat / spherical space scores high. We compute it via the
four-point condition using Gromov products, following the standard sampling
recipe (Khrulkov et al. 2020; Welz et al. 2025): sample many quadruples of
points and take the max defect.

CRITICAL (from the literature review): always report the *normalised*
``delta_rel = 2 * delta / diam`` in [0, 1], never raw delta. Raw delta scales
with the diameter of the embedding, so a compressed layer (small diameter) can
look "hyperbolic" for the wrong reason. delta_rel removes that confound.

We also expose :func:`whiten`, because raw LLM hidden states are strongly
anisotropic (Ethayarajh 2019) and that anisotropy contaminates every pairwise
distance. Measuring delta on whitened states is what separates real hierarchy
from mere late-layer compression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeltaResult:
    """Result of a delta-hyperbolicity estimate."""

    delta: float          # raw Gromov delta (max defect over sampled quadruples)
    delta_rel: float      # normalised 2*delta/diam in [0, 1] (0 = tree, ~1 = flat)
    diam: float           # diameter (max pairwise distance) of the sample
    n_points: int         # number of points used
    n_quadruples: int     # number of quadruples sampled
    std_rel: float        # std of delta_rel across repeats (0 if single repeat)


def whiten(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean, identity-covariance (ZCA) whitening of rows of ``x``.

    Removes the anisotropy that would otherwise dominate pairwise distances.
    ``x`` is ``(N, d)``; returns ``(N, d)``.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    # Covariance and its inverse square root (ZCA).
    cov = np.cov(xc, rowvar=False)
    cov = np.atleast_2d(cov)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    w = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return xc @ w


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    """Dense ``(N, N)`` Euclidean distance matrix."""
    sq = np.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def delta_from_distance_matrix(
    dmat: np.ndarray,
    n_quadruples: int = 1500,
    n_repeats: int = 5,
    rng: np.random.Generator | None = None,
) -> DeltaResult:
    """Estimate delta-hyperbolicity from a precomputed distance matrix.

    Uses the Gromov four-point condition. For a fixed base point ``w`` the
    Gromov product is ``(x|y)_w = 0.5 * (d(x,w) + d(y,w) - d(x,y))``. delta is
    the largest amount by which the four-point condition is violated over the
    sampled quadruples. We sample ``n_quadruples`` quadruples per repeat and
    average ``delta_rel`` across ``n_repeats`` repeats for a stable estimate.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = dmat.shape[0]
    if n < 4:
        raise ValueError(f"need >=4 points for delta, got {n}")
    diam = float(dmat.max())
    if diam <= 0:
        return DeltaResult(0.0, 0.0, 0.0, n, 0, 0.0)

    rels: list[float] = []
    deltas: list[float] = []
    for _ in range(n_repeats):
        idx = rng.integers(0, n, size=(n_quadruples, 4))
        # Use the first index of each quadruple as the base point w.
        w = idx[:, 0]
        a, b, cc = idx[:, 1], idx[:, 2], idx[:, 3]
        dw_a = dmat[w, a]
        dw_b = dmat[w, b]
        dw_c = dmat[w, cc]
        d_ab = dmat[a, b]
        d_ac = dmat[a, cc]
        d_bc = dmat[b, cc]
        # Gromov products relative to w.
        gp_ab = 0.5 * (dw_a + dw_b - d_ab)
        gp_ac = 0.5 * (dw_a + dw_c - d_ac)
        gp_bc = 0.5 * (dw_b + dw_c - d_bc)
        # Four-point defect: min of the two largest Gromov products minus the
        # smallest, per the standard formulation delta = max over quadruples of
        # (second-largest - smallest) among the three sums.
        stacked = np.stack([gp_ab, gp_ac, gp_bc], axis=1)
        stacked.sort(axis=1)  # ascending
        defect = stacked[:, 1] - stacked[:, 0]  # two-largest agree -> use 2nd largest minus min
        delta = float(defect.max())
        deltas.append(delta)
        rels.append(2.0 * delta / diam)

    return DeltaResult(
        delta=float(np.mean(deltas)),
        delta_rel=float(np.mean(rels)),
        diam=diam,
        n_points=n,
        n_quadruples=n_quadruples,
        std_rel=float(np.std(rels)),
    )


def delta_hyperbolicity(
    points: np.ndarray,
    n_quadruples: int = 1500,
    n_repeats: int = 5,
    do_whiten: bool = True,
    max_points: int = 1500,
    seed: int = 0,
) -> DeltaResult:
    """Estimate delta_rel for a cloud of points (rows of ``points``).

    If ``do_whiten`` the points are ZCA-whitened first (recommended for LLM
    hidden states). If there are more than ``max_points`` rows we subsample for
    the distance matrix (delta is a global statistic, so a representative
    subsample is standard practice).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(points, dtype=np.float64)
    if do_whiten:
        x = whiten(x)
    if x.shape[0] > max_points:
        sel = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[sel]
    dmat = _pairwise_euclidean(x)
    return delta_from_distance_matrix(dmat, n_quadruples, n_repeats, rng)
