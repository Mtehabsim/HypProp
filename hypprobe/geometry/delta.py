"""Gromov delta-hyperbolicity of a point cloud.

delta-hyperbolicity measures how "tree-like" a metric space is. delta = 0 means a
perfect tree; a flat / spherical space scores high. We compute it via the
four-point condition using Gromov products, following the standard sampling
recipe (Khrulkov et al. 2020; Welz et al. 2025): sample many quadruples of
points and take a high-percentile defect.

CRITICAL (from the literature review): always report the *normalised*
``delta_rel = 2 * delta / diam`` in [0, 1], never raw delta. Raw delta scales
with the diameter of the embedding, so a compressed layer (small diameter) can
look "hyperbolic" for the wrong reason. delta_rel removes that confound.

Numerical-soundness notes (these bias delta in KNOWN directions, so we control
them explicitly):

  * Whitening in the N << d regime (few points, high dim) is unsound: a full d x d
    covariance is rank-deficient, so inverting clipped near-null eigenvalues
    amplifies noise by huge factors AND pushes points toward mutual equidistance
    (which biases delta_rel toward "flat"). We therefore PCA-reduce to
    ~min(N//3, pca_cap) components BEFORE whitening. See :func:`whiten`.

  * The sampled MAX defect is a downward-biased estimator of the true max, which
    over-reports hyperbolicity. We report a high PERCENTILE (default 99th) of the
    defect and scale the number of quadruples with N. We also keep the seed
    variance (std_rel); no delta_rel claim should be trusted below std_rel.

  * The diameter (the normaliser) is outlier-driven if taken as the raw max: a
    single rare huge-norm token inflates it and deflates delta_rel toward "tree".
    We use a high-percentile diameter (default 99th) instead of the max.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeltaResult:
    """Result of a delta-hyperbolicity estimate."""

    delta: float          # high-percentile Gromov defect over sampled quadruples
    delta_rel: float      # normalised 2*delta/diam in [0, 1] (0 = tree, ~1 = flat)
    diam: float           # percentile diameter (robust scale) of the sample
    n_points: int         # number of points used
    n_quadruples: int     # number of quadruples sampled (per repeat)
    std_rel: float        # std of delta_rel across repeats (do not trust claims below this)
    pca_dim: int = 0      # PCA dimension used before whitening (0 = no whitening/PCA)


def whiten(x: np.ndarray, eps: float = 1e-6, pca_cap: int = 256) -> np.ndarray:
    """Zero-mean, unit-covariance whitening, sound in the N << d regime.

    Raw ZCA on N points in d dims with N < d inverts near-null eigenvalues and
    biases points toward equidistance. To avoid that we first PCA-project onto
    ``k = min(N // 3, pca_cap, d)`` well-supported components, then whiten within
    that subspace (each retained component scaled to unit variance). This keeps
    the anisotropy correction (its purpose) without amplifying noise directions.

    ``x`` is ``(N, d)``; returns ``(N, k)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    k = max(1, min(n // 3, pca_cap, d))
    # SVD of the centred data: columns of Vt are principal directions.
    # (economy SVD; U s Vt = xc)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(k, s.shape[0])
    # Project onto top-k components and scale each to unit variance.
    # Singular value s_i relates to variance as var_i = s_i^2 / (n - 1).
    comps = vt[:k]                      # (k, d)
    proj = xc @ comps.T                 # (N, k)
    std = s[:k] / np.sqrt(max(n - 1, 1))
    std = np.clip(std, eps, None)
    return proj / std[None, :]


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
    defect_percentile: float = 99.0,
    diam_percentile: float = 99.0,
    pca_dim: int = 0,
) -> DeltaResult:
    """Estimate delta-hyperbolicity from a precomputed distance matrix.

    Uses the Gromov four-point condition. For a fixed base point ``w`` the
    Gromov product is ``(x|y)_w = 0.5 * (d(x,w) + d(y,w) - d(x,y))``; the defect
    of a quadruple is (second-largest - smallest) of the three products. We
    report the ``defect_percentile`` (default 99th) of sampled defects rather
    than the max (the max is downward-biased under sampling), and normalise by a
    percentile diameter (robust to outlier points).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = dmat.shape[0]
    if n < 4:
        raise ValueError(f"need >=4 points for delta, got {n}")
    # Robust scale: high-percentile pairwise distance, not the raw max.
    iu = np.triu_indices(n, k=1)
    diam = float(np.percentile(dmat[iu], diam_percentile))
    if diam <= 0:
        return DeltaResult(0.0, 0.0, 0.0, n, 0, 0.0, pca_dim)

    rels: list[float] = []
    deltas: list[float] = []
    for _ in range(n_repeats):
        idx = rng.integers(0, n, size=(n_quadruples, 4))
        w = idx[:, 0]
        a, b, cc = idx[:, 1], idx[:, 2], idx[:, 3]
        gp_ab = 0.5 * (dmat[w, a] + dmat[w, b] - dmat[a, b])
        gp_ac = 0.5 * (dmat[w, a] + dmat[w, cc] - dmat[a, cc])
        gp_bc = 0.5 * (dmat[w, b] + dmat[w, cc] - dmat[b, cc])
        stacked = np.stack([gp_ab, gp_ac, gp_bc], axis=1)
        stacked.sort(axis=1)  # ascending
        defect = stacked[:, 1] - stacked[:, 0]
        # High-percentile defect: less downward-biased than max, less noisy.
        delta = float(np.percentile(defect, defect_percentile))
        deltas.append(delta)
        rels.append(2.0 * delta / diam)

    return DeltaResult(
        delta=float(np.mean(deltas)),
        delta_rel=float(np.mean(rels)),
        diam=diam,
        n_points=n,
        n_quadruples=n_quadruples,
        std_rel=float(np.std(rels)),
        pca_dim=pca_dim,
    )


def delta_hyperbolicity(
    points: np.ndarray,
    n_quadruples: int | None = None,
    n_repeats: int = 5,
    do_whiten: bool = True,
    max_points: int = 1500,
    seed: int = 0,
    defect_percentile: float = 99.0,
    diam_percentile: float = 99.0,
    pca_cap: int = 256,
) -> DeltaResult:
    """Estimate delta_rel for a cloud of points (rows of ``points``).

    If ``do_whiten`` the points are PCA-then-whitened (sound for LLM hidden
    states; see :func:`whiten`). If there are more than ``max_points`` rows we
    subsample. ``n_quadruples`` defaults to ``max(1500, 50 * N)`` so the sampling
    scales with the number of points (a fixed budget under-samples large clouds).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(points, dtype=np.float64)
    pca_dim = 0
    if do_whiten:
        x = whiten(x, pca_cap=pca_cap)
        pca_dim = x.shape[1]
    if x.shape[0] > max_points:
        sel = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[sel]
    if n_quadruples is None:
        n_quadruples = max(1500, 50 * x.shape[0])
    dmat = _pairwise_euclidean(x)
    return delta_from_distance_matrix(
        dmat, n_quadruples, n_repeats, rng,
        defect_percentile=defect_percentile, diam_percentile=diam_percentile,
        pca_dim=pca_dim,
    )
