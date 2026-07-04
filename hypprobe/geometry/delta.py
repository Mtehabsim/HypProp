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
    std_rel: float        # MC (quadruple-sampling) noise: the WEAKER floor
    pca_dim: int = 0      # PCA dimension used before whitening (0 = no whitening/PCA)
    bootstrap_std: float = 0.0  # std of delta_rel over point-resamples: the HONEST floor
    metric: str = "raw"   # which distance metric family member produced this


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


def pca_only(x: np.ndarray, pca_cap: int = 256) -> np.ndarray:
    """PCA-project to k top components but DO NOT rescale to unit variance.

    This isolates the effect of dimensionality reduction from the effect of
    rescaling: comparing ``pca_only`` vs ``per_cloud`` tells us whether a change
    in delta is driven by dropping empty directions or by equalising variance.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    k = max(1, min(n // 3, pca_cap, d))
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(k, s.shape[0])
    return xc @ vt[:k].T            # projected, NOT divided by std


def fit_background(x: np.ndarray, eps: float = 1e-6, pca_cap: int = 256):
    """Fit a GENERIC anisotropy transform from a large background sample.

    Unlike per-cloud whitening (which removes THIS cloud's own 2nd-order
    structure and can erase real hierarchy), the background transform removes
    only the model's *generic* anisotropy, estimated once from a big pooled
    sample, and is then applied to every cloud. This is the "honest middle":
    it strips the shared cone without whitening away each cloud's own structure.

    Returns a callable ``transform(z) -> z'`` mapping any (M, d) into the same
    background-whitened k-dim space.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    k = max(1, min(n // 3, pca_cap, d))
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(k, s.shape[0])
    comps = vt[:k]
    std = np.clip(s[:k] / np.sqrt(max(n - 1, 1)), eps, None)
    W = comps.T / std[None, :]                       # (d, k)

    def transform(z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float64) - mu) @ W

    return transform


def _apply_metric(x, metric, pca_cap, bg_transform):
    """Map raw points into the space chosen by ``metric``."""
    if metric == "raw":
        return np.asarray(x, dtype=np.float64)
    if metric == "pca_only":
        return pca_only(x, pca_cap=pca_cap)
    if metric == "per_cloud":
        return whiten(x, pca_cap=pca_cap)
    if metric == "background":
        if bg_transform is None:
            raise ValueError("metric='background' requires a fitted bg_transform")
        return bg_transform(x)
    if metric == "causal":
        # Park et al. causal inner product (model-level whitening). Estimator is
        # a separate (DGX) artifact; we accept it as bg_transform. If not
        # supplied, refuse rather than fake it.
        if bg_transform is None:
            raise ValueError("metric='causal' requires a model-level transform (Park)")
        return bg_transform(x)
    raise ValueError(f"unknown metric {metric!r}")


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
    do_whiten: bool | None = None,
    metric: str | None = None,
    bg_transform=None,
    max_points: int = 1500,
    seed: int = 0,
    defect_percentile: float = 99.0,
    diam_percentile: float = 99.0,
    pca_cap: int = 256,
    n_bootstrap: int = 0,
) -> DeltaResult:
    """Estimate delta_rel for a cloud of points (rows of ``points``).

    ``metric`` selects the distance-space (the Rung-0 family):
      - "raw"        : original coordinates (reproduces the Atlas).
      - "pca_only"   : PCA-project to k, NO rescale (dims vs rescaling probe).
      - "per_cloud"  : PCA-then-whiten on this cloud (strong, but biased against
                       real hierarchy that lives in the cloud's own directions).
      - "background" : apply a generic anisotropy transform fitted elsewhere
                       (needs ``bg_transform``) -- the honest middle.
      - "causal"     : model-level (Park) transform via ``bg_transform``.

    Backward-compat: if ``metric`` is None we map the old ``do_whiten`` flag
    (True->"per_cloud", False->"raw"); default is "per_cloud".

    ``n_bootstrap`` > 0 adds ``bootstrap_std`` by resampling the point set with
    replacement -- the honest noise floor (vs ``std_rel``, which is only
    quadruple-sampling noise).
    """
    if metric is None:
        metric = "raw" if do_whiten is False else "per_cloud"
    rng = np.random.default_rng(seed)
    x0 = np.asarray(points, dtype=np.float64)
    x = _apply_metric(x0, metric, pca_cap, bg_transform)
    pca_dim = x.shape[1] if metric in ("pca_only", "per_cloud", "background", "causal") else 0
    if x.shape[0] > max_points:
        sel = rng.choice(x.shape[0], size=max_points, replace=False)
        x = x[sel]
    if n_quadruples is None:
        n_quadruples = max(1500, 50 * x.shape[0])
    dmat = _pairwise_euclidean(x)
    res = delta_from_distance_matrix(
        dmat, n_quadruples, n_repeats, rng,
        defect_percentile=defect_percentile, diam_percentile=diam_percentile,
        pca_dim=pca_dim,
    )
    res.metric = metric

    if n_bootstrap and n_bootstrap > 1:
        boot = []
        m = x.shape[0]
        for _ in range(n_bootstrap):
            idx = rng.integers(0, m, size=m)         # resample points w/ replacement
            db = _pairwise_euclidean(x[idx])
            rb = delta_from_distance_matrix(db, n_quadruples, 1, rng,
                                            defect_percentile=defect_percentile,
                                            diam_percentile=diam_percentile)
            boot.append(rb.delta_rel)
        res.bootstrap_std = float(np.std(boot))
    return res
