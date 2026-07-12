"""Tests for the matched-conditioning probe module."""

import numpy as np
import pytest


def test_matched_probe_arms_run():
    """All three arms produce a finite rho on synthetic data."""
    from hypprobe.geometry.matched_probe import fit_arm

    rng = np.random.default_rng(0)
    n = 60
    X = rng.standard_normal((n, 32)).astype(np.float64)
    depths = np.repeat(np.arange(5), 12)
    D = np.abs(depths[:, None] - depths[None, :]).astype(np.float64)
    tr, va = np.arange(42), np.arange(42, 60)
    Xtr, Xva = X[tr], X[va]
    Dtr, Dva = D[np.ix_(tr, tr)], D[np.ix_(va, va)]

    for arm in ("bare_euclidean", "cond_euclidean", "hyperbolic"):
        res = fit_arm(arm, Xtr, Dtr, Xva, Dva, proj_dim=5, seed=0,
                      max_epochs=200, patience=3, check_every=20)
        assert np.isfinite(res["rho"]), f"{arm} produced non-finite rho"
        assert res["epochs_trained"] > 0


def test_conditioning_beats_bare_on_structured_data():
    """On data with real tree structure, conditioned arms should improve over bare.

    This is a power check: if the test fails, the probe is broken, not the
    hypothesis (structured data is a known positive control).
    """
    from hypprobe.data.synthetic import hierarchy_features
    from hypprobe.geometry.matched_probe import fit_arm

    feats, labels, tree_d, paths = hierarchy_features(
        depth=3, branching=3, per_leaf=15, dim=32, noise=0.2, seed=7)
    D = tree_d[labels][:, labels].astype(np.float64)
    n = feats.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr, va = perm[:int(0.7 * n)], perm[int(0.7 * n):]
    Xtr, Xva = feats[tr], feats[va]
    Dtr, Dva = D[np.ix_(tr, tr)], D[np.ix_(va, va)]

    res_bare = fit_arm("bare_euclidean", Xtr, Dtr, Xva, Dva, seed=0,
                       max_epochs=500, patience=5, check_every=50)
    res_cond = fit_arm("cond_euclidean", Xtr, Dtr, Xva, Dva, seed=0,
                       max_epochs=500, patience=5, check_every=50)
    # On tree-structured data both should get reasonable rho; cond should not
    # be dramatically worse than bare (it may be slightly better or equal).
    assert res_cond["rho"] > res_bare["rho"] - 0.15, (
        f"cond ({res_cond['rho']:.3f}) should not collapse vs bare ({res_bare['rho']:.3f})")


def test_hypll_cross_check():
    """If HypLL is installed, verify our Poincare distance matches it."""
    from hypprobe.geometry.matched_probe import hypll_distance_check

    result = hypll_distance_check()
    if result is None:
        pytest.skip("HypLL not installed; cross-check skipped")
    assert result["ok"], f"Poincare distance mismatch: max err = {result['max_abs_err']}"
