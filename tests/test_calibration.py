"""Tests for the regime-matched calibration module."""

import numpy as np
import pytest

from hypprobe.geometry.calibration import (
    CalibratedDelta,
    calibrated_delta,
    cluster_null_cloud,
    embedded_tree_cloud,
)


def test_embedded_tree_scores_low():
    """The tree anchor must produce a delta_rel well below the flat anchor."""
    tree = embedded_tree_cloud(200, 32, seed=0)
    from hypprobe.geometry.delta import delta_hyperbolicity
    r = delta_hyperbolicity(tree, metric="raw", seed=0)
    flat = np.random.default_rng(0).standard_normal((200, 32))
    rf = delta_hyperbolicity(flat, metric="raw", seed=0)
    assert r.delta_rel < rf.delta_rel * 0.6, (
        f"tree anchor ({r.delta_rel:.3f}) should be well below "
        f"flat ({rf.delta_rel:.3f})")


def test_cluster_null_preserves_means():
    """Cluster null must keep the empirical class means."""
    rng = np.random.default_rng(42)
    X = np.vstack([rng.standard_normal((50, 16)) + i * 3 for i in range(4)])
    y = np.repeat(np.arange(4), 50)
    C = cluster_null_cloud(X, y, seed=0)
    for cls in range(4):
        mu_data = X[y == cls].mean(0)
        mu_null = C[y == cls].mean(0)
        np.testing.assert_allclose(mu_null, mu_data, atol=0.7,
                                   err_msg=f"class {cls} mean shifted too far")


def test_calibrated_delta_span_positive():
    """The span (flat - tree) must be positive for the calibration to work."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 32))
    y = np.repeat(np.arange(5), 20)
    cal = calibrated_delta(X, y, metric="raw", seed=0, n_bootstrap=5)
    assert cal.span > 0, f"span should be positive, got {cal.span}"
    # Score can go slightly negative (data marginally less tree-like than its
    # own matched Gaussian reference — sampling noise). The important guarantee
    # is that the span is open and the score is bounded/finite.
    assert -0.2 <= cal.score <= 2.0, f"score wildly out of range, got {cal.score}"


def test_calibrated_score_high_for_tree():
    """A tree cloud should score near 1 (close to the tree anchor)."""
    tree = embedded_tree_cloud(150, 32, seed=1)
    cal = calibrated_delta(tree, metric="raw", seed=0, n_bootstrap=5)
    assert cal.score > 0.5, f"tree data should score high, got {cal.score}"


def test_cluster_null_scores_low():
    """A pure cluster mixture should have near-zero excess_over_cluster."""
    rng = np.random.default_rng(3)
    # create clean clusters
    X = np.vstack([rng.standard_normal((40, 16)) * 0.3 + i * 5 for i in range(5)])
    y = np.repeat(np.arange(5), 40)
    cal = calibrated_delta(X, y, metric="raw", seed=0, n_bootstrap=5)
    assert cal.excess_over_cluster < 0.03, (
        f"pure clusters should not exceed cluster null, excess={cal.excess_over_cluster}")
