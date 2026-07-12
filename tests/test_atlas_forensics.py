"""Tests for the Atlas forensics ceiling check."""

import numpy as np
import pytest

from hypprobe.geometry.atlas_forensics import (
    EUCLIDEAN_CEILING,
    _pairwise,
    candidate_statistics,
    ceiling_check,
)


def test_ceiling_numeric():
    """ceiling_check must confirm the theoretical max (unit square)."""
    rng = np.random.default_rng(0)
    worst, square_val = ceiling_check(rng, n_random_trials=5000)
    # The unit square attains exactly 1 - 1/sqrt(2)
    assert abs(square_val - EUCLIDEAN_CEILING) < 1e-6
    # No random config exceeds it (within numeric tolerance)
    assert worst <= EUCLIDEAN_CEILING + 1e-4


def test_candidate_statistics_bounded():
    """eq56_delta_rel must stay below the Euclidean ceiling on random data."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((100, 16))
    stats = candidate_statistics(pts, 2000, rng)
    assert stats["eq56_delta_rel"] <= EUCLIDEAN_CEILING + 0.01


def test_atlas_reported_unattainable():
    """Confirm: 0.995 (Atlas's reported middle-layer median) > ceiling."""
    assert 0.995 > EUCLIDEAN_CEILING, (
        "The Atlas's reported values should exceed the proven ceiling")
