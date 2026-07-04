"""Tests for attention-sink stripping (the fix for the delta_rel degeneracy)."""

import numpy as np

from hypprobe.io import _sink_mask, pool_features, SINK_NORM_MULT


def _cloud_with_sink(n=200, dim=64, sink_mult=245.0, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, dim))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    h *= 60.0
    sink = rng.standard_normal(dim)
    sink = sink / np.linalg.norm(sink) * (60.0 * sink_mult)
    return np.vstack([sink[None, :], h])   # sink at position 0


def test_sink_mask_drops_exactly_the_sink():
    H = _cloud_with_sink()
    keep = _sink_mask(H)
    assert keep.sum() == H.shape[0] - 1      # exactly one dropped
    assert not keep[0]                        # and it's the sink at position 0


def test_sink_mask_multiplier_insensitive():
    """A principled threshold: the huge sink/normal gap means any reasonable
    multiplier drops exactly the sink."""
    H = _cloud_with_sink()
    for mult in (5.0, 10.0, 20.0, 50.0):
        assert (~_sink_mask(H, mult)).sum() == 1


def test_sink_mask_noop_when_no_outlier():
    """An isotropic cloud with no sink should keep everything."""
    rng = np.random.default_rng(1)
    H = rng.standard_normal((100, 32))
    assert _sink_mask(H).all()


def test_pool_features_drops_sink_and_changes_result():
    """pool_features(drop_sink=True) must differ from drop_sink=False when a sink
    is present -- proving the sink is actually removed before pooling."""
    import torch

    H = _cloud_with_sink(n=50, dim=16)
    n_tok = H.shape[0]
    sample = {
        "hidden": torch.tensor(H[None, :, :], dtype=torch.float32),  # 1 layer
        "is_generated": torch.zeros(n_tok, dtype=torch.bool),        # all "input"
        "is_thinking": torch.zeros(n_tok, dtype=torch.bool),
    }
    with_sink = pool_features(sample, 0, "all", drop_sink=False)
    without = pool_features(sample, 0, "all", drop_sink=True)
    # the sink dominates the mean, so dropping it moves the pooled vector a lot
    assert np.linalg.norm(with_sink - without) > 1.0
    # and the de-sinked pooled norm is near the normal-token scale (~60), not ~14000
    assert np.linalg.norm(without) < 5 * 60


def test_last_source_never_empty_from_sink():
    """'last' pools a single token and must not be nulled by sink-dropping."""
    import torch
    H = _cloud_with_sink(n=20, dim=16)
    sample = {
        "hidden": torch.tensor(H[None, :, :], dtype=torch.float32),
        "is_generated": torch.zeros(H.shape[0], dtype=torch.bool),
        "is_thinking": torch.zeros(H.shape[0], dtype=torch.bool),
    }
    assert pool_features(sample, 0, "last", drop_sink=True) is not None
