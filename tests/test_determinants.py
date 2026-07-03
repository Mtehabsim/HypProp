"""Tests for the determinants module (previously untested -- review gap #1)."""

import numpy as np

from hypprobe.data.mock_activations import generate
from hypprobe.geometry import determinants as det


def _raw_tokens(n_samples=20, n_tokens=10, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_samples):
        h = rng.standard_normal((n_tokens, dim))
        is_think = np.zeros(n_tokens, bool)
        is_think[n_tokens // 2] = True
        out.append((h, is_think))
    return out


def test_placebo_is_exact_noop():
    """The placebo edit must reproduce the base pooling EXACTLY (zero change)."""
    rt = _raw_tokens()
    base = det._pooled_base(rt)
    placebo = det._edit_placebo(rt, np.random.default_rng(1))
    assert np.allclose(base, placebo), "placebo must equal base pooling"


def test_all_edits_share_pooling_shape():
    """Every edit must yield the same number of points and dims as the base
    (shared pooling operator -- no point-set/pooling confound)."""
    rt = _raw_tokens()
    rng = np.random.default_rng(0)
    base = det._pooled_base(rt)
    for fn in (det._edit_token_identity, det._edit_order_shuffle, det._edit_placebo):
        X = fn(rt, rng)
        assert X.shape == base.shape, f"{fn.__name__} changed shape {X.shape} vs {base.shape}"


def test_order_edit_is_not_trivially_null():
    """Order-shuffle uses shuffled POSITION WEIGHTS on the same shared pooling, so
    it actually changes the pooled vectors (unlike shuffling a plain mean, which
    would be a guaranteed null)."""
    rt = _raw_tokens()
    base = det._pooled_base(rt)
    shuffled = det._edit_order_shuffle(rt, np.random.default_rng(3))
    assert not np.allclose(base, shuffled), "order edit must not be a no-op"


def test_run_gates_driver_on_placebo_and_std(tmp_path):
    """End-to-end: attribution.csv has a placebo row (change 0) and the driver
    flag is only set when the change beats both std_rel and the placebo."""
    acts = str(tmp_path / "acts")
    generate(acts, n_samples=60, with_variants=True)
    rows = det.run(acts, str(tmp_path / "det"), whiten=True, source="thinking", seed=0)
    assert rows
    placebo = [r for r in rows if r["edit"] == "placebo"]
    assert placebo and abs(placebo[0]["delta_change"]) < 1e-6
    # Any row flagged as driver must exceed both its std_rel and the placebo mag.
    for r in rows:
        if r.get("is_driver"):
            assert abs(r["delta_change"]) > r["std_rel"]
            assert abs(r["delta_change"]) >= r["placebo_mag"]
