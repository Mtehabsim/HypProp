"""Tests for the harm router (blocking + zero-shot, hyperbolic vs euclidean vs flat)."""
import numpy as np


def test_router_verdict_shape():
    from hypprobe.geometry.harm_router import _verdict
    rows = [dict(eval="blocking", arm="hyperbolic", f1=0.8, acc=0.8),
            dict(eval="blocking", arm="cond_euclidean", f1=0.7, acc=0.7),
            dict(eval="blocking", arm="flat_logreg", f1=0.75, acc=0.75),
            dict(eval="zeroshot", arm="hyperbolic", f1=0.6, acc=0.6),
            dict(eval="zeroshot", arm="cond_euclidean", f1=0.4, acc=0.4)]
    v = _verdict(rows)
    assert v["blocking_hyp_minus_flat"] == 0.05
    assert v["zeroshot_hyp_minus_euc"] == 0.2   # the payoff metric


def test_flat_baseline_runs():
    from hypprobe.geometry.harm_router import _flat_baseline
    rng = np.random.default_rng(0)
    Xtr = np.vstack([rng.normal(0, 1, (30, 8)), rng.normal(3, 1, (30, 8))])
    ytr = np.array([0] * 30 + [1] * 30)
    acc, f1 = _flat_baseline(Xtr, ytr, Xtr, ytr)
    assert acc > 0.9 and f1 > 0.9   # separable -> high
