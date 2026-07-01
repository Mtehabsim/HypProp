"""Geometry sanity checks -- the plan's Phase-1 verification gates."""

import numpy as np
import torch

from hypprobe.data import synthetic
from hypprobe.geometry import poincare
from hypprobe.geometry.delta import delta_hyperbolicity, delta_from_distance_matrix


def test_expmap_logmap_roundtrip():
    """log(exp(v)) == v within tolerance for several curvatures."""
    torch.manual_seed(0)
    v = torch.randn(64, 8) * 0.3
    for c in [0.0, 0.5, 1.0, 2.0]:
        y = poincare.expmap0(v, c)
        v2 = poincare.logmap0(y, c)
        assert torch.allclose(v, v2, atol=1e-4), f"roundtrip failed at c={c}"


def test_points_stay_in_ball():
    """exp map output must satisfy sqrt(c)||x|| < 1."""
    torch.manual_seed(0)
    v = torch.randn(128, 6) * 5.0  # large tangent vectors
    for c in [0.5, 1.0, 3.0]:
        y = poincare.expmap0(v, c)
        norms = y.norm(dim=-1)
        assert torch.all(norms < 1.0 / np.sqrt(c)), f"escaped ball at c={c}"


def test_curvature_zero_is_euclidean():
    """At c=0, geodesic distance == Euclidean distance."""
    torch.manual_seed(0)
    x = torch.randn(32, 5) * 0.2
    y = torch.randn(32, 5) * 0.2
    d_hyp = poincare.dist(x, y, 0.0)
    d_euc = (x - y).norm(dim=-1)
    assert torch.allclose(d_hyp, d_euc, atol=1e-6)


def test_gyroplane_zero_curvature_is_affine():
    """At c=0, gyroplane logits equal the affine logit 2<x-p,a>/||a||."""
    torch.manual_seed(0)
    x = torch.randn(10, 4)
    p = torch.randn(3, 4)
    a = torch.randn(3, 4)
    got = poincare.gyroplane_distance(x, p, a, 0.0)
    diff = x.unsqueeze(1) - p.unsqueeze(0)
    inner = (diff * a.unsqueeze(0)).sum(-1)
    expect = 2.0 * inner / a.norm(dim=-1).unsqueeze(0)
    assert torch.allclose(got, expect, atol=1e-5)


def test_delta_of_tree_is_small():
    """A true tree metric has delta_rel ~ 0."""
    _, tree_d, _ = synthetic.balanced_tree(depth=5, branching=2)
    res = delta_from_distance_matrix(tree_d, n_quadruples=2000, n_repeats=3,
                                     rng=np.random.default_rng(0))
    assert res.delta_rel < 0.05, f"tree delta_rel too high: {res.delta_rel}"


def test_delta_of_random_is_high():
    """A flat Gaussian cloud is far from tree-like (high delta_rel)."""
    x = synthetic.random_gaussian(n=400, dim=16)
    res = delta_hyperbolicity(x, n_quadruples=2000, n_repeats=3, do_whiten=True)
    assert res.delta_rel > 0.2, f"random delta_rel too low: {res.delta_rel}"


def test_tree_more_hyperbolic_than_random():
    """Ordering check: tree is strictly more tree-like than random points."""
    _, tree_d, _ = synthetic.balanced_tree(depth=5, branching=2)
    tree_res = delta_from_distance_matrix(tree_d, n_quadruples=2000, n_repeats=3,
                                          rng=np.random.default_rng(0))
    rand = synthetic.random_gaussian(n=400, dim=16)
    rand_res = delta_hyperbolicity(rand, n_quadruples=2000, n_repeats=3, do_whiten=True)
    assert tree_res.delta_rel < rand_res.delta_rel
