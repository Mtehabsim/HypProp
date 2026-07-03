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
    """At c=0, gyroplane logits equal the affine logit 4<x-p,a>/||a||.

    The constant is 4 because that is the true c->0 limit of the gyroplane
    formula (2/sqrt(c) * arcsinh(2 sqrt(c) ...) -> 4<.>); see the convergence
    test below. It only needs to be a constant multiple of the LR logit for the
    fairness identity, but we pin the exact limit so the c=0 branch equals the
    c>0 path in the limit.
    """
    torch.manual_seed(0)
    x = torch.randn(10, 4)
    p = torch.randn(3, 4)
    a = torch.randn(3, 4)
    got = poincare.gyroplane_distance(x, p, a, 0.0)
    diff = x.unsqueeze(1) - p.unsqueeze(0)
    inner = (diff * a.unsqueeze(0)).sum(-1)
    expect = 4.0 * inner / a.norm(dim=-1).unsqueeze(0)
    assert torch.allclose(got, expect, atol=1e-5)


def test_gyroplane_limit_c_to_zero_converges_to_flat():
    """#9: the c->0 LIMIT of the gyroplane formula must approach the c=0 branch.

    The fairness claim ("only geometry differs; c->0 == logistic regression")
    rests on this limit, not just on the code's c<=0 shortcut. We check the
    actual hyperbolic formula at tiny curvature converges to the flat affine
    logit. Points are kept small so they sit well inside the ball.
    """
    torch.manual_seed(0)
    x = torch.randn(10, 4) * 0.05
    p = torch.randn(3, 4) * 0.05
    a = torch.randn(3, 4)
    flat = poincare.gyroplane_distance(x, p, a, 0.0)
    prev = None
    for c in [1e-1, 1e-2, 1e-3, 1e-4]:
        cur = poincare.gyroplane_distance(x, p, a, c)
        err = (cur - flat).abs().max().item()
        if prev is not None:
            assert err <= prev + 1e-6, f"not converging at c={c} (err {err} > {prev})"
        prev = err
    # At very small curvature the two must be close.
    assert (poincare.gyroplane_distance(x, p, a, 1e-4) - flat).abs().max() < 1e-2


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


def test_whiten_reduces_dim_in_high_d_regime():
    """#4: whitening N<<d PCA-reduces to <= N//3 columns (not full d)."""
    from hypprobe.geometry.delta import whiten
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 3584))  # N=60 << d=3584
    w = whiten(x)
    assert w.shape[0] == 60
    assert w.shape[1] <= 60 // 3 + 1, f"expected PCA reduction, got {w.shape[1]} cols"


def test_delta_survives_whitening_on_tree_in_high_d():
    """#4 regression: a tree embedded in high-d must NOT be forced to 'flat'.

    Before the fix, whitening N<<d points biased them toward equidistance, so a
    genuinely tree-like set would score high delta_rel (false 'flat'). After PCA-
    then-whiten, the tree stays clearly more tree-like than random noise in the
    SAME dimension and point count.
    """
    _, tree_d, _ = synthetic.balanced_tree(depth=5, branching=2)  # exact tree metric
    # Embed those tree distances into high-d via random Gaussian features is not
    # exact; instead compare delta_rel of a low-d tree-realising cloud vs random,
    # both whitened, at N<<d.
    rng = np.random.default_rng(0)
    n, d = 90, 512
    # Random flat cloud.
    flat = rng.standard_normal((n, d))
    flat_res = delta_hyperbolicity(flat, n_repeats=3, do_whiten=True, seed=0)
    # Tree-like cloud: hierarchical prototypes + noise in high-d.
    feats, _, _, _ = synthetic.hierarchy_features(depth=3, branching=3, per_leaf=10,
                                                  dim=d, noise=0.1, seed=0)
    tree_res = delta_hyperbolicity(feats, n_repeats=3, do_whiten=True, seed=0)
    assert tree_res.delta_rel < flat_res.delta_rel, (
        f"tree {tree_res.delta_rel:.3f} not < flat {flat_res.delta_rel:.3f} after whitening")


def test_tree_more_hyperbolic_than_random():
    """Ordering check: tree is strictly more tree-like than random points."""
    _, tree_d, _ = synthetic.balanced_tree(depth=5, branching=2)
    tree_res = delta_from_distance_matrix(tree_d, n_quadruples=2000, n_repeats=3,
                                          rng=np.random.default_rng(0))
    rand = synthetic.random_gaussian(n=400, dim=16)
    rand_res = delta_hyperbolicity(rand, n_quadruples=2000, n_repeats=3, do_whiten=True)
    assert tree_res.delta_rel < rand_res.delta_rel
