"""Tests for the Rung-0 metric family and adjudication harness."""

import numpy as np

from hypprobe.data import synthetic
from hypprobe.data.mock_activations import generate
from hypprobe.geometry.delta import delta_hyperbolicity, fit_background
from hypprobe.geometry import rung0


def test_raw_metric_matches_do_whiten_false():
    """metric='raw' must equal the legacy do_whiten=False path."""
    x = synthetic.random_gaussian(n=200, dim=16, seed=0)
    a = delta_hyperbolicity(x, metric="raw", seed=0, n_repeats=3)
    b = delta_hyperbolicity(x, do_whiten=False, seed=0, n_repeats=3)
    assert abs(a.delta_rel - b.delta_rel) < 1e-9


def test_anisotropy_fakes_tree_then_whitening_reveals_it():
    """The core Rung-0 phenomenon: a rogue high-variance dim makes raw delta look
    tree-like; whitening (per_cloud/background) removes the fake tree-likeness."""
    feats, _, _, _ = synthetic.hierarchy_features(depth=3, branching=3, per_leaf=8,
                                                  dim=32, noise=0.1, seed=0)
    rng = np.random.default_rng(0)
    rogue = rng.standard_normal((feats.shape[0], 1)) * 50.0
    aniso = np.concatenate([feats, rogue], axis=1)

    raw = delta_hyperbolicity(aniso, metric="raw", seed=0, n_repeats=3)
    per = delta_hyperbolicity(aniso, metric="per_cloud", seed=0, n_repeats=3)
    assert raw.delta_rel < 0.1, "rogue dim should fake tree-likeness in raw"
    assert per.delta_rel > raw.delta_rel + 0.1, "whitening should reveal the artifact"


def test_bootstrap_std_is_populated():
    x = synthetic.random_gaussian(n=150, dim=16, seed=0)
    r = delta_hyperbolicity(x, metric="raw", seed=0, n_repeats=2, n_bootstrap=15)
    assert r.bootstrap_std > 0.0


def test_background_requires_transform():
    x = synthetic.random_gaussian(n=100, dim=16, seed=0)
    try:
        delta_hyperbolicity(x, metric="background", seed=0)
        assert False, "should have raised without a bg_transform"
    except ValueError:
        pass


def test_pca_only_does_not_rescale():
    """pca_only projects but must NOT equalise variance (differs from per_cloud)."""
    feats, _, _, _ = synthetic.hierarchy_features(depth=3, branching=3, per_leaf=8,
                                                  dim=32, noise=0.1, seed=0)
    rng = np.random.default_rng(0)
    aniso = np.concatenate([feats, rng.standard_normal((feats.shape[0], 1)) * 50.0], axis=1)
    po = delta_hyperbolicity(aniso, metric="pca_only", seed=0, n_repeats=3)
    pc = delta_hyperbolicity(aniso, metric="per_cloud", seed=0, n_repeats=3)
    # pca_only keeps the rogue variance -> still low; per_cloud rescales -> higher.
    assert po.delta_rel < pc.delta_rel


def test_rung0_harness_recovers_planted_answer_on_mock(tmp_path):
    """End-to-end: on the mock store, controls must behave (tree low, gaussian
    high) -- proving the harness/instruments work before any DGX run."""
    acts = str(tmp_path / "acts")
    generate(acts, n_samples=40, with_variants=False)
    out = str(tmp_path / "geom")
    rows = rung0.run(acts, out, seed=0, n_bootstrap=10, pca_cap=32,
                     project_root=str(tmp_path))  # no PREREGISTER -> defaults
    assert rows
    tree = [r for r in rows if r["cloud_kind"] == "tree_control"]
    gauss = [r for r in rows if r["cloud_kind"] == "gaussian_control"]
    assert tree and gauss
    # The estimator must be calibrated: a TRUE tree metric ~ 0, gaussian clearly high.
    assert np.mean([r["delta_rel"] for r in tree]) < 0.15
    assert np.mean([r["delta_rel"] for r in gauss]) > 0.15
    import os
    assert os.path.exists(os.path.join(out, "rung0.csv"))
    assert os.path.exists(os.path.join(out, "rung0_verdict.md"))
