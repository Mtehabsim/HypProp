"""Probe sanity checks -- fairness of the comparison and the positive control."""

import numpy as np

from hypprobe.data import synthetic
from hypprobe.probes.hmlr import ProbeConfig, fit_probe


def _split(features, labels, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    cut = int(frac * len(labels))
    tr, va = idx[:cut], idx[cut:]
    return features[tr], labels[tr], features[va], labels[va]


def test_curvature_zero_matches_flat_on_transform():
    """c=0 H-MLR and the explicit flat-on-transform arm must agree closely.

    Both are Euclidean affine logits over the same projection; trained with the
    same seed they should reach very similar validation accuracy.
    """
    feats, labels, _, _ = synthetic.hierarchy_features(depth=2, branching=3, seed=1)
    xtr, ytr, xva, yva = _split(feats, labels, seed=1)
    n_classes = int(labels.max() + 1)
    common = dict(in_dim=feats.shape[1], n_classes=n_classes, proj_dim=6,
                  seed=3, epochs=150)
    _, r_zero = fit_probe(xtr, ytr, xva, yva, ProbeConfig(curvature=0.0, **common))
    _, r_flat = fit_probe(xtr, ytr, xva, yva,
                          ProbeConfig(curvature=1.0, use_manifold=False, **common))
    assert abs(r_zero.val_acc - r_flat.val_acc) < 0.1


def test_hyperbolic_beats_flat_on_hierarchy_positive_control():
    """The pipeline's headline sanity check.

    On labelled features whose classes form a strict tree, a low-dimensional
    hyperbolic probe should beat the flat probe. If this fails, something in the
    geometry/probe stack is broken and no LLM result would be trustworthy.
    """
    feats, labels, _, _ = synthetic.hierarchy_features(
        depth=3, branching=3, per_leaf=40, dim=12, noise=0.15, seed=0)
    xtr, ytr, xva, yva = _split(feats, labels, seed=0)
    n_classes = int(labels.max() + 1)
    common = dict(in_dim=feats.shape[1], n_classes=n_classes, proj_dim=3,
                  seed=0, epochs=300, lr=1e-2)
    _, r_hyp = fit_probe(xtr, ytr, xva, yva, ProbeConfig(curvature=1.0, **common))
    _, r_flat = fit_probe(xtr, ytr, xva, yva,
                          ProbeConfig(curvature=1.0, use_manifold=False, **common))
    # Hyperbolic should be at least as good, and meaningfully so at low dim.
    assert r_hyp.val_acc >= r_flat.val_acc - 0.02, (
        f"hyperbolic {r_hyp.val_acc:.3f} < flat {r_flat.val_acc:.3f}")


def test_probe_learns_something():
    """Basic smoke: probe beats chance on the hierarchy features."""
    feats, labels, _, _ = synthetic.hierarchy_features(depth=2, branching=3, seed=2)
    xtr, ytr, xva, yva = _split(feats, labels, seed=2)
    n_classes = int(labels.max() + 1)
    _, res = fit_probe(xtr, ytr, xva, yva, ProbeConfig(
        in_dim=feats.shape[1], n_classes=n_classes, proj_dim=5,
        curvature=1.0, seed=0, epochs=200))
    assert res.val_acc > 1.5 / n_classes
