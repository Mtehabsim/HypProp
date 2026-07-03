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


def test_hyperbolic_wins_on_structural_distortion_positive_control():
    """The pipeline's headline sanity check -- on the RIGHT task.

    Hyperbolic geometry's established advantage (Nickel-Kiela, Ganea) is
    low-distortion embedding of TREE STRUCTURE, not flat leaf classification.
    So the positive control is a structural distance-fidelity task: fit a probe
    so pairwise distances match the taxonomy tree distances, and check the
    hyperbolic probe achieves clearly lower distortion than the Euclidean one.
    If this fails, the geometry/probe stack is broken.
    """
    from hypprobe.geometry.structural_probe import fit_structural

    feats, labels, tree_d, _ = synthetic.hierarchy_features(
        depth=3, branching=3, per_leaf=1, dim=12, noise=0.0, seed=0)
    rho_e, dist_e = fit_structural(feats, tree_d, 0.0, proj_dim=5, epochs=500, seed=0)
    rho_h, dist_h = fit_structural(feats, tree_d, 1.0, proj_dim=5, epochs=500, seed=0)
    assert dist_h < dist_e, (
        f"hyperbolic distortion {dist_h:.3f} not < euclidean {dist_e:.3f}")


def test_hyperbolic_does_not_lose_on_flat_classification():
    """Companion honesty check: on FLAT leaf classification (not a structural
    task) hyperbolic is expected only to TIE, not beat, the flat probe. We assert
    it does not badly underperform -- documenting that flat classification is not
    where curvature helps (consistent with the plan and the literature)."""
    feats, labels, _, _ = synthetic.hierarchy_features(
        depth=3, branching=3, per_leaf=60, dim=12, noise=0.10, seed=0)
    xtr, ytr, xva, yva = _split(feats, labels, seed=0)
    n_classes = int(labels.max() + 1)
    common = dict(in_dim=feats.shape[1], n_classes=n_classes, proj_dim=5,
                  seed=0, epochs=800, lr=5e-2)
    _, r_hyp = fit_probe(xtr, ytr, xva, yva, ProbeConfig(curvature=1.0, **common))
    _, r_flat = fit_probe(xtr, ytr, xva, yva,
                          ProbeConfig(curvature=1.0, use_manifold=False, **common))
    # Tie within a reasonable band; hyperbolic must not collapse.
    assert r_hyp.val_acc >= r_flat.val_acc - 0.08, (
        f"hyperbolic {r_hyp.val_acc:.3f} collapsed vs flat {r_flat.val_acc:.3f}")


def test_probe_learns_something():
    """Basic smoke: probe beats chance on the hierarchy features."""
    feats, labels, _, _ = synthetic.hierarchy_features(depth=2, branching=3, seed=2)
    xtr, ytr, xva, yva = _split(feats, labels, seed=2)
    n_classes = int(labels.max() + 1)
    _, res = fit_probe(xtr, ytr, xva, yva, ProbeConfig(
        in_dim=feats.shape[1], n_classes=n_classes, proj_dim=5,
        curvature=1.0, seed=0, epochs=200))
    assert res.val_acc > 1.5 / n_classes
