"""Tests for the label-alignment metric (joint selection, Phase 1)."""

import numpy as np

from hypprobe.data import synthetic
from hypprobe.geometry.label_alignment import (label_alignment,
                                               tree_distance_from_paths)


def test_tree_distance_from_paths():
    paths = [[0, 0], [0, 1], [1, 2]]
    d, uniq = tree_distance_from_paths(paths)
    assert len(uniq) == 3
    # Same domain (0,*) closer than cross-domain.
    assert d[0, 1] < d[0, 2]
    assert d[0, 0] == 0.0


def test_alignment_high_on_true_hierarchy():
    """On strictly-hierarchical features, prototype distances track the tree."""
    feats, labels, _, paths_list = synthetic.hierarchy_features(
        depth=3, branching=3, per_leaf=30, dim=12, noise=0.1, seed=0)
    # Build a label_path per sample from the leaf label's true path.
    label_paths = []
    for lab in labels:
        p = paths_list[lab]
        label_paths.append([p[0], lab])  # (domain, leaf)
    res = label_alignment(feats, label_paths, curvature=1.0, do_whiten=False)
    # The taxonomy should be recovered (positive correlation), and reasonably strong.
    assert res.proto_corr_hyperbolic > 0.3
    assert res.n_classes > 3


def test_alignment_low_on_random():
    """Random features carry no taxonomy structure -> weak alignment."""
    rng = np.random.default_rng(0)
    n, dim = 300, 12
    feats = rng.standard_normal((n, dim))
    # 9 classes with random paths unrelated to the features.
    label_paths = [[int(i % 3), int(i % 9)] for i in range(n)]
    res = label_alignment(feats, label_paths, curvature=1.0, do_whiten=True)
    assert abs(res.proto_corr_hyperbolic) < 0.5  # no strong structure
