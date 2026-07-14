"""Tests for the branching-ontology generator (PREREGISTER3)."""

import numpy as np

from hypprobe.data.prontoqa_tree import (_build_tree, _tree_distance,
                                         build_prontoqa_tree,
                                         build_prontoqa_tree_all)


def test_branching_controls_shape_at_matched_node_count():
    for b in (1, 2, 3):
        parent, depth = _build_tree(b, 15)
        assert len(parent) == 15                       # matched node count
        # every non-root has a parent < itself (valid tree)
        assert all(0 <= parent[i] < i for i in range(1, 15))


def test_b1_is_a_path():
    """Branching 1 must be a linear chain — the negative control where a path
    embeds isometrically in 1-D and hyperbolic can buy nothing."""
    parent, depth = _build_tree(1, 12)
    assert parent == [-1] + list(range(11))            # 0<-1<-2<-...
    D = _tree_distance(parent)
    # path distances are |i-j|
    assert D[0, 11] == 11
    assert int(D.max()) == 11                          # diameter = n-1 (a path)


def test_branching_reduces_diameter():
    """Higher branching at matched node count -> shallower tree -> smaller diameter."""
    d1 = int(_tree_distance(_build_tree(1, 15)[0]).max())
    d2 = int(_tree_distance(_build_tree(2, 15)[0]).max())
    d3 = int(_tree_distance(_build_tree(3, 15)[0]).max())
    assert d1 > d2 >= d3                               # path is deepest


def test_tree_meta_retained_and_consistent():
    rows = build_prontoqa_tree(branching=2, naming="fictional", n_nodes=15,
                               n_prompts=10, seed=0)
    for r in rows:
        tm = r["tree_meta"]
        assert set(tm) >= {"branching", "naming", "n_nodes", "names", "parent",
                           "depth", "entity", "entity_node", "target_node", "answer"}
        assert len(tm["names"]) == tm["n_nodes"] == len(tm["parent"]) == len(tm["depth"])
        # premises = one per edge = n_nodes - 1
        assert r["prompt"].count("Every ") == tm["n_nodes"] - 1
        # names are unique (so concept alignment is unambiguous)
        assert len(set(tm["names"])) == len(tm["names"])


def test_answer_matches_ancestry():
    """answer=1 iff the queried target is a proper ancestor of the entity's leaf."""
    from hypprobe.data.prontoqa_tree import _ancestors
    rows = build_prontoqa_tree(branching=2, naming="fictional", n_nodes=15,
                               n_prompts=40, seed=1)
    for r in rows:
        tm = r["tree_meta"]
        anc = _ancestors(tm["entity_node"], tm["parent"])
        is_anc = tm["target_node"] in anc
        assert bool(tm["answer"]) == is_anc


def test_real_naming_uses_real_words():
    rows = build_prontoqa_tree(branching=2, naming="real", n_nodes=12,
                               n_prompts=5, seed=0)
    real_words = {"animal", "mammal", "dog", "cat", "bird", "vehicle", "car",
                  "plant", "tree", "fish", "reptile", "carnivore", "primate"}
    hit = 0
    for r in rows:
        if any(w in real_words for w in r["tree_meta"]["names"]):
            hit += 1
    assert hit == len(rows)                            # every real prompt has real names


def test_all_conditions_present():
    rows = build_prontoqa_tree_all(n_prompts=6, n_nodes=15, seed=0)
    from collections import Counter
    arms = Counter((r["tree_meta"]["naming"], r["tree_meta"]["branching"]) for r in rows)
    assert arms[("fictional", 1)] == 6
    assert arms[("fictional", 2)] == 6
    assert arms[("fictional", 3)] == 6
    assert arms[("real", 2)] == 6


def test_registered_as_builder():
    from hypprobe.data.prepare import BUILDERS
    assert "prontoqa_tree" in BUILDERS
