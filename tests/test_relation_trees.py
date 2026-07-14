"""Tests for the relation-type generality dataset (+ flat negative control)."""

from collections import Counter

from hypprobe.data.relation_trees import RELATIONS, build_relation_trees


def test_all_relations_present_matched_shape():
    rows = build_relation_trees(n_prompts=12, n_nodes=15, seed=0)
    arms = Counter(r["tree_meta"]["relation"] for r in rows)
    for rel in RELATIONS:
        assert arms[rel] == 12
    # every non-flat arm uses the SAME branching-2 tree shape -> matched difficulty
    for r in rows:
        assert r["tree_meta"]["n_nodes"] == 15
        assert r["prompt"].count(" is ") + r["prompt"].count(" causes ") >= 15


def test_flat_set_is_a_star_no_hierarchy():
    """flat_set target must be a depth-1 star (no graded tree metric to recover)."""
    rows = build_relation_trees(n_prompts=8, n_nodes=15, seed=0)
    fs = [r for r in rows if r["tree_meta"]["relation"] == "flat_set"]
    assert fs
    for r in fs:
        tm = r["tree_meta"]
        assert tm["parent"] == [-1] + [0] * (tm["n_nodes"] - 1)   # star
        assert set(tm["depth"]) == {0, 1}                          # no depth grading


def test_hierarchical_relations_have_graded_depth():
    rows = build_relation_trees(n_prompts=8, n_nodes=15, seed=0)
    for rel in ("is_a", "part_of", "causes"):
        r = [x for x in rows if x["tree_meta"]["relation"] == rel][0]
        assert max(r["tree_meta"]["depth"]) >= 2                   # real tree, not a star


def test_relation_phrasing_differs():
    rows = build_relation_trees(n_prompts=4, n_nodes=12, seed=0)
    by = {rel: [r for r in rows if r["tree_meta"]["relation"] == rel][0]["prompt"]
          for rel in RELATIONS}
    assert "is a" in by["is_a"]
    assert "part of" in by["part_of"]
    assert "causes" in by["causes"]
    assert "is an item" in by["flat_set"]


def test_registered_as_builder():
    from hypprobe.data.prepare import BUILDERS
    assert "relation_trees" in BUILDERS
