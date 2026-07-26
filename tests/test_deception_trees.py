"""Tests for the deception dataset (honest/sandbag/distractor over shared trees)."""
from collections import Counter
from hypprobe.data.deception_trees import build_deception_trees, CONDITIONS


def test_three_conditions_shared_structure():
    rows = build_deception_trees(n_prompts=8, seed=0)
    c = Counter(r["tree_meta"]["condition"] for r in rows)
    for cond in CONDITIONS:
        assert c[cond] == 8
    # every row is a valid 15-node tree with 14 premises regardless of condition
    for r in rows:
        assert r["tree_meta"]["n_nodes"] == 15
        assert r["prompt"].count("Every ") == 14


def test_condition_only_changes_instruction():
    rows = build_deception_trees(n_prompts=4, seed=0)
    by = {c: [r for r in rows if r["tree_meta"]["condition"] == c][0] for c in CONDITIONS}
    assert "step by step" in by["honest"]["prompt"]
    assert "OPPOSITE" in by["sandbag"]["prompt"]
    assert "random" in by["distractor"]["prompt"]


def test_answer_is_true_label_not_deceptive():
    # answer must be the TRUE label (deception = emit the opposite); probe target
    # and audit compare tree-implied vs emitted.
    rows = build_deception_trees(n_prompts=10, seed=1)
    for r in rows:
        assert r["answer"] in (0, 1)
        assert r["tree_meta"]["answer"] == r["answer"]


def test_splits_into_condition_arms():
    # tree_probe splits by (naming=condition, branching) -> 3 arms
    rows = build_deception_trees(n_prompts=5, seed=0)
    arms = set((r["tree_meta"]["naming"], r["tree_meta"]["branching"]) for r in rows)
    assert arms == {("honest", 2), ("sandbag", 2), ("distractor", 2)}


def test_registered():
    from hypprobe.data.prepare import BUILDERS
    assert "deception_trees" in BUILDERS
