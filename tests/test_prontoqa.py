"""Tests for the PrOntoQA reasoning-eliciting builder."""

from collections import Counter

from hypprobe.data.prepare import build_prontoqa


def test_balanced_across_depths():
    rows = build_prontoqa(n_per_depth=20, seed=0)
    by_depth = Counter(r["label"] for r in rows)
    assert set(by_depth) == {1, 2, 3, 4, 5}
    assert all(c == 20 for c in by_depth.values())


def test_prompts_are_multihop_and_elicit_reasoning():
    rows = build_prontoqa(n_per_depth=10, seed=0)
    for r in rows:
        # every example asks for step-by-step reasoning
        assert "step by step" in r["prompt"].lower()
        # depth d means d chained 'Every X is a Y' facts
        assert r["prompt"].count("Every ") == r["label"]


def test_label_path_is_graded():
    rows = build_prontoqa(n_per_depth=10, seed=0)
    for r in rows:
        coarse, depth = r["label_path"]
        assert depth == r["label"]
        assert coarse == (0 if depth <= 2 else 1)   # shallow vs deep grouping


def test_balanced_true_false():
    rows = build_prontoqa(n_per_depth=40, seed=1)
    ans = Counter(r["answer"] for r in rows)
    # roughly balanced true/false (half each by construction)
    assert abs(ans[0] - ans[1]) <= len(rows) * 0.1


def test_registered_as_builder():
    from hypprobe.data.prepare import BUILDERS
    assert "prontoqa" in BUILDERS
