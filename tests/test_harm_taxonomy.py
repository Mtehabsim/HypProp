"""Tests for the harm-detection dataset (hazard taxonomy + binary views)."""

import numpy as np

from hypprobe.data.harm_taxonomy import _row, _safe_id
from hypprobe.geometry.structural_probe import _depth_target, _taxonomy_target


def test_safe_id_removes_path_breakers():
    # Aegis category 'Controlled/Regulated Substances' has a slash that broke
    # the extractor's file path in run5.
    sid = _safe_id("aegis_PhysicalHarm_Controlled/Regulated Substances_3")
    assert "/" not in sid and " " not in sid
    assert sid == "aegis_PhysicalHarm_Controlled_Regulated_Substances_3"


def test_row_schema():
    r = _row("x", "some text", "PhysicalHarm", "Weapons", 1)
    assert set(r) >= {"sample_id", "prompt", "label", "label_path", "answer",
                      "harm_supercategory", "harm_category"}
    assert len(r["label_path"]) == 2
    assert r["label"] == r["answer"] == 1


def test_taxonomy_target_is_graded_not_star():
    """The taxonomy view must be a graded 2-level tree (within-cat 0, within-super
    2, cross-super 4) — a hashed leaf id made it a degenerate star (bug caught in
    CPU validation), which would flatten the hierarchy-vs-binary contrast."""
    rows = ([_row(f"w{i}", "x", "PhysicalHarm", "Weapons", 1) for i in range(2)]
            + [_row("v", "x", "PhysicalHarm", "Violence", 1)]
            + [_row("s", "x", "SelfHarm", "Suicide and Self Harm", 1)]
            + [_row("safe", "x", "Safe", "Safe", 0)])

    class S(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    samples = [S(label=r["answer"], label_path=r["label_path"]) for r in rows]
    T = _taxonomy_target(samples)
    assert T[0, 1] == 0            # two Weapons -> same category
    assert T[0, 2] == 2            # Weapons vs Violence -> same super, diff cat
    assert T[0, 3] == 4            # PhysicalHarm vs SelfHarm -> cross super
    assert T.max() > T[T > 0].min()  # graded (not all equal = not a star)


def test_binary_target_is_two_class():
    rows = [_row("a", "x", "PhysicalHarm", "Weapons", 1),
            _row("b", "x", "Safe", "Safe", 0)]

    class S(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    samples = [S(label=r["answer"], label_path=r["label_path"]) for r in rows]
    B = _depth_target(samples)
    assert B[0, 1] == 1 and B[0, 0] == 0     # unsafe vs safe = distance 1


def test_registered():
    from hypprobe.data.prepare import BUILDERS
    assert "harm_taxonomy" in BUILDERS
    assert "harm_beavertails" in BUILDERS
