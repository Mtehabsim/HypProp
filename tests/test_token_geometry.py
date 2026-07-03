"""Tests for the token-level geometry analysis (point_mode token / token_type)."""

import numpy as np

from hypprobe.data.mock_activations import generate
from hypprobe.io import build_token_matrix, frequent_token_types
from hypprobe.geometry import token_geometry


def _store(tmp_path, n=60):
    out = str(tmp_path / "acts")
    generate(out, n_samples=n, with_variants=False)
    return out


def test_build_token_matrix_point_is_token(tmp_path):
    out = _store(tmp_path)
    X = build_token_matrix(out, "mock/tree-7b", "wordnet_control", layer=7)
    # Many tokens per sample -> far more rows than samples.
    assert X.ndim == 2 and X.shape[0] > 60


def test_token_filter_selects_type(tmp_path):
    out = _store(tmp_path)
    X, toks, pos, sids = build_token_matrix(
        out, "mock/tree-7b", "wordnet_control", layer=7,
        token_filter=lambda t: t == "The", with_meta=True)
    assert X.shape[0] >= 8
    assert all(t == "The" for t in toks)
    # "The" was placed at varying positions -> multiple distinct positions.
    assert len(set(pos.tolist())) >= 2


def test_frequent_token_types_includes_shared(tmp_path):
    out = _store(tmp_path)
    types = frequent_token_types(out, "mock/tree-7b", "wordnet_control",
                                 top_k=20, min_count=5)
    assert "The" in types


def test_token_geometry_runs_and_emits_axes(tmp_path):
    out = _store(tmp_path)
    outdir = str(tmp_path / "geom")
    rows = token_geometry.run(out, outdir, whiten=True, min_count=5, seed=0)
    # We should get the whole-cloud row and per-type rows.
    modes = {r["point_mode"] for r in rows}
    assert "token" in modes and "token_type" in modes
    # "The" varies in position -> a position-axis row should exist for it.
    the_axes = {r["axis"] for r in rows
                if r["token_type"] == "The" and r["point_mode"] == "token_type"}
    assert "position" in the_axes
