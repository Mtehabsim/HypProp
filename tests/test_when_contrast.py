"""Tests for the WHEN H1/H2/H3 scorer (reads rung0.csv-style rows)."""

from hypprobe.geometry import when_contrast

TH = {"h1_margin": 0.05, "h1_min_effect_over_boot": 2.0, "h2_gap_margin": 0.05}


def _row(model, layer, source, metric, delta, floor=0.005, kind="data"):
    return dict(model=model, dataset="d", layer=layer, token_source=source,
                metric=metric, delta_rel=delta, noise_floor=floor, cloud_kind=kind)


def _two_model_rows(gen_reasoning, gen_base):
    """Final layer = 1; give prompt=0.60 for both models, vary generated delta."""
    rows = []
    for model, gd in (("deepseek-r1-distill-qwen-7b", gen_reasoning),
                      ("qwen2.5-7b", gen_base)):
        rows.append(_row(model, 1, "input", "background", 0.60))
        rows.append(_row(model, 1, "generated", "background", gd))
        rows.append(_row(model, 1, "thinking", "background", gd - 0.02))
    return rows


def test_h1_passes_when_generated_more_treelike():
    # reasoning model: generated far below prompt (0.40 vs 0.60) -> PASS
    rows = _two_model_rows(gen_reasoning=0.40, gen_base=0.58)
    res = when_contrast.score(rows, TH)
    h1 = {r["model"]: r["verdict"] for r in res["h1"]}
    assert h1["deepseek-r1-distill-qwen-7b"] == "PASS"


def test_h1_fails_when_generated_not_lower():
    rows = _two_model_rows(gen_reasoning=0.62, gen_base=0.61)  # generated >= prompt
    res = when_contrast.score(rows, TH)
    # powered (floor 0.005 < 0.02 target) + diff<=0 -> clean FAIL
    assert all(r["verdict"] == "FAIL" for r in res["h1"])


def _two_model_rows_floor(gen_reasoning, gen_base, floor):
    rows = []
    for model, gd in (("deepseek-r1-distill-qwen-7b", gen_reasoning),
                      ("qwen2.5-7b", gen_base)):
        rows.append(_row(model, 1, "input", "background", 0.60, floor=floor))
        rows.append(_row(model, 1, "generated", "background", gd, floor=floor))
    return rows


def test_underpowered_ambiguous_flagged_not_null():
    # small positive diff (0.04 < 0.05 margin) AND high floor (0.05 > 0.02 target)
    rows = _two_model_rows_floor(gen_reasoning=0.56, gen_base=0.56, floor=0.05)
    res = when_contrast.score(rows, TH)
    assert all(r["underpowered"] for r in res["h1"])
    assert all("underpowered" in r["verdict"] for r in res["h1"])


def test_powered_null_is_clean_fail():
    # generated NOT lower, low floor -> genuine powered FAIL (not underpowered)
    rows = _two_model_rows_floor(gen_reasoning=0.61, gen_base=0.61, floor=0.004)
    res = when_contrast.score(rows, TH)
    assert all(not r["underpowered"] for r in res["h1"])
    assert all(r["verdict"] == "FAIL" for r in res["h1"])


def test_h2_detects_reasoning_specificity():
    # reasoning gap large (0.20), base gap tiny (0.02) -> margin 0.18 > 0.05 -> PASS
    rows = _two_model_rows(gen_reasoning=0.40, gen_base=0.58)
    res = when_contrast.score(rows, TH)
    assert res["h2"] is not None
    assert res["h2"]["verdict"] == "PASS"
    assert res["h2"]["margin"] > TH["h2_gap_margin"]


def test_h2_null_when_gaps_similar():
    rows = _two_model_rows(gen_reasoning=0.55, gen_base=0.56)  # both small gaps
    res = when_contrast.score(rows, TH)
    assert res["h2"]["verdict"].startswith("FAIL")


def test_reasoning_classifier():
    assert when_contrast._is_reasoning("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    assert not when_contrast._is_reasoning("Qwen/Qwen2.5-7B")


def test_run_writes_verdict(tmp_path):
    import csv, os
    geom = tmp_path / "geometry"
    geom.mkdir()
    rows = _two_model_rows(gen_reasoning=0.40, gen_base=0.58)
    cols = ["model", "dataset", "layer", "token_source", "metric",
            "delta_rel", "noise_floor", "cloud_kind"]
    with open(geom / "rung0.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    when_contrast.run(str(geom), project_root=str(tmp_path))
    assert os.path.exists(geom / "when_verdict.md")
