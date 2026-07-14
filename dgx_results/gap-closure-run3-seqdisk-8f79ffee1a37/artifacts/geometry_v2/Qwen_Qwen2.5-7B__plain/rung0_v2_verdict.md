# Rung 0 v2 verdict

Thresholds source: **PREREGISTER2.md**
`{"parity_min_plateau": 0.2, "span_effect_frac": 0.15, "min_effect_over_floor": 3.0, "h1_alpha": 0.05, "gate_a_flat_frac": 0.5, "gate_a_tree_frac": 0.5}`

## Qwen/Qwen2.5-7B / prontoqa

**Gate 0 (Atlas-object raw plateau)**: 0.080 (threshold 0.2; Euclidean-formula ceiling is 0.293 — see atlas_forensics) -> **OBJECT/EXTRACTION MISMATCH**

**Gate A (regime-matched anchors keep the span open under every metric)**: **FAIL**
  - L0/input/raw: span collapsed (flat <= tree)
  - L0/generated/raw: span collapsed (flat <= tree)
  - L0/thinking/raw: span collapsed (flat <= tree)
  - L0/last/raw: span collapsed (flat <= tree)
  - L0/all/raw: span collapsed (flat <= tree)
  - L1/input/raw: span collapsed (flat <= tree)
  - L1/generated/raw: span collapsed (flat <= tree)
  - L1/thinking/raw: span collapsed (flat <= tree)

**Gate B (span-relative drop, per source x metric)**

- input/background: plateau_score=-1.513 final_score=-1.503 drop=+0.011 (floor 0.100, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- input/per_cloud: plateau_score=+0.211 final_score=+0.150 drop=-0.061 (floor 0.043, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/background: plateau_score=-1.142 final_score=-1.388 drop=-0.245 (floor 0.107, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/per_cloud: plateau_score=+0.373 final_score=+0.175 drop=-0.198 (floor 0.056, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/background: plateau_score=-1.136 final_score=-1.548 drop=-0.412 (floor 0.176, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/per_cloud: plateau_score=+0.806 final_score=+0.500 drop=-0.306 (floor 0.074, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/background: plateau_score=+0.284 final_score=+1.079 drop=+0.795 (floor 0.093, beats_cluster=True) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/per_cloud: plateau_score=+0.695 final_score=+0.624 drop=-0.072 (floor 0.087, beats_cluster=True) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/background: plateau_score=-1.087 final_score=-1.484 drop=-0.397 (floor 0.114, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/per_cloud: plateau_score=+0.253 final_score=+0.125 drop=-0.128 (floor 0.055, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**

**H1 paired (per-prompt, matched token counts, raw, final layer)**: {'n': 100, 'median_diff': 0.0138, 'frac_positive': 0.75, 'wilcoxon_p': 8.94981124911862e-10, 'verdict': 'PASS(generated more tree-like)', 'model': 'Qwen/Qwen2.5-7B', 'dataset': 'prontoqa'}

## H2 v2 (direction consistency across models)

- Qwen/Qwen2.5-7B: PASS(generated more tree-like) (median_diff=0.0138, p=8.94981124911862e-10)

H2 requires >= 2 reasoning/base model PAIRS showing the same
ordering of median_diff; with a single pair report direction only,
make NO tuning-causes-it claim (the v1 design error).

