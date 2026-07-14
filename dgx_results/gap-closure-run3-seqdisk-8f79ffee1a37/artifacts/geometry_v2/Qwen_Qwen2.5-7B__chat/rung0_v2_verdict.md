# Rung 0 v2 verdict

Thresholds source: **PREREGISTER2.md**
`{"parity_min_plateau": 0.2, "span_effect_frac": 0.15, "min_effect_over_floor": 3.0, "h1_alpha": 0.05, "gate_a_flat_frac": 0.5, "gate_a_tree_frac": 0.5}`

## Qwen/Qwen2.5-7B / prontoqa

**Gate 0 (Atlas-object raw plateau)**: 0.058 (threshold 0.2; Euclidean-formula ceiling is 0.293 — see atlas_forensics) -> **OBJECT/EXTRACTION MISMATCH**

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

- input/background: plateau_score=-1.476 final_score=-1.763 drop=-0.287 (floor 0.115, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- input/per_cloud: plateau_score=+0.099 final_score=+0.093 drop=-0.006 (floor 0.044, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/background: plateau_score=-0.853 final_score=-1.605 drop=-0.751 (floor 0.093, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/per_cloud: plateau_score=+0.446 final_score=+0.108 drop=-0.338 (floor 0.044, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/background: plateau_score=-1.533 final_score=-1.473 drop=+0.060 (floor 0.222, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/per_cloud: plateau_score=+0.806 final_score=+0.610 drop=-0.196 (floor 0.062, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/background: plateau_score=-0.342 final_score=-1.667 drop=-1.326 (floor 0.427, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/per_cloud: plateau_score=+0.919 final_score=+0.604 drop=-0.315 (floor 0.070, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/background: plateau_score=-0.804 final_score=-1.601 drop=-0.796 (floor 0.109, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/per_cloud: plateau_score=+0.287 final_score=+0.135 drop=-0.152 (floor 0.044, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**

**H1 paired (per-prompt, matched token counts, raw, final layer)**: {'n': 100, 'median_diff': -0.0013, 'frac_positive': 0.46, 'wilcoxon_p': 0.2536519936353009, 'verdict': 'NULL(no paired difference)', 'model': 'Qwen/Qwen2.5-7B', 'dataset': 'prontoqa'}

## H2 v2 (direction consistency across models)

- Qwen/Qwen2.5-7B: NULL(no paired difference) (median_diff=-0.0013, p=0.2536519936353009)

H2 requires >= 2 reasoning/base model PAIRS showing the same
ordering of median_diff; with a single pair report direction only,
make NO tuning-causes-it claim (the v1 design error).

