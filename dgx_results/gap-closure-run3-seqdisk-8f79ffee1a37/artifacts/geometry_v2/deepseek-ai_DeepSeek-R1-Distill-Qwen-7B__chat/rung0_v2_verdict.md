# Rung 0 v2 verdict

Thresholds source: **PREREGISTER2.md**
`{"parity_min_plateau": 0.2, "span_effect_frac": 0.15, "min_effect_over_floor": 3.0, "h1_alpha": 0.05, "gate_a_flat_frac": 0.5, "gate_a_tree_frac": 0.5}`

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B / prontoqa

**Gate 0 (Atlas-object raw plateau)**: 0.067 (threshold 0.2; Euclidean-formula ceiling is 0.293 — see atlas_forensics) -> **OBJECT/EXTRACTION MISMATCH**

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

- input/background: plateau_score=-1.241 final_score=-1.164 drop=+0.077 (floor 0.070, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- input/per_cloud: plateau_score=+0.167 final_score=+0.096 drop=-0.071 (floor 0.072, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/background: plateau_score=-1.134 final_score=-0.500 drop=+0.634 (floor 0.077, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/per_cloud: plateau_score=+0.263 final_score=+0.121 drop=-0.142 (floor 0.039, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/background: plateau_score=-0.719 final_score=-0.498 drop=+0.221 (floor 0.108, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/per_cloud: plateau_score=+0.531 final_score=+0.299 drop=-0.231 (floor 0.058, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/background: plateau_score=-0.349 final_score=-1.065 drop=-0.716 (floor 0.231, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/per_cloud: plateau_score=+0.832 final_score=+0.691 drop=-0.141 (floor 0.061, beats_cluster=True) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/background: plateau_score=-1.085 final_score=-0.561 drop=+0.524 (floor 0.073, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/per_cloud: plateau_score=+0.262 final_score=+0.119 drop=-0.142 (floor 0.046, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**

**H1 paired (per-prompt, matched token counts, raw, final layer)**: {'n': 100, 'median_diff': 0.0179, 'frac_positive': 0.79, 'wilcoxon_p': 1.30808185424902e-10, 'verdict': 'PASS(generated more tree-like)', 'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'dataset': 'prontoqa'}

## H2 v2 (direction consistency across models)

- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: PASS(generated more tree-like) (median_diff=0.0179, p=1.30808185424902e-10)

H2 requires >= 2 reasoning/base model PAIRS showing the same
ordering of median_diff; with a single pair report direction only,
make NO tuning-causes-it claim (the v1 design error).

