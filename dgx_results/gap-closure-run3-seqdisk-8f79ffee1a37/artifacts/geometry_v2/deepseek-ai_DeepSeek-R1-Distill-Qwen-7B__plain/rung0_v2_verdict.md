# Rung 0 v2 verdict

Thresholds source: **PREREGISTER2.md**
`{"parity_min_plateau": 0.2, "span_effect_frac": 0.15, "min_effect_over_floor": 3.0, "h1_alpha": 0.05, "gate_a_flat_frac": 0.5, "gate_a_tree_frac": 0.5}`

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B / prontoqa

**Gate 0 (Atlas-object raw plateau)**: 0.023 (threshold 0.2; Euclidean-formula ceiling is 0.293 — see atlas_forensics) -> **OBJECT/EXTRACTION MISMATCH**

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

- input/background: plateau_score=-1.335 final_score=-1.815 drop=-0.480 (floor 0.096, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- input/per_cloud: plateau_score=+0.204 final_score=+0.082 drop=-0.121 (floor 0.057, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/background: plateau_score=+0.114 final_score=+0.029 drop=-0.085 (floor 0.442, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- generated/per_cloud: plateau_score=+0.439 final_score=+0.349 drop=-0.090 (floor 0.110, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/background: plateau_score=-0.206 final_score=-0.792 drop=-0.586 (floor 0.188, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- thinking/per_cloud: plateau_score=+0.548 final_score=+0.413 drop=-0.134 (floor 0.066, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/background: plateau_score=-0.194 final_score=+0.774 drop=+0.969 (floor 0.297, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- last/per_cloud: plateau_score=+1.095 final_score=+1.017 drop=-0.078 (floor 0.056, beats_cluster=True) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/background: plateau_score=+0.156 final_score=-0.013 drop=-0.168 (floor 0.477, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**
- all/per_cloud: plateau_score=+0.447 final_score=+0.351 drop=-0.096 (floor 0.117, beats_cluster=False) -> **NO_RAW_EFFECT(nothing to adjudicate)**

**H1 paired (per-prompt, matched token counts, raw, final layer)**: {'n': 100, 'median_diff': 0.0177, 'frac_positive': 0.81, 'wilcoxon_p': 2.3129702605871595e-12, 'verdict': 'PASS(generated more tree-like)', 'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'dataset': 'prontoqa'}

## H2 v2 (direction consistency across models)

- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B: PASS(generated more tree-like) (median_diff=0.0177, p=2.3129702605871595e-12)

H2 requires >= 2 reasoning/base model PAIRS showing the same
ordering of median_diff; with a single pair report direction only,
make NO tuning-causes-it claim (the v1 design error).

