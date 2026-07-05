# Rung 0 verdict

Thresholds source: **PREREGISTER.md**  
`{"tree_control_max": 0.15, "gaussian_control_min": 0.15, "survive_margin": 0.1, "min_effect_over_boot": 2.0}`

## Gate A -- calibration (is whitening trustworthy?)
- tree_control stays low (<= 0.15) under every metric: **PASS**
- gaussian_control stays high (>= 0.15): **PASS**

## Gate B -- does the data's tree-likeness survive `background` whitening?

- **Qwen/Qwen2.5-7B / all**: raw drop=-0.013, background drop=-0.015 (floor 0.006) -> **ANISOTROPY_ARTIFACT**
- **Qwen/Qwen2.5-7B / generated**: raw drop=0.005, background drop=0.006 (floor 0.008) -> **ANISOTROPY_ARTIFACT**
- **Qwen/Qwen2.5-7B / input**: raw drop=-0.033, background drop=-0.031 (floor 0.008) -> **ANISOTROPY_ARTIFACT**
- **Qwen/Qwen2.5-7B / last**: raw drop=-0.104, background drop=0.074 (floor 0.008) -> **ANISOTROPY_ARTIFACT**
- **Qwen/Qwen2.5-7B / thinking**: raw drop=-0.058, background drop=-0.054 (floor 0.008) -> **ANISOTROPY_ARTIFACT**
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B / all**: raw drop=0.036, background drop=0.025 (floor 0.006) -> **ANISOTROPY_ARTIFACT**
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B / generated**: raw drop=0.042, background drop=0.038 (floor 0.007) -> **ANISOTROPY_ARTIFACT**
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B / input**: raw drop=-0.007, background drop=-0.02 (floor 0.016) -> **ANISOTROPY_ARTIFACT**
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B / last**: raw drop=-0.062, background drop=0.06 (floor 0.003) -> **ANISOTROPY_ARTIFACT**
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B / thinking**: raw drop=0.032, background drop=0.044 (floor 0.004) -> **ANISOTROPY_ARTIFACT**

## VERDICT: **ANISOTROPY_ARTIFACT -> PIVOT to the deflationary paper (Park wins; publish as an Atlas correction). Do NOT build a probe expecting a win.**
