# WHEN verdict (H1/H2/H3 — the Raj <-> Atlas bridge)

Thresholds source: **PREREGISTER.md**  
`{"h1_margin": 0.05, "h1_min_effect_over_boot": 2.0, "h2_gap_margin": 0.05, "max_bootstrap_std_final": 0.02}`

Measured under the `background` metric at the final layer.

## H1 — does generation amplify the final-layer compression?
(PASS = generated is more tree-like than prompt, beyond margin and noise)

- **Qwen/Qwen2.5-7B**: δ(prompt)=0.2405 vs δ(generated)=0.23 → diff=0.0105 (floor 0.0042, target 0.02) → **AMBIGUOUS(powered)**
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B**: δ(prompt)=0.2264 vs δ(generated)=0.1933 → diff=0.0331 (floor 0.0048, target 0.02) → **AMBIGUOUS(powered)**

## H2 — is the prompt→generated gap larger in reasoning-tuned models?
(the mechanism Raj never gave for DeepSeek-vs-Qwen)

- gap(reasoning)=0.0331 vs gap(base)=0.0105 → margin=0.0226 → **FAIL/AMBIGUOUS**

## H3 — is the thinking-token source the lowest-δ locus? (secondary)

- **Qwen/Qwen2.5-7B**: lowest-δ source = `last` (thinking lowest: False) — {'input': 0.2405, 'generated': 0.23, 'thinking': 0.225, 'last': 0.0949, 'all': 0.2376}
- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B**: lowest-δ source = `last` (thinking lowest: False) — {'input': 0.2264, 'generated': 0.1933, 'thinking': 0.1522, 'last': 0.0818, 'all': 0.2113}

## Summary
- **H1 does not pass and the run IS powered → genuinely drop the 'generation amplifies' framing.** Raj's effect is not about generation.
