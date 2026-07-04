# Pre-registration — decisive Rung-0 + WHEN run

Written **before** the DGX run, on purpose: the whole point is that an ambiguous
number must not become a Rorschach test that triggers another replan. The
pass/fail/ambiguous rules below are committed in advance. `rung0.py` reads the
`rung0` thresholds from the JSON block at the bottom of this file.

## What this run decides

One small run (two models: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and
`Qwen/Qwen2.5-7B`; one dataset; a few hundred prompts; **generation on**) answers
two questions and nothing else:

1. **Rung 0 — is the tree-likeness real or anisotropy?** (gates the whole thesis)
2. **WHEN / H1–H2 — does generation amplify the final-layer compression, more so
   in the reasoning model?** (the Raj↔Atlas bridge)

Everything downstream (probe, checkpoint-contrast, causal patching, safety) stays
**unbuilt** until this run returns.

## Rung 0 — pre-registered thresholds & verdict rule

- **Gate A (calibration must behave):**
  - `tree_control` δ_rel ≤ **0.15** under *every* metric (raw/pca_only/per_cloud/
    background). Else whitening is broken → "vanished" is uninterpretable → STOP.
  - `gaussian_control` δ_rel ≥ **0.15** under every metric. Else whitening is
    *manufacturing* tree-likeness → STOP.
- **Gate B (does data survive `background` whitening?):** per (model, token_source),
  let `drop = δ_mid − δ_final`.
  - **REAL_HIERARCHY** iff `drop > 0.10` **and** `drop > 2.0 × noise_floor`
    (noise_floor = max(std_rel, bootstrap_std)).
  - **ANISOTROPY_ARTIFACT** otherwise (esp. if the large *raw* drop collapses
    under `background`).
- **Overall:** all cells REAL → proceed; all ANISOTROPY → pivot to the
  deflationary paper (Atlas correction); mixed → proceed only on REAL cells.

## WHEN / H1–H2 — pre-registered predictions (decide BEFORE looking)

Effect sizes here are unknown, so we commit thresholds now. Let
`gap(model) = δ_final(prompt) − δ_final(generated)` measured under `background`.

- **H1 (generation amplifies compression):** PASS iff, within a model,
  `δ_final(generated) < δ_final(prompt)` by **> 0.05 and > 2 × noise_floor**.
  AMBIGUOUS if the difference is positive but within that band. FAIL if ≤ 0.
- **H2 (explains Raj's model-specificity):** PASS iff
  `gap(DeepSeek-R1-Distill) − gap(Qwen2.5-7B) > 0.05` beyond the pooled noise
  floor. FAIL/AMBIGUOUS otherwise → drop the "generation amplifies" framing.
- **H3 (locus, secondary):** thinking-marker tokens are the lowest-δ source at
  the final layer. Reported, not gating.

## Atlas parity check (day-one pipeline validation)

Before trusting any whitened verdict: under `metric=raw`, our extraction must
reproduce the Atlas's qualitative shape — a high mid-layer plateau (δ_rel ≈ 0.9+)
dropping to ≈ 0.4–0.7 at the final layer. If our raw numbers don't show that
shape, the problem is **extraction**, not geometry — fix that first.

## Power / scope

- Enough prompts that `bootstrap_std` is small vs the thresholds above (target
  bootstrap_std ≲ 0.02 at the final layer; increase sample count if not).
- Two models only (the minimal pair H2 needs). One dataset. Generation on.
- Mock-first is for harness validation only (tree low, gaussian high) — never
  evidence.

```json
{
  "rung0": {
    "tree_control_max": 0.15,
    "gaussian_control_min": 0.15,
    "survive_margin": 0.10,
    "min_effect_over_boot": 2.0
  },
  "when": {
    "h1_margin": 0.05,
    "h1_min_effect_over_boot": 2.0,
    "h2_gap_margin": 0.05,
    "max_bootstrap_std_final": 0.02
  }
}
```
