# Pre-registration v2 — the gap-closure run

Written **before** the second DGX run. This version exists because the first
protocol broke in three documented ways, so its verdict cannot be read as
confirmatory in either direction. v2 discloses the deviations, fixes the ruler,
and commits new decision rules that are properties of the data (signs, pairings,
significance), not of absolute constants on an uncalibrated scale.

## Protocol deviations in run 1 (disclosed, not litigated)

1. **The Atlas-parity STOP gate was never implemented in code.** PREREGISTER.md
   required the raw measurements to reproduce the Atlas's plateau shape before
   any Gate B verdict ("fix extraction first"). Raw medians came out 0.03–0.13
   (no plateau) at every layer, yet the Gate B pivot verdict was issued anyway.
2. **The estimator changed after seeing run-1 data** (attention-sink stripping,
   commits 80e7802/dc84bdd), so the committed thresholds were calibrated against
   a different instrument.
3. **The margins were absolute constants on an uncalibrated scale.** The
   estimator's measured flat↔tree span at the operating regime is ~0.11–0.19,
   so survive_margin=0.10 and h1_margin=0.05 demanded 50–90% of the entire
   usable range — near-unsatisfiable by construction. Additionally the flat
   anchor itself is dimension-dependent (0.19 at d=64 vs 0.11 at k=256 vs 0.03
   at d=3584), and the controls ran at d=64 / one metric each while verdicts
   were scored at k=256 under `background`.

**Consequence: run 1 is reclassified as EXPLORATORY.** Its verdict
(ANISOTROPY_ARTIFACT) is void — not because the data was wrong, but because
7/10 cells showed no raw effect to adjudicate in the first place, and the ruler
was mis-calibrated for the rest.

## What run 2 decides (in order; each stage gates the next)

### Stage 1 — Generation audit (`audit_generations`)
Did the reasoning model actually reason, and were traces truncated?
- **STOP rule:** if the DeepSeek arm shows CoT in < 50% of generations under a
  regime, that regime's H1/H2 cells are void; use the chat-template arm.
- Truncation > 25% voids 'last'-token analyses for that arm.

### Stage 2 — Atlas forensics (`atlas_forensics`)
The four-point sum-form delta_rel (Atlas Eq. 5/6) has a **hard Euclidean
ceiling of 1 − 1/√2 ≈ 0.293** (unit square extremal; verified analytically and
numerically). The Atlas's reported ~0.995 medians are unattainable under their
own printed formula, so the "reproduce 0.99 then ablate" plan is impossible by
construction. Instead:
- Confirm the ceiling numerically on our activations (no cloud may exceed 0.293).
- Sweep candidate statistics (1 − δ_rel, max-Gromov-product/diam, defect/median-
  distance, …) and report which, if any, lands in the Atlas's reported range.
- **Pre-committed conclusion rules:** if a candidate matches their plateau AND
  final-drop shape → the correction paper's thesis is "the Atlas's headline
  statistic is X, not Eq. 5/6, and X saturates by construction." If none
  matches → thesis is "the Atlas's numbers are not reconstructible from the
  within-prompt object"; either way the ceiling proof stands on its own.

### Stage 3 — Rung 0 v2 (`rung0_v2`)
- **Gate 0 (parity, now in code):** Gate B is only issued if the raw
  Atlas-object plateau ≥ 0.20 (i.e. the phenomenon to be adjudicated exists in
  our data at ≥ ~2/3 of the theoretical ceiling). Otherwise the cell verdict is
  `OBJECT/EXTRACTION MISMATCH` and no anisotropy claim is made.
- **Gate A (regime-matched):** the matched flat anchor and matched embedded-tree
  anchor must keep the span open (flat > tree) under EVERY metric at each
  cell's own (N, k). Span collapse ⇒ `CANNOT ADJUDICATE` for that cell.
- **Gate B (span-relative):** an effect is real iff
  `drop > 0.15 × span` AND `drop > 3 × noise_floor` (bootstrap-over-points).
  Label set: `NO_RAW_EFFECT` (nothing to explain) / `ANISOTROPY_ARTIFACT`
  (raw effect vanished under correction) / `CLUSTER_STRUCTURE` (survives but
  does not beat the cluster null) / `REAL_SIGNAL` (survives AND beats the
  cluster null). Only `REAL_SIGNAL` licenses the word "hierarchy".
- **H1 v2 (paired):** per-prompt paired δ_rel (own prompt-cloud vs own
  generated-cloud, matched token counts, raw coordinates), Wilcoxon signed-rank.
  PASS = p < 0.05 with positive median (generated more tree-like). No margins.
- **H2 v2:** with only one reasoning/base pair, we report DIRECTION only and
  make no tuning-causes-it claim. A causal H2 claim requires ≥ 2 pairs with a
  consistent ordering (declared future work unless a second pair is run).

### Stage 4 — Matched-conditioning probe (`matched_probe`) — the decisive one
Three arms per layer, identical parameter shapes, whitened features (fit on
train), held-out pairs, ≥ 5 seeds, rho-convergence training:
`bare_euclidean` (Raj's actual flat arm) / `cond_euclidean` (full Raj
conditioning, no curvature) / `hyperbolic` (full Raj recipe).
- **Pre-committed reading:**
  - `gap_geometry = rho(hyp) − rho(cond_euc)` > 0.05 with Wilcoxon p < 0.05 on
    ≥ 2 of the probed layers → **geometry is real** (Raj right, mechanism open).
  - `gap_conditioning = rho(cond_euc) − rho(bare_euc)` > 0.05 while
    `gap_geometry ≤ 0.02` everywhere → **conditioning explains Raj**
    (Park-consistent deflation; this is the publishable negative).
  - Anything else → report per-layer, no headline claim.

### Stage 5 — Determinants v2 (`determinants_v2`)
- Source-respecting edits on the `generated` source (the axis the Atlas never
  measured), exponential-weight pooling (powered order test), REAL nulls:
  edit-redraw null for identity/order, split-half null for meaning.
- A driver is declared only if |Δδ| beats its own null's 95th percentile; the
  order verdict is reported alongside the operator's measured power ceiling
  (an underpowered "no order effect" is flagged uninterpretable, not null).
- The last-token cluster adjudication decides H3's meaning: if the last-token
  cloud's δ does not beat the matched cluster null, H3's "low-δ locus" is
  answer-class clustering and is reported as such.

## Power / scope for run 2

- Same two models (DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B), prontoqa,
  **both chat regimes** (plain AND chat) so the regime is a measured axis, not
  an assumption. `max_new_tokens=1024`.
- ≥ 300 prompts per (model, regime) so the paired H1 Wilcoxon has ≥ 200
  usable pairs.
- Seeds {0,1,2,3,4} for every stochastic stage.
- The instrument (sink strip, estimator, thresholds below) is FROZEN at the
  commit that adds this file; any further estimator change voids the run.

```json
{
  "rung0_v2": {
    "parity_min_plateau": 0.20,
    "span_effect_frac": 0.15,
    "min_effect_over_floor": 3.0,
    "h1_alpha": 0.05,
    "gate_a_flat_frac": 0.5,
    "gate_a_tree_frac": 0.5
  },
  "matched_probe": {
    "geometry_gap_min": 0.05,
    "conditioning_gap_min": 0.05,
    "geometry_null_max": 0.02,
    "alpha": 0.05,
    "min_layers_significant": 2
  },
  "audit": {
    "min_cot_frac": 0.5,
    "max_truncated_frac": 0.25
  }
}
```
