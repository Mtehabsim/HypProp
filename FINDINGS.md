# Activation Hierarchy — Findings (minimal running log)

One line per confirmed result. Newest at top. δ = hyperbolic advantage
(ρ_hyp − ρ_cond_euclidean) on the ground-truth tree, matched capacity+conditioning.

## Reasoning-specific? NO (run2 complete: DeepSeek-distill vs Qwen-base, verified per-seed)
- Qwen2.5-7B **base** reproduces DeepSeek-R1-distill almost exactly: real ρ_hyp=0.92
  from L0; fictional −0.05@L0 → +0.27@L8-12; dim-collapse (peak m5, negative by m16);
  radial ρ≈0.69. ⇒ the hierarchy is a property of **representation, not reasoning**
  (as v2 found for conditioning). Both fail the strict branching dose-response.

## Headline (run2, DeepSeek-R1-Distill-Qwen-7B; confirmed on Qwen-7B base)
- **WHERE**: hyperbolic beats matched Euclidean at **low dim (m=3–5)** on concept
  tokens (premise/query), **collapses to ~0 by m=16** → curvature substitutes for
  dimension (the real "why hyperbolic"). Strongest **mid-stack L8–12**.
- **WHAT/WHY-1 (two mechanisms)**: real taxonomy (dog→mammal) is **retrieved** —
  tree-structured from **layer 0** (ρ_hyp=0.92 @ m3). Novel/fictional is-a trees
  are **assembled in-context** — ~0 at L0, rise to +0.30 by L8.
- **Radial code has a lifecycle**: ρ(‖rep‖, node-depth) peaks ~**0.65 at L12**,
  decays to ~0 by L28 (repurposed for next-token). ⇒ read-out head belongs at L8–12.
- **CAVEAT**: strict branching dose-response did NOT cleanly pass (b1 path not
  quite 0 at m≥5; b2≈b3). Lead with the dimension-collapse fingerprint instead.

## ⚠️ CAMPAIGN STALLED — needs a 1-line DGX action on return
- run3 hung on a Qwen2.5-1.5B tree_probe cell (fictional_b1 last L16); then the
  DGX AGENT itself stopped (last heartbeat 210m @ 18:59 EDT, none since — process
  killed / session dropped / network). Origin frozen; can't resolve via git.
- **ON RETURN, on the DGX:** `pkill -f tree_probe; pkill -f hidden_state_extractor`
  then restart the agent: `./dgx_agent.sh` (or `nohup ./dgx_agent.sh &`). It will
  pull, see hierarchy-campaign-run4b (current job.sh), and launch it. run4b is
  timeout-hardened (no cell can wedge it) + resumable (imports run3's shipped
  verdicts, skips Phase B) + scale ladder reordered 14B→3B→1.5B so the flaky 1.5B
  rung runs LAST and can't block the informative rungs. If a stale
  `.dgx_agent.pid` blocks start, `rm .dgx_agent.pid` first.
- **RESULTS ARE SAFE regardless:** run2 (reasoning-specificity) + Phase B relation
  generality + negative control are shipped, committed, and recorded below. Only
  the confirmatory scale-ladder + cross-family remain.

## Campaign (run3, 18h) — questions & status
- [ ] Phase B relation types: is δ is-a-specific or generic? **flat_set must give δ≈0** (neg control).
- [ ] Phase A scale ladder Qwen2.5 {1.5B,3B,7B,14B}: does δ sharpen / shift layer with scale?
- [ ] Phase C cross-family Llama-3.1-8B, Mistral-7B: does L8–12 / low-dim fingerprint replicate?
- [ ] Reasoning-specific? DeepSeek(distill) vs Qwen(base) — from run2 once Qwen ships.

## Pre-flight validations (CPU mock, before spending DGX time)
- **Negative control PASSES (relation_trees mock)**: same prompt shape, vary the
  relation. is_a Δ=+0.25 (ρ_hyp .64), part_of +0.15 (.63), causes +0.19 (.62),
  **flat_set (star, no hierarchy) Δ=+0.05, ρ_hyp=0.03** → rig recovers real
  relational trees and correctly finds ~nothing in a flat set. Trustworthy.

## Offline deep-dive (run2 CSVs, no DGX) — WHY sharpened, two distinct layer-signatures
- **Assembled (fictional)**: emergence curve rises from ρ=−0.05@L0, crosses 50%-peak
  by L4, PEAKS L8(DeepSeek)/L16(Qwen), decays to ~67% by L28 → tree BUILT in first
  third, held mid-stack, overwritten late. **Retrieved (real)**: flat-high all layers
  (0.92→0.85, spread .07) → present at L0, not built. Same in both models (onset L4
  identical) ⇒ representation-not-reasoning, reinforced. Read-out head → L8-16.

- **Two INDEPENDENT signals co-localize (run2 CSVs)**: radial-norm↔depth (training-free)
  peaks mid-stack (DeepSeek L12 ρ=.64, Qwen L20 ρ=.69) at the SAME band where tree-
  decodability peaks (L8-16), both decaying by L28. Convergence of a fit-based and a
  fit-free measure ⇒ hierarchy is a real property of the representation, not a probe
  artifact. (Aside: radial dips at L4 while decodability already rises — depth-order
  and full-tree-geometry consolidate at slightly different rates.)

## Campaign results (run3)
- **Phase B — relation generality + neg control (Qwen2.5-7B, real activations, per-seed verified):**
  hierarchy is GENERAL to structured relations, not is-a-specific. is_a Δ=+0.17,
  part_of +0.16, causes +0.15 (all peak ~L8, ρ_hyp≈0.22, shuffle≈0); **flat_set
  (star) Δ=+0.003, ρ_hyp=−0.007** — same prompt shape, no relation → no hierarchy.
  Clean discriminator on real data: the rig measures structure, not tokens.

## Log
- 2026-07-14: run2 launched (DeepSeek+Qwen 7B, prontoqa_tree). DeepSeek verdict above.
- 2026-07-14: run3 campaign staged (relations + scale ladder + cross-family);
  pushes only after run2 DONE (editing job.sh mid-run would corrupt the agent).
