# Activation Hierarchy — Findings (minimal running log)

One line per confirmed result. Newest at top. δ = hyperbolic advantage
(ρ_hyp − ρ_cond_euclidean) on the ground-truth tree, matched capacity+conditioning.

## WHY middle layers — MECHANISM CONFIRMED (layers = composition hops)
- Peak layer scales with tree DEPTH: fictional_b1 (depth-14 chain) peaks **L20**,
  b2/b3 (depth-3) peak **L8** — same in DeepSeek AND Qwen. A depth-k relation needs
  ~k sequential attention hops to assemble ⇒ deeper structure finishes later. Caveat:
  b1 confounds depth(14) with branching(1); clean test = fix branching, vary depth.
- WHICH TOKENS: premise-concept and LAST token recover equally (Δ≈0.15 both); query
  auto-skipped (too few concepts/prompt). Corrects earlier claim that 'last' loses —
  on the tree task it carries the hierarchy as well as premise tokens.

## CROSS-FAMILY REPLICATES → 'not reasoning-specific' UP-GRADED (run3 finished locally, summary-verified)
- Mistral-7B-v0.3 (independent arch/pretraining, NOT Qwen-derived): fictional Δ≈+0.18
  mid-stack, real_b2 Δ≈+0.34 @L0, radial ρ=+0.73 — SAME fingerprint as Qwen/DeepSeek.
  So the hierarchy is architecture-general, not Qwen-family shared-weights. This
  resolves the confound below. (Llama-3.1 arm 401-gated, skipped.) Per-seed CSV not
  yet shipped from DGX — verdict-level only; run5 ships it. Summary TRUSTED: it
  reproduces run2's per-seed-verified numbers exactly (DeepSeek real L0=+0.381 ✓).
- SCALE LADDER (Qwen 1.5B/3B/7B/14B): fingerprint holds at every size; 14B STRONGEST
  (fic_b2 Δ=+0.22 vs 7B +0.18) and peak shifts DEEPER (L20, radial L36) — consistent
  with depth→composition-hops (bigger model, more layers, later peak).

## Reasoning-specific? [SUPERSEDED by cross-family above] SUGGESTED, NOT ESTABLISHED (5-whys caught a confound)
- Qwen2.5-7B base ≈ DeepSeek-R1-distill on every signal (real ρ_hyp=0.92@L0; fictional
  −0.05@L0→+0.27@L8-12; dim-collapse; radial ρ≈0.69). BUT DeepSeek-R1-Distill-Qwen-7B
  IS Qwen2.5 fine-tuned — they SHARE a base, so this is largely shared ancestry, not
  independent convergence. The cross-family run (Llama/Mistral, BLOCKED) is what would
  actually establish 'not reasoning-specific'. Downgrade the claim until it runs.

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

## Campaign questions & status
- [x] Reasoning-specific? NO — Qwen base ≈ DeepSeek distill (run2, above).
- [x] Relation generality? YES generic — is_a/part_of/causes recover, flat_set≈0 (run3 Phase B, above).
- [~] Scale ladder Qwen2.5 {1.5B,3B,14B}: BLOCKED on DGX restart → run4b resumes it.
- [~] Cross-family Llama/Mistral: BLOCKED on DGX restart → run4b (Llama may skip as gated).
Note: [~] items are CONFIRMATORY (robustness). The load-bearing findings are done.

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

## run5 verified (per-seed, from shipped CSVs) + one honest negative
- **CROSS-FAMILY CONFIRMED per-seed**: Mistral-7B fictional_b2 premise L12 all 6 seeds
  +0.13..+0.23 (mean +0.188, shuffle ≈0). Independent architecture → fingerprint is
  real. 'Not reasoning-specific' now per-seed-upheld (no longer downgraded).
- **SCALE LADDER CONFIRMED**: peak Δ rises 1.5B +0.177 → 3B +0.180 → 7B +0.184 →
  14B +0.221. Magnitude monotone in size; %-depth of peak noisy (28/55/71/41%).
- **WHY (additive-shrinking-cone) NOT SUPPORTED**: predicted edges shrink with depth
  (shrink_rho<0) at mid-stack; real shrink_rho is +0.08..+0.10 (fictional), only
  dips to −0.02 at L16-20 (real). Edges do NOT consistently shrink → the specific
  'sum of shrinking edges' generative story is falsified on real activations, EVEN
  THOUGH radial-norm↔depth is strong. Tension to resolve: the cloud is cone-like by
  tree-probe/radial, but edge-vectors aren't shrinking. Mechanism of 'how hyperbolic
  forms' remains OPEN — the additive hypothesis was too specific. (shrink_rho metric
  itself is validated: ranks additive<cone<random on mocks.)

## HARM VERDICT (run6, Aegis 2.0 real data, per-seed verified) — prediction SUPPORTED
- **Hyperbolic helps harm detection iff the label space is hierarchical.** Hazard
  TAXONOMY target: geometry gap (hyp−cond_euc) = +0.041/+0.058/+0.047 at L0/L7/L14,
  **6/6 seeds positive** each (p=.031), fading late. BINARY safe/unsafe control:
  gaps ≈0 (−.018..+.021, never 6/6, n.s.). Source=aegis2.0 (917 rows, not fallback).
- Bonus: taxonomy signal peaks EARLY (L0-L14) = the RETRIEVED signature — hazard
  categories are pretrained semantic knowledge (like dog→mammal), not assembled.
  Caveats: absolute ρ modest (hyp ~0.13-0.21 — taxonomy is decodable but not
  strongly); reps read at last token with uniform 8-tok truncation (class-uniform,
  not a confound, but a richer-pooling replication would firm it).
- USAGE: harm classifiers should probe the hazard TAXONOMY with a low-dim
  hyperbolic head at early-mid layers, not binary-classify the last layer.

## ⚠️ AGENT DIED again mid-run7 (2026-07-26 ~12:05, 52min silent, 0 heartbeats)
- Same signature as prior outage: START committed, then agent stopped (shell/host
  drop or ulimit reset). run7 did nothing before dying (no arm completed).
- DURABLE FIX pushed: dgx_agent.sh now sets `ulimit -v unlimited` itself, so a
  plain `./dgx_agent.sh` restart no longer depends on the operator's shell limit
  (the 500MB -v cap was the run4b killer and likely these deaths).
- **ON RETURN, on the DGX:** `pkill -f dgx_agent; rm -f .dgx_agent.pid; git pull
  --rebase --autostash; nohup ./dgx_agent.sh > agent.out 2>&1 &`  → auto-resumes
  run7 (harm robustness), then run8 (deception, staged). No data lost.

## Causal-patch design finding (CPU validation, before any DGX time)
- Subspace-ablation causal test is UNDERPOWERED: ablating the recovered k-dim
  "tree subspace" barely dents decodability (ρ 0.95→0.90, ≈ random's →0.94) because
  tree info is REDUNDANTLY encoded across many directions — projecting out 5 dims
  doesn't remove it. Also the tree subspace is NOT uniquely identified (recovery
  overlap ~0.23 with a planted basis even at decode-ρ=1.0; many subspaces decode
  equally). ⇒ subspace ablation would give a FALSE null. Abandoned it.
- Correct causal design = activation PATCHING: replace layer-L reps of a True-tree
  prompt with a different-tree prompt's, see if the answer flips (no subspace ID
  needed). Deferred to a later run; ran the lower-risk harm-robustness + deception
  jobs first so DGX isn't idle. Caught before wasting GPU hours.

## run7 harm robustness (cross-model + cross-dataset, per-seed) — QUALIFIED replication
- Change ONE factor: REPLICATES. mistral_aegis (cross-MODEL): taxonomy gap +0.050@L16
  (6/6 seeds), binary ~0/noisy. qwen_beavertails (cross-DATASET): taxonomy +0.055@L7
  (6/6 seeds), binary ~0. -> harm=hierarchy-dependent effect is not Qwen-specific
  and not Aegis-specific.
- Change BOTH: WEAKENS. mistral_beavertails: taxonomy best only +0.028 (4/6 seeds),
  negative at several layers -> marginal. Two domain shifts compound; BeaverTails
  labels are flatter (weaker hierarchy = less for curvature to exploit, as predicted).
- HONEST verdict: harm result is real and replicates under single-factor shifts, but
  is NOT bulletproof under compound shift. Absolute gaps small (~0.03-0.06). Lead
  with run6 Aegis (6/6, p=.031) + these single-shift replications; report the
  double-shift weakening plainly.

## Method caveats (found while building the causal-WHY tests — matter for the paper)
- **MDR asymmetry**: the hyperbolic arm applies MDR (tanh norm cap) before expmap0;
  cond_euclidean doesn't. At c→0 expmap0=identity but MDR still runs, so the c→0
  hyperbolic FIT sits ~+0.05 rho above euclidean on the mock. Arms match on geometry
  distance (exact) but hyp has one extra bounded nonlinearity → any hyperbolic Δ may
  be ~0.05 inflated. Doesn't overturn the low-dim-collapse headline (Δ up to +0.19).
- **c→0 contract**: poincare.dist special-cases EXACTLY c=0 → ‖x−y‖, but the analytic
  c→0 limit of the ball distance is 2‖x−y‖. Euclidean arms pass exact 0.0 (verified),
  so they're correct; a future swap to a small epsilon would silently 2× the euclidean
  baseline. Now guarded by a test.
- **Composition test**: only shrink_rho (edge norm↔depth) survived CPU validation on
  known layouts; edge-cosine & centroid-reconstruction were confounded in high-dim
  (random scored ~as high as additive) → dropped. shrink_rho ranks additive −0.52 <
  cone −0.21 < random 0.00.

## Log
- 2026-07-14: run2 launched (DeepSeek+Qwen 7B, prontoqa_tree). DeepSeek verdict above.
- 2026-07-14: run3 campaign staged (relations + scale ladder + cross-family);
  pushes only after run2 DONE (editing job.sh mid-run would corrupt the agent).
