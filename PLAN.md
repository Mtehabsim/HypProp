# Project Plan: What Makes an LLM's Thoughts Hyperbolic (and Using It for Safety)

## What this project is

We study the *shape* of a language model's internal activations while it reasons.

Some data has a **tree shape** (a hierarchy: Harm → Hate Speech → Religious hate). Trees fit much
better in **hyperbolic space** (a curved space that has room for branches) than in normal flat
(**Euclidean**) space. A recent workshop paper (Raj 2026) showed that a hyperbolic "probe" reads a
reasoning model's hidden states more reliably than a flat one — but only at the last layer, and
only because that layer gets squeezed/compressed, not because the data was really a tree.

We want to go further and answer *why* the activations become hyperbolic, then turn that knowledge
into a better, harder-to-fool safety probe.

The project has **three parts**:

1. **Science (main goal):** Figure out what actually makes a model's "thoughts" hyperbolic. Is it
   the **specific tokens** (like "wait", "therefore"), the **order** of the tokens, or the
   **meaning** of the text?
2. **Method (the payoff):** Build a probe that automatically uses hyperbolic geometry when it
   helps and flat geometry when it doesn't. The science from part 1 decides how this switch works.
3. **Safety use-case (core result):** Use the probe to detect harmful content, and test whether it
   makes an attacker's job **harder** (raises their cost) compared to a normal flat probe.

## The questions we are trying to answer

1. Do a model's reasoning activations really take a tree/hierarchy shape? At which **layer**, with
   how many **tokens**, and from which tokens (the **input**, the **generated "thinking" tokens**,
   or **all** of them)?
2. **What causes** that hyperbolic shape — the tokens themselves, their order, or their meaning?
3. Can we build a probe that uses this to be **more efficient** (fewer dimensions, fewer training
   examples, and still works where flat probes break)?
4. For safety: does a hyperbolic + generation-token + multi-layer probe make it **harder / more
   costly** for an attacker to hide harmful activations?

## The phases

- **Phase 0 – Setup:** Pick data and models, build a clean tool to pull out hidden states.
- **Phase 1 – Map + Science:** Measure the shape (hyperbolicity) across layers and token choices,
  then run experiments to find *what causes* it. **This is the main contribution.**
- **Phase 2 – Build the probe:** Build the adaptive probe. Its design depends on Phase 1 results.
- **Phase 3 – Safety test:** Attack the probe (obfuscation) and measure how much harder hyperbolic
  makes the attack.

---

## Background we already checked (so we don't repeat or miss anything)

We read the key papers directly and verified which are peer-reviewed. Short version:

- **Raj 2026** (the paper this builds on) is the user's own **workshop paper, not peer-reviewed**.
  Verified facts: on DeepSeek-R1 the flat probe collapses at the last layer (score 0.488) while the
  hyperbolic one stays strong (0.967); this does **not** happen in Qwen2.5. Its own key line: the
  target it used was a simple 1-D chain that fits *both* geometries equally, so its hyperbolic win
  is really about **surviving late-layer compression**, not about hierarchy. It never tested
  input-only tokens, and it lists "branching hierarchies" as future work — that gap is our opening.
- **HyperGuide** (arXiv:2605.24140) — a reasoning paper (Game-of-24, etc.), **not** safety. It uses
  the same kind of machinery (hyperbolic head on frozen LLM states) so we cite it as a method
  cousin, but it does not compete with our safety idea.
- **H-Probes** (arXiv:2605.00847) — note: "H" means **Hierarchical, not hyperbolic**. It uses flat
  probes on branching trees and proves the hierarchy lives in a **low-dimensional** subspace and is
  **causally important**. We reuse its low-rank idea and its causal-test method.
- **Obfuscated Activations** (Bailey et al., arXiv:2412.09565) — the reason our safety angle
  matters. It shows attackers can hide harmful activations and drop a probe's catch-rate from 100%
  to 0% while still jailbreaking 90% of the time. **But** it also verified three useful facts:
  probes that read **generated** tokens are harder to bypass than input-only ones, **multiple
  layers** help, and hiding has a **real cost** to the attacker. Important warning: attacks
  **transfer** between probe types, so simply switching to hyperbolic is *not* a secret defense —
  the value has to come from those real levers, not from the attacker "not knowing."

**Closest prior work to cite:** Chen et al. 2021 (hyperbolic probe on BERT, ICLR), Ganea et al.
2018 (defines the hyperbolic classifier, NeurIPS), Park et al. 2025 (the main counter-view: LLM
hierarchy is flat *after whitening*, ICLR oral), Yang et al. 2025 (HypLoRA, NeurIPS), Zhao et al.
2025 (closest harm-probing baseline).

---

## Quick note: yes, we use logistic regression

A linear probe *is* logistic regression, and we use it in both arms:
- **Flat probe = logistic regression** (`sklearn` softmax on `Wx+b`) — the baseline.
- **Hyperbolic probe = H-MLR** = logistic regression rewritten for curved space (the flat
  hyperplanes become curved "gyroplanes", distance is geodesic). When the curvature is set to 0 it
  becomes ordinary logistic regression — that is what makes the comparison fair.
- One exception: to *measure the tree shape* in Phase 1 we also use a **structural probe**, which is
  a distance regression (not logistic regression) — a different tool for a different job (measuring
  geometry, not classifying harm).

## The rules we follow for every comparison (so results are trustworthy)

These come from the literature review and are non-negotiable:

- **Always whiten the features first.** Flat probes look bad on raw activations only because the
  space is skewed (anisotropic). Whitening is the fair baseline. If a hyperbolic win disappears
  after whitening, it was compression, not hierarchy.
- **Report the normalized shape score `δ_rel` (0 = perfect tree, ~1 = flat), never the raw number.**
  Raw scores shrink just because a layer is compressed, which would fool us into calling a squeezed
  layer "hyperbolic."
- **Match everything between the two probes:** same dimensions, same number of parameters, same
  regularization. Add a "flat probe on the same transformed features" arm to separate the curve
  from the transform.
- **Run ≥5 seeds and a significance test.** Judge by **selectivity** and **MDL** (sample
  efficiency), not just accuracy.
- **Numbers:** store activations in **fp32** (not bf16 — it breaks near the edge of the ball), clip
  feature norms, and watch for the "log(exp) cancels out" bug that silently turns hyperbolic back
  into flat.
- **Extraction:** get hidden states straight from `generate(...)`, keep token positions aligned
  (fix the known off-by-one bugs), match multi-token words correctly per tokenizer, and **do not
  force** the model to say "wait/therefore" — that biases the data.

---

## Phase 0 — Setup

- **Data:**
  - Main harm labels: **MLCommons AILuminate** + **Aegis/Nemotron** (real branching harm
    taxonomies).
  - Harmful prompts to trigger reasoning: **HarmBench / AdvBench**.
  - Controls: **WOS** (a real 2-level hierarchy), a **WordNet** set (hyperbolic *should* win here —
    proves the pipeline works), and a **flat/binary** set (hyperbolic should *not* win).
  - Dropped: **GoEmotions** (not a real tree) and **AdvGLUE** (not harm data).
- **Models:** DeepSeek-R1-Distill-Qwen-7B (reasoning), Qwen2.5-7B-Instruct (same base, clean
  control), Llama-Guard-3-8B (safety specialist).
- **Tool:** a clean hidden-state extractor (fp32, correct token alignment, no forced tokens).

## Phase 1 — Map the shape + find what causes it (MAIN CONTRIBUTION)

- **Map:** sweep across **layers × token source** {input-only, thinking-tokens, last-token,
  all-tokens}. For each, measure `δ_rel` (on whitened states) and whether the harm taxonomy embeds
  as a tree. (Input-only is new — Raj never tested it.)
- **Find the cause (the novel part):** run controlled edits on the same activations —
  - swap thinking-markers for random words → tests **token identity**,
  - shuffle token order → tests **order/position**,
  - paraphrase / replace content words but keep structure → tests **meaning**.
  See which edit changes the hyperbolicity. That tells us *why* the thoughts are hyperbolic.
- **Pick the best setting** by **both** low `δ_rel` **and** good taxonomy fit. Freeze this choice on
  a separate split so we don't fool ourselves; evaluate later phases on held-out data.

## Phase 2 — Build the adaptive probe (design depends on Phase 1)

- Core probe: hyperbolic classifier (H-MLR) via **geoopt**, with clipping and MDR for stability.
- **Adaptive gate:** measure the shape, then pick geometry/curvature (curve → 0 falls back to flat)
  to maximize the user's goal. **What the gate keys on comes from Phase 1:** if identity-driven, it
  keys on the tokens; if order-driven, on position; if meaning-driven, on the semantics. We do not
  finalize this until Phase 1 is done.
- Compare against fair baselines (whitened flat probe, flat-on-same-transform, curve→0).
- **Efficiency measured four ways:** works where flat breaks, fewer training samples (MDL), fewer
  dimensions, and (Phase 3) higher attacker cost.

## Phase 3 — Safety test: does it cost the attacker more? (CORE RESULT)

- Run an obfuscation attack (Bailey-style) against a flat probe vs our probe. Measure the
  **attacker's budget** (effort to drop catch-rate to 0 while keeping the jailbreak working) and
  the **damage to the model's own performance** the attack causes.
- Test **transfer** (does an attack tuned on the flat probe also beat the hyperbolic one?) and the
  proven levers (generation tokens, multiple layers).
- **Honest claim:** "raises the attacker's cost / forces more damage," **not** "unbreakable."

---

## Files we will build — and exactly what each one saves for us to check

Everything writes to a `results/` folder with a fixed structure so we can inspect and compare runs.
Each script logs to `results/logs/<script>.log` and saves machine-readable output (`.json`/`.csv`/
`.pt`) plus a plot where useful.

- `extract/hidden_state_extractor.py`
  - **Saves:** `results/activations/<model>/<dataset>/<sample_id>.pt` — per-layer hidden states
    (fp32) + token metadata (token string, position, layer, is-thinking-token flag, model id) +
    the generated text and label.
  - **Logs:** how many samples processed, tokens per sample, how many thinking-tokens matched,
    any samples skipped (and why).
- `geometry/delta_hyperbolicity.py`
  - **Saves:** `results/geometry/delta_rel.csv` — one row per (model, layer, token-source) with
    `δ_rel`, its variance, diameter, and the tree/sphere sanity values. Plot:
    `results/geometry/delta_by_layer.png`.
  - **Logs:** the most-hyperbolic settings, and any that fail the sanity check.
- `geometry/determinants.py` *(main science module)*
  - **Saves:** `results/determinants/attribution.csv` — for each edit (token-swap / order-shuffle /
    paraphrase), the change in `δ_rel` (Δδ), so we can read off which driver (identity / order /
    meaning) matters most. Plot: `results/determinants/driver_effects.png`.
  - **Logs:** a plain-English summary line, e.g. "shuffling order changed δ_rel by X; swapping
    markers by Y; paraphrase by Z → main driver = …".
- `geometry/structural_probe.py`
  - **Saves:** `results/geometry/structural_probe.csv` — flat vs hyperbolic distance-fit scores
    (Spearman ρ + distortion) per layer. This is the module that **reproduces Raj's L27 result** as
    a sanity check.
  - **Logs:** the reproduced numbers vs Raj's (target: flat 0.488 / hyperbolic 0.967 at DeepSeek L27).
- `probes/hmlr.py` — hyperbolic logistic regression (geoopt, clipping, MDR).
  - **Saves:** trained probe to `results/probes/hmlr_<setting>.pt` + `metrics.json` (accuracy,
    macro-F1, per-class scores, learned curvature).
- `probes/baselines.py` — flat logistic regression, flat-on-same-transform, curve→0.
  - **Saves:** each baseline to `results/probes/<name>_<setting>.pt` + matching `metrics.json`.
- `probes/adaptive_gate.py` — the shape-driven geometry switch (design set by Phase 1).
  - **Saves:** `results/probes/adaptive/gate_decisions.csv` — for each input, the measured `δ_rel`,
    the geometry it chose, and why. Plus the trained probe + metrics.
- `eval/compare.py`
  - **Saves:** `results/eval/comparison.csv` — every probe × setting × seed with accuracy,
    **selectivity**, **MDL/codelength**, and dimension used; plus `results/eval/summary.md` (a
    human-readable table) and `results/eval/significance.json` (paired-test p-values). Plots:
    `dimension_efficiency.png`, `mdl_curve.png`.
  - **Logs:** the headline "hyperbolic vs flat" verdict per setting, with whether it survived
    whitening.
- `security/obfuscation_attack.py`
  - **Saves:** `results/security/attack.csv` — attacker **budget** (steps / perturbation size to
    drop recall to ~0 at fixed jailbreak rate), the **behavior-damage** cost, and **transfer**
    results (attack-on-flat vs attack-on-hyperbolic, both directions). Plot:
    `results/security/budget_comparison.png`.
  - **Logs:** the honest one-line claim, e.g. "hyperbolic raised attacker budget by N× / forced M%
    more behavior damage".
- `data/` — loaders for AILuminate, Aegis, HarmBench/AdvBench, WOS, WordNet control.
  - **Saves:** cached, label-mapped datasets to `results/data_cache/`.

## How we check the work

- **Basic checks:** `δ_rel` ≈ 0 on a made-up tree, high on random points; math round-trips
  correctly; edits keep labels intact.
- **Prove the pipeline:** hyperbolic beats whitened-flat on the WordNet control *before* touching
  safety data.
- **Reproduce Raj:** get the DeepSeek L27 pattern (flat 0.488 vs hyperbolic 0.967) to confirm our
  tools are right.
- **Stop early if:** no setting is truly more hyperbolic than the whitened control, or whitened-flat
  ties hyperbolic on the taxonomy, or the gain vanishes after whitening (means it was compression),
  or the cause turns out to be a meaningless artifact.

## Decisions already locked in

- Project is reframed: **science first (what causes hyperbolicity)** → **adaptive probe** → **safety
  test**. Determinants define the probe's gate. Obfuscation test is a core deliverable.
- Data: AILuminate + Aegis (+ HarmBench/AdvBench prompts; WOS + WordNet controls). GoEmotions and
  AdvGLUE dropped.
- Token sweep includes **input-only** (new vs Raj) plus thinking/last/all × layers.
- "Reasoning model = hyperbolic" and "layer 27" are things we *test*, not assume.

## How to run

The whole flow is driven by `run_all.sh` at the project root. It runs the phases in order and
writes everything under `results/`. You can run a single phase with, e.g., `STAGE=geometry
./run_all.sh`, and each phase re-reads the previous phase's saved files, so you can stop and inspect
between phases. Start small (one model, the WordNet control) before the full matrix.
