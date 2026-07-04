# Plan 2 — Reframed: connect Raj ↔ Atlas, answer WHEN + WHY, then the harm probe

This supersedes the framing (not the guardrails) of `PLAN.md`. It exists because two
things changed since the original plan:

1. **A direct competitor appeared and clarified the landscape.** "A Hyperbolicity Atlas of
   Large Language Model Hidden States" (anonymous ACL submission, read in full) already does
   the *descriptive* map of δ_rel across scale × depth × family × domain, on **prompt tokens
   only**, over MATH500 / HumanEval / WinoGrande / TruthfulQA. It explicitly leaves the causal
   "why" and generated tokens as future work.
2. **The codebase was hardened** (commit `ba91ed9`, "Address deep review"): the headline probe
   comparison is now whitened and capacity-matched; selectivity + MDL are wired; the
   determinants edits share one pooling operator and are gated on a placebo + std_rel; the gate
   reads the determinants driver; a negative `flat_control` and a taxonomy structural-probe
   target exist. Most of the original `PLAN.md`'s open defects are closed.

So this document is about the **scientific reframe** and the **specific new code** it needs —
not about re-litigating what `ba91ed9` already fixed.

---

## 0. The connective insight (the spine of the whole project)

Look at what the two papers find at the **final layer**, and notice they are the *same
phenomenon measured two different ways*:

- **Atlas (prompt tokens):** δ_rel drops from a ~0.99 mid-layer plateau to ~0.4–0.7 at the
  final layer, and they interpret it as **compression toward the next-token objective**
  ("final layers compress the representation … and that compression can produce a more
  tree-like distance structure").
- **Raj (reasoning tokens):** at the final layer a hyperbolic probe *survives* (0.967) where a
  flat one *collapses* (0.488) — which the honest reframe reads as **compression-robustness**,
  not hierarchy.

The Atlas measures the geometry directly (δ drops); Raj measures a *downstream probe
consequence* of the same compression (a hyperbolic reader tolerates the compressed layer).
**Neither connected them, and the Atlas never saw generation** — even though generated tokens
are literally the representations produced *by* the next-token objective both papers credit.

**Unifying hypothesis (new, falsifiable):**

> The final-layer tree-likeness is prediction-oriented compression, and **generation amplifies
> it**. Generated/reasoning tokens should be *more* tree-like at the final layers than prompt
> tokens, most strongly in reasoning-tuned models — and that is precisely what makes Raj's
> hyperbolic probe survive on DeepSeek-R1 but not on Qwen2.5.

Everything below tests this, decomposes its cause, and only then builds the harm probe on top.

---

## 1. Two questions, restated so they are actually novel

### WHEN — "at which layer / token-source does it become tree-like, and does generation amplify it?"
The generic layer×scale×family×domain sweep is now **occupied by the Atlas** and largely
converges on the known compression story — do **not** redo it as a descriptive sweep. Our
defensible version adds the one axis the Atlas excluded (**generated / thinking tokens**) and
frames it as *connecting Raj to the Atlas*: reasoning-token hyperbolic-probe survival (Raj) is
the generated-token continuation of the prompt-token compression trajectory (Atlas).

### WHY — "what causes the tree-likeness: token identity, order, or meaning — and is it real hierarchy or just anisotropy/compression?"
The Atlas states outright: *"the present measurements do not identify the mechanism causally"*
and *"geometry-changing interventions are left to future work."* This is the uncontested cell.
Our contribution is the **causal attribution**, done cleanly, plus the **whitening
adjudication** the Atlas skipped (they measured on raw Euclidean coordinates → possibly
anisotropy per Ethayarajh/Park).

### The payoff — harm/harmless probe
Once WHEN gives the operating point and WHY gives the driver, the probe is *determined*, and
the honesty of the safety claim follows from the driver (meaning-driven → hard to obfuscate;
identity/compression-driven → cheap to mimic, Bailey-style).

---

## 2. Rung 0 — the whitening adjudication (gates the entire thesis; run first)

**Why first:** every downstream part assumes a *real* hyperbolic signal. If the tree-likeness
is just anisotropy, Park et al. (2025) win and we should write the deflationary paper. One
cheap experiment greenlights or kills the thesis.

**Why it is NOT a clean yes/no** (the correction we converged on):
- Per-cloud ZCA removes the cloud's own 2nd-order structure — and hierarchy often lives *in*
  the anisotropic directions, so per-cloud whitening can **erase real signal** ("vanished ≠
  fake").
- δ_rel under whitening in the N≪d regime is **non-monotone in the retained PCA dim k**, so a
  "survives" verdict can secretly mean "survives at k=256."
- Per-cloud ZCA is **not Park's hypothesis**. Park whiten by a *model-level* covariance (the
  causal inner product over unembedding directions). A per-cloud verdict neither confirms nor
  rebuts Park.

**So Rung 0 is a metric FAMILY × calibration controls × regimes, read as a pattern:**

| Metric | Removes | A low-δ here means… |
|---|---|---|
| `raw` (Atlas) | nothing | baseline; could be anisotropy |
| `pca_only` | nothing, just drops empty dims | is it dims or rescaling doing the work? |
| `per_cloud` | this cloud's 2nd-order structure | strong but biased *against* real hierarchy |
| `background` | *generic* anisotropy (global cov), keeps cloud structure | the honest middle |
| `causal` (optional) | model-level anisotropy (Park metric) | rebuts/supports Park directly |

Controls (from `synthetic.py`): **synthetic tree must stay low-δ under every metric** (else
whitening is broken and "vanished" is uninterpretable); **Gaussian/sphere must stay high**
(else whitening is manufacturing tree-likeness). **Sweep k** and require the *verdict*, not
just the value, to be stable. Everything gated on a **bootstrap-over-points** std, not just MC
quadruple noise.

---

## 3. Detailed code changes (mapped to the current, post-`ba91ed9` files)

Legend: **[NEW]** new file · **[EDIT]** modify existing · **[P0/P1/P2]** priority.

### P0 — Rung 0 + generated-token axis (unlocks WHEN and gates everything)

#### 3.1 [EDIT] `hypprobe/io.py` — add the `generated` token source  **[P0]**
- `TOKEN_SOURCES = ("input", "thinking", "last", "all")` → add **`"generated"`**.
  Result: `("input", "generated", "thinking", "last", "all")`.
- In `pool_features`, add the branch:
  ```python
  elif token_source == "generated":
      mask = is_gen if is_gen is not None and is_gen.size else np.zeros(h.shape[0], bool)
  ```
  (`is_generated` is already stored per sample — nothing upstream changes.)
- **Why:** this is the single change that lets us contrast prompt vs generated vs thinking and
  reproduce the Atlas trajectory *and* extend it — the core of WHEN. `input` stays as the
  Atlas-comparable prompt-only source.
- Note for interpretation: `pool_features` mean-pools (order-invariant). That is fine for the
  *probe/label-alignment* view; the *order* question is handled in determinants with the
  position-weighted operator, and in `token_geometry` with `pos_dist_corr`. Document this so no
  one re-introduces the "mean-pool kills order" confound.

#### 3.2 [EDIT] `hypprobe/geometry/delta.py` — whitening becomes a metric family + honest std  **[P0]**
- Add `metric: str = "per_cloud"` to `delta_hyperbolicity(...)` with values
  `{"raw", "pca_only", "per_cloud", "background", "causal"}`:
  - `raw` → `do_whiten=False` (reproduce the Atlas).
  - `pca_only` → PCA-project to k (existing `whiten` SVD path) but **do not divide by std**
    (skip the `/ std` step) — isolates dimensionality from rescaling.
  - `per_cloud` → current `whiten()` behavior.
  - `background` → accept `bg_transform` (a callable/`(mean, W)` fit elsewhere) and apply it
    instead of per-cloud fit. Add a param `bg_transform=None`.
  - `causal` → optional; accept a model-level whitening `(mean, W)` (Park causal inner product).
    Leave a stub + docstring if the estimator isn't ready; do not fake it.
- Add `k` / `pca_cap` sweep support: allow `pca_cap` to be a list; return one `DeltaResult` per
  k so the caller can check verdict stability. (Or expose a thin helper `delta_over_k(...)`.)
- **New honest noise floor:** add `bootstrap_std` to `DeltaResult` — resample the *point set*
  (with replacement) B times, recompute δ_rel, take std. `std_rel` (MC quadruple noise) stays
  but is explicitly labelled the *weaker* floor. Downstream gates use `max(std_rel,
  bootstrap_std)`.
- **Why:** turns "whiten vs raw" (a strawman for Park, knob-dependent) into the real
  adjudication, and fixes the "std_rel only measures quadruple noise" weakness at the source.

#### 3.3 [NEW] `hypprobe/geometry/rung0.py` — the adjudication experiment  **[P0]**
- For each (model, layer, token_source) × each `metric` × a small `k`-grid: compute δ_rel +
  bootstrap_std. Also run the **calibration clouds** from `synthetic.py` (synthetic tree,
  Gaussian, sphere) through the *same* metrics/k.
- Fit the **background** transform once per (model, layer) from a large generic pooled-token
  sample (reuse `build_token_matrix` over all datasets) and pass it in.
- Emit `results/geometry/rung0.csv` with columns:
  `model, layer, token_source, metric, k, delta_rel, bootstrap_std, cloud_kind` where
  `cloud_kind ∈ {data, tree_control, gaussian_control, sphere_control}`.
- Emit a plain-English `rung0_verdict.md`:
  - **Gate A:** does `tree_control` stay low-δ under every metric/k? (else: whitening broken.)
  - **Gate B:** does the data's final-layer drop survive `per_cloud` AND `background`, stably
    across k, above bootstrap_std? → `REAL_HIERARCHY` / `ANISOTROPY_ARTIFACT` /
    `AMBIGUOUS(k-dependent)`.
- **Why:** this one CSV *is* the "does the Atlas headline survive whitening?" result —
  publishable independently, and the go/no-go for the whole project.

#### 3.4 [EDIT] `hypprobe/data/synthetic.py` — guarantee the calibration clouds exist  **[P0]**
- Ensure there are clean generators: a **synthetic tree** point cloud (known δ≈0), an
  **isotropic Gaussian** (δ high), a **sphere** (δ high). If any are missing, add them. These
  are the instruments Rung 0 depends on; they must be model-free and deterministic per seed.

### P1 — statistical honesty + the matched-checkpoint "why" lever

#### 3.5 [EDIT] `hypprobe/geometry/delta_hyperbolicity.py` — seed loop + metric-family columns  **[P1]**
- Currently runs `seed=0` only. Loop over the same `SEEDS` used in `run_all.sh`; report
  **seed-level std** (across-seed variance of δ_rel), which is the honest floor the guardrails
  actually meant. Add a `--metrics` arg so the layer map can be emitted under `raw` (Atlas
  parity) and `per_cloud`/`background` (our correction) side by side.
- Add the **generated** source to the sweep (from 3.1) so `delta_rel.csv` carries the
  prompt-vs-generated trajectory, not just prompt.
- **Why:** makes the depth trajectory directly comparable to the Atlas *and* shows whether
  their result survives whitening — plus fixes single-seed fragility.

#### 3.6 [NEW] `hypprobe/geometry/when_contrast.py` — WHEN, the Raj↔Atlas bridge  **[P1]**
- Input: the activation store. For each model, compute δ_rel (whitened + raw) at every layer
  for `token_source ∈ {input(prompt), generated, thinking, last}`.
- Emit `results/geometry/when_contrast.csv` and test the three pre-registered hypotheses:
  - **H1 (generation amplifies):** `δ(generated, final) < δ(prompt, final)` within a model.
  - **H2 (explains Raj model-specificity):** the prompt→generated gap is larger for
    reasoning-tuned models (DeepSeek-R1-Distill) than for base/instruct — logged as the
    mechanism Raj never gave.
  - **H3 (locus):** thinking-marker tokens are the low-δ locus.
- Report each with the bootstrap/seed std and an explicit pass/fail.
- **Why:** this is the self-contained WHEN contribution and the literal connection between the
  two papers. Depends on 3.1 (`generated` source) and real generated activations (DGX).

#### 3.7 [NEW] `hypprobe/geometry/checkpoint_contrast.py` — the cleanest WHY lever  **[P1]**
- The Atlas's strongest empirical finding is that **specialization reshapes geometry**. Turn
  their *observation* into our *controlled intervention*: same prompts, same tokenizer/arch/
  scale, vary only training — `{Qwen2.5-7B-Base, -Instruct, -Coder-Instruct, -Math-7B,
  DeepSeek-R1-Distill}`.
- Emit per-checkpoint `delta_rel` (metric family) + `align_hyp` (taxonomy recovery via
  `label_alignment`) on the same dataset. Headline contrast: **base vs aligned on harmful
  prompts** — does alignment *build* the hierarchy a harm probe reads?
- **Why:** cleaner than input interventions (no pooling/tokenization confound; the input is
  literally identical), and directly safety-relevant.

#### 3.8 [EDIT] `hypprobe/data/variants.py` — stop the degenerate meaning control  **[P1]**
- The paraphrase control swaps from an 11-word synonym dict → most prompts unchanged → a fake
  null. Two acceptable fixes:
  1. Wire a real paraphraser (small local model or a broader rule set), **or**
  2. **Hard-gate:** in `augment_jsonl`, if `make_paraphrase(prompt) == prompt` for more than a
     small fraction, mark those rows `variant="paraphrase_degenerate"` so determinants **refuses
     to report `meaning_paraphrase`** on them (a degenerate control must not masquerade as a
     null result).
- In `determinants.py`, relabel `meaning_topPC` in the CSV as `meaning_topPC_NONCAUSAL` and
  exclude it from driver selection (it's a PCA artifact, not a meaning edit) — keep it only as
  a diagnostic column.
- **Why:** the meaning leg is a third of the main claim; a degenerate control silently produces
  "meaning doesn't matter."

#### 3.9 [EDIT] `hypprobe/geometry/determinants.py` — seed loop + use bootstrap_std  **[P1]**
- Loop seeds (as 3.5); the driver's `trustworthy` test should use `max(bootstrap_std,
  seed_std, placebo_mag)`, not just MC `std_rel`. The placebo + shared-pooling design from
  `ba91ed9` stays — this only strengthens the noise floor.

### P2 — causal payoff + safety (higher effort; the conference-bar rungs)

#### 3.10 [NEW] `hypprobe/geometry/causal_patch.py` — Rung 3, mechanism  **[P2]**
- Find the low-rank hierarchy subspace (top directions of the whitened, taxonomy-aligned
  features). **Ablate/patch** it in the activations and measure whether **(a)** taxonomy
  recovery drops *and* **(b)** model behavior (refusal/output on a held-out set) changes.
- Emit `results/geometry/causal_patch.csv`.
- **Why:** everything else is "change input → watch geometry" (associational). This is the
  causal, behavior-linked test — and it is the *real* obfuscation lever (obfuscating a
  behavior-relevant direction has a measurable cost, the Bailey claim the current feature-space
  attack cannot support). This replaces the argmax-driver→safety bridge with a mechanism.

#### 3.11 [EDIT] `hypprobe/security/obfuscation_attack.py` — bidirectional transfer + honest scope  **[P2]**
- Add the missing **`hyp→flat`** transfer direction (currently only `flat→hyp` via
  `_transfer_row`), so an asymmetry claim is supportable at all.
- Keep the honest `margin_l2 = diagnostic` labeling; scale the headline claim to whatever
  Rung 3 (3.10) shows about behavioral cost.
- **Why:** one-directional transfer cannot distinguish "hyperbolic is more robust" from "just
  different coordinates."

#### 3.12 [EDIT] `hypprobe/eval/compare.py` — Wilcoxon + multiple-comparison correction  **[P2]**
- Replace the normal-approx z-test in `_paired_test` with a **paired Wilcoxon signed-rank**
  test, and apply **Holm/BH correction** across the (model × dataset × layer × source) cells
  (don't pool correlated seeds into one test). Selectivity stays the primary metric.
- **Why:** the current test is anti-conservative at small n and pools correlated seeds.

---

## 4. Orchestration — `run_all.sh` changes + hard kill-switches  **[P1]**

Add a `rung0` stage *before* geometry and make three gates **hard stops** (exit non-zero with a
clear message), not just log lines:

```
STAGE=rung0     → rung0.py                 → rung0.csv + rung0_verdict.md
    GATE A (tree control survives)  : else STOP "whitening broken"
    GATE B (data survives whitening): if ANISOTROPY_ARTIFACT → STOP + banner
                                       "pivot to deflationary paper"
STAGE=geometry  → delta map (seed-looped, metric family) + when_contrast.py
                  + checkpoint_contrast.py + determinants (seed-looped)
                  + structural_probe (depth=Raj repro, taxonomy=harm) + token_geometry
STAGE=probes    → baselines + hmlr + adaptive_gate (already whitened, selectivity/MDL)
    GATE E (controls): WordNet positive control hyperbolic-wins AND flat_control nulls
                       → else STOP "pipeline not trustworthy on harm data"
STAGE=security  → obfuscation (bidirectional) + causal_patch behavioral cost
```

Implement gates as a small `hypprobe/eval/gates.py` that reads the relevant CSV and returns
exit codes; `run_all.sh` calls it between stages.

---

## 5. Experiment plan with pre-registered predictions

| Stage | Question | Metric / DV | Pass condition | If it fails |
|---|---|---|---|---|
| **Rung 0** | Is the tree-likeness real or anisotropy? | δ_rel across metric family, bootstrap-std, k-swept, calibrated | data survives `per_cloud`+`background`, stable in k, tree-control survives | **pivot to deflationary paper** (Park wins) — still publishable as an Atlas correction |
| **WHEN (H1)** | Does generation amplify compression? | δ(generated,final) vs δ(prompt,final) | generated significantly lower | Raj's effect isn't about generation; drop that framing |
| **WHEN (H2)** | Why Raj on DeepSeek not Qwen? | prompt→generated gap, reasoning vs base | gap larger in reasoning models | model-specific compression, weaker story |
| **WHEN (H3)** | Where is the locus? | δ by token_source | thinking tokens lowest-δ | locus is elsewhere; re-target probe |
| **WHY (driver)** | Identity / order / meaning? | Δδ_rel per edit, placebo + bootstrap/seed-std gated | one driver beats placebo AND noise | "no trustworthy driver" — honest null |
| **WHY (checkpoint)** | Does alignment build hierarchy? | δ + align_hyp, base vs aligned | aligned differs above noise | hierarchy is pretrained, not alignment |
| **WHY (causal)** | Is the subspace load-bearing? | behavior change under patch | patch changes behavior | structure is epiphenomenal; safety claim weakens |
| **Controls** | Pipeline trustworthy? | WordNet win + flat_control null | both hold | fix pipeline before harm claims |
| **Safety** | Raises attacker cost? | bidirectional transfer + behavioral cost | asymmetry + real cost | honest claim: "geometry gives little defense" |

**The paper this yields (either branch is publishable):**
- *Positive:* "Prompt-token hyperbolicity (Atlas) and reasoning-token hyperbolic-probe survival
  (Raj) are two views of one prediction-oriented compression trajectory; we complete it on
  generated tokens, show it survives whitening as real hierarchy, attribute it causally to
  [meaning / alignment] via matched-checkpoint and subspace-patching experiments, and build a
  harm probe whose obfuscation cost we can actually measure."
- *Deflationary:* "The reasoning-LLM hyperbolic-probe win is prediction-oriented compression /
  token-identity, not hierarchy: it does not survive whitening (correcting the Atlas), and
  hyperbolic geometry does not raise obfuscation cost (attacks transfer)." — consistent with
  Park 2025 + Bailey 2024, and honest.

---

## 6. Positioning vs. the literature (cite these explicitly)

- **Atlas** (anon ACL, prompt-only δ map over MATH500/HumanEval/WinoGrande/TruthfulQA): the
  descriptive "when." We do the causal "why", add **generated tokens**, and **whiten** (they
  used raw Euclidean coordinates — our Rung 0 tests whether their headline survives).
- **Raj 2026** (non-archival workshop; DeepSeek L27 hyperbolic-probe survival): we explain *why*
  it is model-specific (H2) and connect it to the Atlas trajectory.
- **Park et al. 2025** (ICLR oral; hierarchy is Euclidean after the causal metric): the null
  Rung 0 must beat; add the `causal` metric arm to test it directly.
- **Bailey et al. 2024** (obfuscated activations): the safety threat model; our claim must rest
  on the behavioral cost from Rung 3, not feature-space margin.
- **Neighbors to distinguish, not compete with:** HyPE (ICLR 2026, VLM shared-embedding,
  unsupervised); CurvaLID / Yung et al. 2025 (curvature+LID on *prompts*, input-space);
  frequency/anisotropy attribution (Timkey 2021, Ethayarajh 2019, Zhou 2021). Our uncontested
  cell: **hyperbolic + decoder-LLM generated-token activations + branching harm taxonomy +
  causal-subspace obfuscation.**

---

## 7. Build order (smallest first, each independently valuable)

1. **P0:** `io.py` generated source → `delta.py` metric family + bootstrap_std →
   `synthetic.py` controls → `rung0.py`. Run on **mock** first to prove the harness recovers a
   *planted* answer (tree stays low, Gaussian stays high); then on DGX activations.
2. **P1:** seed-loop geometry + determinants; `when_contrast.py`; `checkpoint_contrast.py`;
   `variants.py` gate; `run_all.sh` gates A/B/E.
3. **P2:** `causal_patch.py`; bidirectional transfer; Wilcoxon + correction in `compare.py`.

**What's already done (do not redo):** whitened matched probe arms, selectivity + MDL,
determinants shared-pooling + placebo + std gate, gate reads driver, taxonomy structural target,
`flat_control`, `STAGE` env override, position-axis no-op fix.

## 8. Honest risks
- **Rung 0 may null the thesis** (the point of running it first). Budget for the deflationary
  paper as a real outcome.
- **Generated-token axis needs real decoding** (DGX; the mock cannot stand in for H1–H3).
- **The safety space is filling fast** — stay strictly inside the uncontested cell (§6).
- **Rung 3 (causal patching) is the highest-effort, highest-payoff rung** — it is what turns
  "another δ paper" into a mechanistic account that earns the safety claim.
