# PREREGISTER3 — What makes activations hierarchical, and where hyperbolic space helps

Status: pre-registered **before** the DGX run. Decision rules are fixed here so
the analysis cannot drift to fit whatever the data shows (the discipline that
kept v2 honest). Supersedes the *goal* of PREREGISTER2 (which asked "does the
hyperbolic reasoning probe beat Euclidean?" and answered NO — see
`dgx_results/gap-closure-run3-*`). This asks a sharper, mechanistic pair of
questions and — critically — fixes the flaw that made v2's null partly
inevitable.

---

## 0. The flaw v2 could not see, and why it matters

`build_prontoqa` generates `chain = rng.sample(stems, depth+1)` — every prompt
is a **linear chain** `A→B→C→D`. A path is a degenerate tree with **branching
factor 1**, and a path **embeds isometrically into 1-D Euclidean space**.
Hyperbolic geometry's entire advantage is exponential volume growth to separate
the *leaves of a branching tree* (Sarkar 2011; Sala et al. 2018: a tree needs
2-D hyperbolic vs Ω(#nodes) Euclidean dims — but only when it branches). On a
path there is nothing for curvature to buy, so a null hyperbolic advantage on
linear-chain ProntoQA is **forced by the stimulus**, not discovered in the model.

We own the generator. The fix is to make **branching factor the independent
variable**. This converts "no advantage" from a foregone conclusion into a
measurable dose-response curve, and gives us a built-in negative control (b=1).

---

## 1. Questions

- **Q-WHAT (cause):** Under what conditions do LLM activations encode a
  *branching* hierarchy that a decoder can recover — and is that hierarchy
  *assembled in-context* or *retrieved from pretraining*?
- **Q-WHERE (position):** At which (layer × token-role × decode-dimension) cells,
  if any, does hyperbolic geometry recover the **ground-truth** tree at lower
  distortion than a capacity- and conditioning-matched Euclidean decoder?

Both are answerable **either way**: a positive answer localizes a usable
hyperbolic read-out; a negative answer is a clean "even with a genuine branching
ground-truth tree, matched capacity, and in-context assembly, the residual
stream encodes hierarchy flatly" — which kills a fashionable assumption.

---

## 2. Instrument (replaces cloud four-point δ as the headline)

We never again headline the four-point δ of an undifferentiated token cloud
(ceiling-bounded ≤0.293; confounded by class clustering — proven in v2). The new
primary instrument is **distortion of a KNOWN ground-truth tree at matched
capacity**:

Given, for one prompt, the concept set `{c_k}` with ground-truth pairwise tree
distance `D_tree[k,l]` (path length in the is-a DAG we generated), and the
concept-token representations `x_k` at (layer ℓ, role r):

1. Fit a decoder `g_m: x → R^m` (or Poincaré ball `B^m_c`) minimizing stress
   `Σ (dist_g(x_k,x_l) − D_tree[k,l])² / Σ D_tree²`, and score **Spearman ρ** of
   decoded vs ground-truth distances on a **held-out concept/pair split**.
2. Two geometries, **identical** conditioning (LayerNorm + spectral-norm +
   bounded scaling + MDR), identical params, identical epochs/opt/lr/init, **no
   learnable curvature** (capacity match) — reusing `matched_probe.MatchedProbe`:
   - `cond_euclidean` (curvature 0)
   - `hyperbolic` (Poincaré, c=0.5)
   (`bare_euclidean` is retained only as the Raj-confound reference, not for the
   geometry gap.)
3. **Advantage** `Δ(ℓ,r,m) = ρ_hyperbolic − ρ_cond_euclidean`.

Secondary instruments:
- **Distortion-vs-dimension curve** `Δ(m)` for `m ∈ {2,3,5,8,16}`. The
  *fingerprint of curvature actually being used* is `Δ` **increasing as m
  shrinks** (hyperbolic packs a branching tree in 2-D that Euclidean cannot).
- **Radial-norm ↔ generality** (cheapest, sharpest): rank-correlation between a
  concept's representation norm (whitened) — and its `poincare.dist0` after the
  hyperbolic embedding — and its **node depth** in the ground-truth tree (root
  shallow/central, leaves deep/peripheral). Class clustering is *angular*;
  radial ordering by generality is **not** explained by a cluster null, so this
  cleanly separates hierarchy from the clustering artifact that sank v2's
  last-token story.
- **Ancestor-retrieval mAP** (Nickel–Kiela) as a distortion-free cross-check.

Diagnostic only (never headline): δ_rel of the *decoded* subspace, now
interpretable because we control its contents.

---

## 3. Manipulations (Q-WHAT) — the generator knobs

New generator `data/prontoqa_tree.py` produces genuine **branching** ontologies
and RETAINS the ground-truth tree (v2's generator discarded it). Per prompt we
store `tree_meta`: the edge list, per-concept node id, node depth, the queried
entity, the target concept, and the gold answer.

| Knob | Levels | Controlled for | Prediction if hierarchy is real & used |
|---|---|---|---|
| **branching b** | 1, 2, 3 | **matched node count** (deeper when narrower) so difficulty/token-count ≈ constant | Δ and the low-dim advantage **grow with b**; **b=1 ⇒ Δ≈0** (negative control) |
| **naming** | fictional (nonce) / real (curated is-a) | same tree shapes | fictional recovers ⇒ tree **assembled in-context by attention**, not retrieved |
| **shuffled tree** (decode-time) | on/off | same tokens & reps, `D_tree` permuted | ρ **collapses to ~0** — isolates "tracks *this* tree" from "tokens are separable" |

The b=1 arm is the single most important control: it is the same experiment on a
path, where the theory says Δ *must* be ~0. If Δ>0 at b=1, the instrument is
manufacturing advantage and every positive result is void.

---

## 4. Token roles (Q-WHERE, the "which representation")

Concept representations are read at each role via string alignment on the saved
`tokens` (sliding-window, sub-token aware — reuse `reason_markers.ThinkingMatcher`):
- **premise** — the concept token(s) inside the `Every X is a Y` premises (the
  in-context definition site).
- **query** — the concept token(s) in the `Question: … is a Z?` clause.
- **last** — final generated token (v2 showed this is mostly answer-class
  clustering; kept as the contrast that should *lose*).

Layers: full sweep at coarse stride for the map, dense near any winning band.
A-priori bet (superposition/linear-rep prior): any advantage is **modest and
localized** — mid-depth (~50–70%), on **premise/query concept tokens, not
last** — and shows as a **radial-norm signal + low-dim distortion advantage**,
not a global cloud property.

---

## 5. Decision rules — a cell is a SUITABLE POSITION for hyperbolic space iff ALL hold

Fixed thresholds (JSON below). All on **whitened** features (except the radial
fingerprint, read on RAW features — whitening erases the norm↔depth signal),
train-only fit, held-out **prompt** split, **≥6 seeds**, **one-sided** Wilcoxon
signed-rank on per-seed paired diffs. (Six seeds, one-sided, is a deliberate fix:
the two-sided signed-rank floor is 2/2ⁿ, so at n=5 the smallest achievable p is
0.0625 > 0.05 and G1 could *never* fire — the trap that made v2's large
conditioning gaps read as "p≈0.062, just short". The hypothesis is directional,
which licenses one-sided; n=6 gives a one-sided floor of 1/64 = 0.016.)

1. **G1 advantage:** `mean Δ(ℓ,r,m*) ≥ delta_margin` at the best low dim m*,
   with one-sided Wilcoxon `p < 0.05`.
2. **G2 curvature fingerprint:** slope of `Δ(m)` in m is **negative** (Δ rises as
   m shrinks) with `Δ(m=2) − Δ(m=16) ≥ slope_margin`. A one-off high-dim win
   fails this gate.
3. **G3 controls:** shuffled-tree ρ `< shuffle_ceiling` (structure is real);
   within-ontology **cluster-null excess** `> cluster_excess_margin` (beyond
   class separation); HypLL cross-check passes (`max_abs_err < 1e-4`) or the run
   halts.
4. **G4 causal (stretch, reported separately):** patching the decoded
   tree-subspace changes the model's multi-hop answer above `patch_effect_min`.
   Reported as supporting evidence; NOT required for a "position identified"
   claim, which rests on G1–G3.

**Dose-response headline (Q-WHAT):** hierarchy is declared *genuine and used*
only if Δ is monotone non-decreasing in b across {1,2,3} with `Δ(b=1) ≤
delta_margin/2` AND `Δ(b=3) ≥ delta_margin`. Otherwise report "no
branching-dependent hyperbolic advantage" — a valid, publishable negative.

**Radial go/no-go (pre-run gate):** if, on the CPU positive control
(`synthetic.hierarchy_features`, a genuine branching tree), the hyperbolic
decoder does NOT beat Euclidean at m=2 AND radial-norm does NOT track depth
(Spearman ≥ 0.5), the instrument is broken → fix before any GPU time.

```json
{
  "delta_margin": 0.05,
  "slope_margin": 0.03,
  "shuffle_ceiling": 0.10,
  "cluster_excess_margin": 0.05,
  "radial_depth_rho_min": 0.30,
  "positive_control_radial_min": 0.50,
  "patch_effect_min": 0.05,
  "n_seeds": 6,
  "wilcoxon_alpha": 0.05,
  "wilcoxon_alternative": "greater",
  "decode_dims": [2, 3, 5, 8, 16],
  "branching_levels": [1, 2, 3],
  "curvature": 0.5
}
```

---

## 6. Confounds already burned, and how they are closed here

- **Probe conditioning (Raj):** geometry gap is hyp − **cond_euclidean**, both
  fully conditioned. Never hyp − bare.
- **Class clustering (v2 last-token):** ground-truth tree is over *concepts
  within a prompt*, not answer classes; cluster-null excess (G3) + radial-norm
  (angular-invariant) required.
- **Dimension scale of δ (v2):** we do not use raw δ as a headline; ρ vs a fixed
  ground-truth `D_tree` is dimension-comparable, and capacity is matched.
- **Broken variant (v2 Qwen paraphrase):** generation audit runs every arm;
  reps are read at **prompt-side** premise/query roles (present regardless of
  generation quality), so a degenerate generation cannot silently null the
  concept reps. Small `max_new_tokens` (we only need the answer, not a long CoT)
  also shrinks disk.
- **fp32 near the ball boundary:** enforced (extractor already upcasts); MDR +
  `project_to_ball` guard the boundary.
- **HypLL never validated (v2):** `pip install hypll` on the DGX; cross-check is
  a hard gate.

---

## 7. Deliverables

1. `Δ(ℓ, role, m)` heatmaps per arm; the branching dose-response curve.
2. Radial-norm↔depth correlation per (ℓ, role).
3. The list of SUITABLE POSITIONS passing G1–G3 (possibly empty — that is a
   result), with the usage each enables.
4. A verdict doc mirroring v2's, and a memory update.

Positive → "hierarchy is built in-context at layers X on concept tokens;
hyperbolic recovers it at Y× lower dim; usage = low-dim read-out head / radial
steering / deception-as-tree-distortion detector." Negative → "even with a
genuine branching ground-truth tree and matched capacity, the stream encodes
hierarchy flatly; hyperbolic helps only as an explicit low-dim bottleneck."
