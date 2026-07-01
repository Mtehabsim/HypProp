# hypProbe

Studying **what makes an LLM's reasoning activations hyperbolic**, and using it to
build a harder-to-fool safety probe. See `PLAN.md` for the full plan and the
verified literature behind it.

Three parts: **(1) science** — what drives hyperbolicity (token identity / order /
meaning); **(2) method** — an adaptive probe that uses hyperbolic geometry when it
helps; **(3) safety** — does it raise an attacker's obfuscation budget.

## Layout

```
hypprobe/
  geometry/    poincare.py (ball ops), delta.py (delta_rel), delta_hyperbolicity.py (Phase 1 map),
               determinants.py (Phase 1 MAIN science), structural_probe.py (reproduces Raj)
  probes/      hmlr.py (hyperbolic MLR = logistic regression on the ball), baselines.py,
               run_baselines.py, run_hmlr.py, adaptive_gate.py
  eval/        compare.py (matched comparison, MDL, significance)
  security/    obfuscation_attack.py (Phase 3: attacker budget + transfer)
  extract/     hidden_state_extractor.py (DGX Phase 0), reason_markers.py (tokenizer-aware markers)
  data/        prepare.py (datasets), synthetic.py (tests), mock_activations.py (CPU e2e)
  io.py        activation store format + pooling
tests/         geometry, probe, and marker unit tests
run_all.sh     the full pipeline (STAGE=extract|geometry|probes|security or all)
```

## Split: develop here, run on DGX

- **Core (geometry, probes, eval, security):** pure PyTorch/NumPy, **no geoopt, no
  transformers** — runs and is fully tested on a laptop CPU.
- **Extraction (Phase 0):** needs a GPU + `transformers`; runs on the **DGX**.
  The Poincare geometry is implemented from scratch, so nothing else needs the GPU.

## Quick start

```bash
pip install -r requirements.txt          # torch/numpy/sklearn already enough for the core
python -m pytest tests/ -q                # 14 tests: geometry, probes, markers

# CPU end-to-end smoke test with MOCK activations (no LLM):
python -m hypprobe.data.mock_activations --out ./results/activations
STAGE=geometry ./run_all.sh
STAGE=probes   ./run_all.sh
STAGE=security ./run_all.sh
```

On the DGX, prepare real data then run the full flow:

```bash
python -m hypprobe.data.prepare --datasets ailuminate aegis wos wordnet_control --raw ./raw_data
./run_all.sh          # runs extract -> geometry -> probes -> security
```

## What each phase saves

- `results/geometry/delta_rel.csv` — hyperbolicity (delta_rel, whitened) per layer & token source,
  PLUS label-alignment (`align_euc`, `align_hyp`, `norm_depth_corr`) and a `joint_score` implementing
  the plan's joint selection (low delta_rel AND the taxonomy embeds as a tree).
- `results/determinants/attribution.csv` — which edit drives hyperbolicity: `token_identity`,
  `order_shuffle`, and the MEANING control (`meaning_nonce`/`meaning_paraphrase` when variants were
  extracted, else the `meaning_topPC` fallback). Prepare with `--variants` to enable the real control.
- `results/geometry/structural_probe.csv` — Euclidean vs hyperbolic distance fit (reproduces Raj).
- `results/eval/summary.md` — matched hyperbolic-vs-flat verdict (+ significance.json).
- `results/security/attack.csv` — attacker budget flat vs hyperbolic, and transfer rate.

## Ground rules (enforced in code)

Whiten features before every comparison; report normalized `delta_rel` (not raw delta); match
capacity across arms; fp32 + norm clipping near the ball boundary; never coerce the model into
emitting "thinking" markers. A hyperbolic win only counts if it **survives whitening** and is
significant across seeds.
```
