#!/usr/bin/env bash
# run_all.sh — full hypProbe pipeline: extract → geometry+science → probes → safety
#
# Runs the four phases in order and writes everything under ./results.
# Run one phase only:   STAGE=geometry ./run_all.sh
# Smoke test on CPU:     MODELS="" DATASETS=(wordnet_control) ./run_all.sh   (skips extract if no GPU)
# Each phase re-reads the previous phase's saved files, so you can stop and
# inspect between phases. Start small (one model, the wordnet_control) first.
set -euo pipefail

# ---- config (edit these) ----
MODELS=("Qwen/Qwen2.5-7B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-Guard-3-8B")
DATASETS=("ailuminate" "aegis" "wos" "wordnet_control")   # controls first is fine
SEEDS=(0 1 2 3 4)
RESULTS_DIR="./results"
DTYPE="fp32"          # fp32 recommended (bf16 breaks near the Poincare boundary)
DEVICE="cuda"         # DGX
LIMIT=0               # cap samples per dataset (0 = all); set small for a smoke test
SOURCE="last"         # token source for the probe phases (input|thinking|last|all)
STAGE="all"           # override: STAGE=geometry ./run_all.sh
# -----------------------------

mkdir -p "$RESULTS_DIR"/{logs,activations,geometry,determinants,probes,eval,security,data_cache}
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_DIR/logs/run_all.log"; }
run_stage() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# Phase 0 — data + extraction
if run_stage extract; then
  log "Phase 0: caching datasets (with nonce/paraphrase variants for the meaning control)"
  python -m hypprobe.data.prepare --datasets "${DATASETS[@]}" --variants --out "$RESULTS_DIR/data_cache"
  log "Phase 0: extracting hidden states (needs GPU + transformers)"
  for m in "${MODELS[@]}"; do
    [ -z "$m" ] && continue
    python -m hypprobe.extract.hidden_state_extractor --model "$m" --datasets "${DATASETS[@]}" \
      --dtype "$DTYPE" --device "$DEVICE" --limit "$LIMIT" \
      --cache "$RESULTS_DIR/data_cache" --out "$RESULTS_DIR/activations" \
      2>&1 | tee -a "$RESULTS_DIR/logs/extract.log"
  done
fi

# Phase 1 — geometry map + determinants (main science) + Raj reproduction
if run_stage geometry; then
  log "Phase 1: delta-hyperbolicity map (whitened)"
  python -m hypprobe.geometry.delta_hyperbolicity --activations "$RESULTS_DIR/activations" \
    --whiten --out "$RESULTS_DIR/geometry" 2>&1 | tee -a "$RESULTS_DIR/logs/delta.log"
  log "Phase 1: reproduce Raj structural-probe (sanity)"
  python -m hypprobe.geometry.structural_probe --activations "$RESULTS_DIR/activations" \
    --dataset prontoqa --out "$RESULTS_DIR/geometry" 2>&1 | tee -a "$RESULTS_DIR/logs/struct.log" || \
    log "  (structural probe skipped: no prontoqa activations yet)"
  log "Phase 1: determinants (token / order / meaning)"
  python -m hypprobe.geometry.determinants --activations "$RESULTS_DIR/activations" \
    --whiten --out "$RESULTS_DIR/determinants" 2>&1 | tee -a "$RESULTS_DIR/logs/determinants.log"
fi

# Phase 2 — probes (flat LR baselines + hyperbolic H-MLR + adaptive gate)
if run_stage probes; then
  log "Phase 2: training probes across seeds"
  for s in "${SEEDS[@]}"; do
    python -m hypprobe.probes.run_baselines --activations "$RESULTS_DIR/activations" --seed "$s" \
      --source "$SOURCE" --out "$RESULTS_DIR/probes" 2>&1 | tee -a "$RESULTS_DIR/logs/baselines.log"
    python -m hypprobe.probes.run_hmlr --activations "$RESULTS_DIR/activations" --seed "$s" \
      --source "$SOURCE" --out "$RESULTS_DIR/probes" 2>&1 | tee -a "$RESULTS_DIR/logs/hmlr.log"
    python -m hypprobe.probes.adaptive_gate --activations "$RESULTS_DIR/activations" --seed "$s" \
      --source "$SOURCE" --determinants "$RESULTS_DIR/determinants" \
      --out "$RESULTS_DIR/probes" 2>&1 | tee -a "$RESULTS_DIR/logs/adaptive.log"
  done
  log "Phase 2: comparison (accuracy, selectivity, MDL, dimension curve)"
  python -m hypprobe.eval.compare --probes "$RESULTS_DIR/probes" --out "$RESULTS_DIR/eval" \
    2>&1 | tee -a "$RESULTS_DIR/logs/compare.log"
fi

# Phase 3 — safety: obfuscation attack + budget + transfer
if run_stage security; then
  log "Phase 3: obfuscation attack (flat vs hyperbolic, transfer both ways)"
  python -m hypprobe.security.obfuscation_attack --activations "$RESULTS_DIR/activations" \
    --source "$SOURCE" --out "$RESULTS_DIR/security" 2>&1 | tee -a "$RESULTS_DIR/logs/security.log"
fi

log "DONE. Key outputs:"
log "  geometry map   -> $RESULTS_DIR/geometry/delta_rel.csv"
log "  main science   -> $RESULTS_DIR/determinants/attribution.csv"
log "  probe compare  -> $RESULTS_DIR/eval/summary.md"
log "  safety result  -> $RESULTS_DIR/security/attack.csv"
