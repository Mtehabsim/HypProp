#!/usr/bin/env bash
# run_gaps.sh — the ordered DGX gap-closure run (answers the re-evaluation).
#
# Runs the five pre-registered stages of PREREGISTER2.md IN ORDER; each stage
# gates the next. Everything writes under ./results/geometry_v2 (+ manifests),
# leaving the run-1 artifacts untouched as the exploratory record.
#
#   STAGE=extract   re-extraction (both chat regimes, max_new_tokens=1024)
#   STAGE=audit     Stage 1: generation audit (did the model reason? truncated?)
#   STAGE=forensics Stage 2: Atlas forensics (ceiling + candidate sweep)
#   STAGE=rung0v2   Stage 3: corrected adjudication + paired H1
#   STAGE=probe     Stage 4: matched-conditioning probe (THE decisive one)
#   STAGE=determinants Stage 5: determinants v2
#   STAGE=all       everything in order (default)
#
# Usage on the DGX:
#   ./run_gaps.sh                       # full ordered run
#   STAGE=audit ./run_gaps.sh           # one stage
#   LIMIT=50 ./run_gaps.sh              # smoke test
set -euo pipefail

# ---- config ----
MODELS=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "Qwen/Qwen2.5-7B")
DATASET="prontoqa"
RESULTS_DIR="./results"
V2_DIR="$RESULTS_DIR/geometry_v2"
ACT_DIR="$RESULTS_DIR/activations_v2"
MAX_NEW_TOKENS=1024          # v1's 256 truncated traces (audit finding)
LIMIT="${LIMIT:-300}"        # >=300 prompts for the paired H1 power target
SEEDS="0 1 2 3 4"
STAGE="${STAGE:-all}"
# ----------------

mkdir -p "$RESULTS_DIR"/logs "$V2_DIR" "$ACT_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_DIR/logs/run_gaps.log"; }
run_stage() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# Optional but recommended: HypLL cross-check for the matched probe
python -c "import hypll" 2>/dev/null || \
  log "NOTE: hypll not installed (pip install hypll) — the matched probe will skip the library cross-check"

# ---------------------------------------------------------------------------
# Phase 0 — re-extraction. BOTH chat regimes per model, so 'plain vs chat' is a
# measured axis (run-1 assumed plain was fine; the audit will tell us).
# Variants (nonce/paraphrase) included — the new augment_jsonl HARD-FAILS if a
# variant comes out identical to its original (the run-1 vacuous-paraphrase bug).
# ---------------------------------------------------------------------------
if run_stage extract; then
  log "Phase 0: preparing $DATASET (+ non-vacuous variants)"
  python -m hypprobe.data.prepare --datasets "$DATASET" --variants \
    --out "$RESULTS_DIR/data_cache_v2" 2>&1 | tee -a "$RESULTS_DIR/logs/prepare_v2.log"
  for m in "${MODELS[@]}"; do
    for mode in plain chat; do
      # Qwen2.5-7B base may have no chat template -> 'chat' would raise; use
      # auto for the chat arm so template-less models fall back loudly.
      cm="$mode"; [ "$mode" = "chat" ] && cm="auto"
      log "Phase 0: extracting $m [$mode] ($LIMIT samples, max_new_tokens=$MAX_NEW_TOKENS)"
      python -m hypprobe.extract.hidden_state_extractor --model "$m" \
        --datasets "$DATASET" --dtype fp32 --device cuda --limit "$LIMIT" \
        --chat-mode "$cm" --max-new-tokens "$MAX_NEW_TOKENS" \
        --cache "$RESULTS_DIR/data_cache_v2" --out "$ACT_DIR/$mode" \
        2>&1 | tee -a "$RESULTS_DIR/logs/extract_v2.log"
    done
  done
fi

# ---------------------------------------------------------------------------
# Stage 1 — generation audit (GATES everything: exits non-zero on a failed
# audit so the || below surfaces it and the ordered run stops).
# ---------------------------------------------------------------------------
if run_stage audit; then
  for mode in plain chat; do
    log "Stage 1: auditing generations [$mode] (CoT presence, truncation)"
    python -m hypprobe.extract.audit_generations --activations "$ACT_DIR/$mode" \
      --out "$V2_DIR/$mode" 2>&1 | tee -a "$RESULTS_DIR/logs/audit_v2.log" || {
        log "AUDIT FAILED for [$mode] — read $V2_DIR/$mode/generation_audit.csv."
        log "Per PREREGISTER2: a regime whose reasoning arm shows <50% CoT is VOID for H1/H2."
        [ "$STAGE" = "all" ] && [ "$mode" = "chat" ] && exit 2
      }
  done
  log "Stage 1 done -> READ $V2_DIR/{plain,chat}/generation_audit.csv before continuing"
fi

# ---------------------------------------------------------------------------
# Stage 2 — Atlas forensics (ceiling proof + candidate-statistic sweep).
# Uses whichever regime the audit blessed; runs on both for completeness.
# ---------------------------------------------------------------------------
if run_stage forensics; then
  for mode in plain chat; do
    log "Stage 2: Atlas forensics [$mode]"
    python -m hypprobe.geometry.atlas_forensics --activations "$ACT_DIR/$mode" \
      --out "$V2_DIR/$mode" 2>&1 | tee -a "$RESULTS_DIR/logs/forensics_v2.log"
  done
  log "Stage 2 done -> READ $V2_DIR/*/atlas_forensics_verdict.md"
fi

# ---------------------------------------------------------------------------
# Stage 3 — Rung 0 v2 (regime-matched calibration, balanced per-layer
# backgrounds, span-relative Gate B with the corrected labels, paired H1).
# ---------------------------------------------------------------------------
if run_stage rung0v2; then
  for mode in plain chat; do
    log "Stage 3: rung0_v2 [$mode]"
    python -m hypprobe.geometry.rung0_v2 --activations "$ACT_DIR/$mode" \
      --out "$V2_DIR/$mode" --project-root . \
      2>&1 | tee -a "$RESULTS_DIR/logs/rung0_v2.log"
  done
  log "Stage 3 done -> READ $V2_DIR/*/rung0_v2_verdict.md"
fi

# ---------------------------------------------------------------------------
# Stage 4 — matched-conditioning probe (THE decisive experiment: was Raj's
# hyperbolic win geometry, or probe conditioning?). Chat regime by preference
# (that's where the reasoning traces are), plus plain for the contrast.
# ---------------------------------------------------------------------------
if run_stage probe; then
  for mode in chat plain; do
    log "Stage 4: matched-conditioning probe [$mode] (seeds: $SEEDS)"
    python -m hypprobe.geometry.matched_probe --activations "$ACT_DIR/$mode" \
      --out "$V2_DIR/$mode" --dataset "$DATASET" --target depth \
      --seeds $SEEDS 2>&1 | tee -a "$RESULTS_DIR/logs/matched_probe.log"
  done
  log "Stage 4 done -> READ $V2_DIR/*/matched_probe.csv + the verdict lines in logs/matched_probe.log"
fi

# ---------------------------------------------------------------------------
# Stage 5 — determinants v2 (real nulls, powered order test, source-respecting,
# last-token cluster adjudication). Runs on the audit-blessed regime.
# ---------------------------------------------------------------------------
if run_stage determinants; then
  for mode in chat plain; do
    log "Stage 5: determinants_v2 [$mode] (source=generated)"
    python -m hypprobe.geometry.determinants_v2 --activations "$ACT_DIR/$mode" \
      --out "$RESULTS_DIR/determinants_v2/$mode" --source generated \
      2>&1 | tee -a "$RESULTS_DIR/logs/determinants_v2.log"
  done
  log "Stage 5 done -> READ $RESULTS_DIR/determinants_v2/*/attribution_v2.csv"
fi

log "GAP-CLOSURE RUN COMPLETE. Read in this order:"
log "  1. $V2_DIR/{plain,chat}/generation_audit.csv     (was the stimulus valid?)"
log "  2. $V2_DIR/*/atlas_forensics_verdict.md          (what did the Atlas actually compute?)"
log "  3. $V2_DIR/*/rung0_v2_verdict.md                 (corrected adjudication + paired H1)"
log "  4. logs/matched_probe.log verdict lines          (geometry vs conditioning — decisive)"
log "  5. $RESULTS_DIR/determinants_v2/*/attribution_v2.csv (drivers, with real nulls)"
