#!/usr/bin/env bash
# NAME: causal-harm-run5
#
# Three goals (user's 24h mandate: causal WHY + usage + harm detection):
#   SHIP  — re-ship the scale-ladder + cross-family per-seed CSVs that run3
#           computed on DGX-local disk (results/tree_probe_v3/) but never git-
#           shipped (only the summary did). Needed to per-seed-verify those wins.
#   WHY   — composition_test (shrink_rho: do edges shrink with depth = additive
#           cone) on a fresh Qwen2.5-7B extraction, per layer.
#   HARM  — harm_taxonomy (Aegis 2.0, BeaverTails fallback): matched-conditioning
#           hyperbolic-vs-euclidean probe with the hazard TAXONOMY target vs the
#           BINARY safe/unsafe target. Prediction: hyperbolic helps on taxonomy
#           (Δ>0), not on binary (Δ≈0) — hierarchy helps iff labels are hierarchical.
#
# Hardened like run4: exact-venv torch, ulimit, per-stage timeout, disk-safe.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
ulimit -v unlimited 2>/dev/null || true       # the 500MB -v cap that broke run4b
export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"; mkdir -p "$HF_HOME" results/logs
LIMIT="${LIMIT:-320}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"; SEEDS="0 1 2 3 4 5"
CACHE="results/data_cache_v3"; ART="$JOB_OUT/artifacts"; mkdir -p "$ART"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-3600}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-6000}"

echo "=== causal-harm run5 on $(hostname) ==="
df -h /mnt/lab | tail -1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || { echo "torch broken — abort"; exit 1; }
python -c "import hypll;print('hypll ok')" 2>/dev/null || echo "hypll absent (closed-form gate still runs)"

# geometry gate (fast)
python - <<'PY' || { echo "geometry gate FAILED — abort"; exit 1; }
import sys; from hypprobe.geometry.matched_probe import hypll_distance_check
c=hypll_distance_check(); print("closed-form gate:",c["closed_form_ok"],"err",f"{c['closed_form_max_abs_err']:.2e}")
sys.exit(0 if c["closed_form_ok"] else 1)
PY

collect(){ local s="$1"; [ -d "$s" ]||return 0; find "$s" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) -size -20M -print0 2>/dev/null|while IFS= read -r -d '' f; do d="$ART/${f#results/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; done; }
disk_free_gb(){ df -BG /mnt/lab|tail -1|awk '{gsub(/G/,"",$4);print $4}'; }

# ---- SHIP: rescue run3's local CSVs (scale ladder + cross-family) ----
echo "########## SHIP: rescuing run3 local CSVs ##########"
if [ -d results/tree_probe_v3 ]; then
  for d in results/tree_probe_v3/*/; do
    tag="$(basename "$d")"; mkdir -p "$ART/tree_probe_v3/$tag"
    cp "$d"/*.csv "$d"/*.json "$d"/*.md "$ART/tree_probe_v3/$tag/" 2>/dev/null && echo "shipped $tag" || echo "no files for $tag"
  done
else echo "results/tree_probe_v3 absent — run3 locals not on this host"; fi

# ---- prepare datasets (tree for composition, harm for detection) ----
echo "########## prepare ##########"
python -m hypprobe.data.prepare --datasets prontoqa_tree --out "$CACHE" 2>&1 | tee -a results/logs/prep5.log
echo "--- preparing harm_taxonomy (Aegis 2.0; BeaverTails fallback) ---"
python -m hypprobe.data.prepare --datasets harm_taxonomy --out "$CACHE" 2>&1 | tee -a results/logs/prep5.log || echo "harm prepare warned"

# ---- WHY: composition_test on a fresh Qwen2.5-7B extraction ----
echo "########## WHY: composition (shrink_rho) ##########"
M="Qwen/Qwen2.5-7B"; msafe="$(echo "$M"|tr / _)"
ACT="results/activations_v3/${msafe}__prontoqa_tree_c"; GEO="results/composition_v5/${msafe}"; mkdir -p "$GEO"
if [ ! -f "$ART/composition_v5/${msafe}/composition_test.csv" ]; then
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor --model "$M" \
    --datasets prontoqa_tree --dtype fp32 --device cuda --limit "$LIMIT" --chat-mode plain \
    --max-new-tokens "$MAX_NEW_TOKENS" --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/extract5.log
  if [ -n "$(find "$ACT" -name '*.pt' 2>/dev/null|head -1)" ]; then
    timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.composition_test --activations "$ACT" \
      --out "$GEO" --dataset prontoqa_tree 2>&1 | tee -a results/logs/comp5.log || echo "composition failed"
    collect "$GEO"; rm -rf "$ACT"
  else echo "composition extract produced nothing — SKIP"; rm -rf "$ACT"; fi
fi
echo "free after WHY: $(disk_free_gb)G"

# ---- HARM: extract harm prompts, matched probe taxonomy vs binary ----
echo "########## HARM: taxonomy vs binary ##########"
HARM_ACT="results/activations_v3/${msafe}__harm"; HARM_GEO="results/harm_v5/${msafe}"; mkdir -p "$HARM_GEO"
if [ -f "$CACHE/harm_taxonomy.jsonl" ]; then
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor --model "$M" \
    --datasets harm_taxonomy --dtype fp32 --device cuda --limit 600 --chat-mode plain \
    --max-new-tokens 8 --cache "$CACHE" --out "$HARM_ACT" 2>&1 | tee -a results/logs/harm5.log
  if [ -n "$(find "$HARM_ACT" -name '*.pt' 2>/dev/null|head -1)" ]; then
    # TAXONOMY target (hierarchical hazard tree) — expect hyperbolic Δ>0
    timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.matched_probe --activations "$HARM_ACT" \
      --out "$HARM_GEO/taxonomy" --dataset harm_taxonomy --target taxonomy --seeds $SEEDS \
      2>&1 | tee -a results/logs/harm5.log || echo "harm taxonomy probe failed"
    # BINARY target (flat safe/unsafe) — expect Δ≈0 (the control)
    timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.matched_probe --activations "$HARM_ACT" \
      --out "$HARM_GEO/binary" --dataset harm_taxonomy --target depth --seeds $SEEDS \
      2>&1 | tee -a results/logs/harm5.log || echo "harm binary probe failed"
    python -m hypprobe.extract.audit_generations --activations "$HARM_ACT" --out "$HARM_GEO" 2>&1 | tee -a results/logs/harm5.log || true
    collect "$HARM_GEO"; rm -rf "$HARM_ACT"
  else echo "harm extract produced nothing — SKIP"; rm -rf "$HARM_ACT"; fi
else echo "harm_taxonomy.jsonl absent — prepare failed; SKIP harm"; fi
echo "free after HARM: $(disk_free_gb)G"

cp results/logs/*5.log "$ART/" 2>/dev/null || true
echo "=== run5 complete; artifacts in $JOB_OUT/artifacts ==="
