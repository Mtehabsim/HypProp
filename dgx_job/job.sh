#!/usr/bin/env bash
# NAME: harm-run6
#
# run6 = HARM ONLY. run5 already succeeded on SHIP (scale/cross-family CSVs) and
# WHY (composition shrink_rho) — both shipped + committed. run5's HARM arm crashed
# because Aegis category names contain '/' (e.g. 'Controlled/Regulated Substances')
# → sample_id → nonexistent nested path. Fixed in harm_taxonomy._safe_id. This job
# ONLY re-runs harm: Aegis 2.0 (917 rows, loaded fine — not gated) hazard TAXONOMY
# target vs BINARY safe/unsafe, matched-conditioning hyperbolic-vs-euclidean probe.
# Prediction: hyperbolic Δ>0 on taxonomy, Δ≈0 on binary (hierarchy helps iff labels
# are hierarchical).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
ulimit -v unlimited 2>/dev/null || true       # the 500MB -v cap that broke run4b
export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"; mkdir -p "$HF_HOME" results/logs
SEEDS="0 1 2 3 4 5"
CACHE="results/data_cache_v3"; ART="$JOB_OUT/artifacts"; mkdir -p "$ART"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-3600}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-6000}"

echo "=== harm run6 on $(hostname) ==="
df -h /mnt/lab | tail -1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || { echo "torch broken — abort"; exit 1; }

collect(){ local s="$1"; [ -d "$s" ]||return 0; find "$s" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) -size -20M -print0 2>/dev/null|while IFS= read -r -d '' f; do d="$ART/${f#results/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; done; }
disk_free_gb(){ df -BG /mnt/lab|tail -1|awk '{gsub(/G/,"",$4);print $4}'; }

M="Qwen/Qwen2.5-7B"; msafe="$(echo "$M"|tr / _)"

# ---- prepare harm dataset (fresh, with slash-safe sample_ids) ----
echo "########## prepare harm_taxonomy ##########"
rm -f "$CACHE/harm_taxonomy.jsonl"    # force rebuild with fixed ids
python -m hypprobe.data.prepare --datasets harm_taxonomy --out "$CACHE" 2>&1 | tee -a results/logs/harm6.log || echo "harm prepare warned"

# ---- HARM: extract harm prompts, matched probe taxonomy vs binary ----
echo "########## HARM: taxonomy vs binary ##########"
HARM_ACT="results/activations_v3/${msafe}__harm"; HARM_GEO="results/harm_v6/${msafe}"; mkdir -p "$HARM_GEO"
if [ -f "$CACHE/harm_taxonomy.jsonl" ]; then
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor --model "$M" \
    --datasets harm_taxonomy --dtype fp32 --device cuda --limit 600 --chat-mode plain \
    --max-new-tokens 8 --cache "$CACHE" --out "$HARM_ACT" 2>&1 | tee -a results/logs/harm6.log
  if [ -n "$(find "$HARM_ACT" -name '*.pt' 2>/dev/null|head -1)" ]; then
    # TAXONOMY target (hierarchical hazard tree) — expect hyperbolic Δ>0
    timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.matched_probe --activations "$HARM_ACT" \
      --out "$HARM_GEO/taxonomy" --dataset harm_taxonomy --target taxonomy --seeds $SEEDS \
      2>&1 | tee -a results/logs/harm6.log || echo "harm taxonomy probe failed"
    # BINARY target (flat safe/unsafe) — expect Δ≈0 (the control)
    timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.matched_probe --activations "$HARM_ACT" \
      --out "$HARM_GEO/binary" --dataset harm_taxonomy --target depth --seeds $SEEDS \
      2>&1 | tee -a results/logs/harm6.log || echo "harm binary probe failed"
    python -m hypprobe.extract.audit_generations --activations "$HARM_ACT" --out "$HARM_GEO" 2>&1 | tee -a results/logs/harm6.log || true
    collect "$HARM_GEO"; rm -rf "$HARM_ACT"
  else echo "harm extract produced nothing — SKIP"; rm -rf "$HARM_ACT"; fi
else echo "harm_taxonomy.jsonl absent — prepare failed; SKIP harm"; fi
echo "free after HARM: $(disk_free_gb)G"

cp results/logs/*6.log "$ART/" 2>/dev/null || true
echo "=== run6 (harm) complete; artifacts in $JOB_OUT/artifacts ==="
