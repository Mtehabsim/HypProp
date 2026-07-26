#!/usr/bin/env bash
# NAME: harm-robust-run7
#
# Harden the run6 harm win (hyperbolic helps harm detection IFF labels are
# hierarchical) across MODEL and DATASET:
#   - Mistral-7B on Aegis 2.0  (cross-MODEL: does the taxonomy>binary gap replicate
#     on an independent architecture, as the tree fingerprint did?)
#   - Qwen2.5-7B on BeaverTails (cross-DATASET: independent 2nd harm corpus)
# Each = matched-conditioning hyperbolic-vs-euclidean probe, TAXONOMY target vs
# BINARY safe/unsafe. Prediction (per arm): taxonomy geometry gap >0, binary ≈0.
#
# Reuses the exact run6 harm arm (validated). ulimit-safe, per-stage timeout,
# disk-safe (delete acts per arm).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
ulimit -v unlimited 2>/dev/null || true
export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"; mkdir -p "$HF_HOME" results/logs
SEEDS="0 1 2 3 4 5"; CACHE="results/data_cache_v3"; ART="$JOB_OUT/artifacts"; mkdir -p "$ART"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-3600}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-6000}"

echo "=== harm-robust run7 on $(hostname) ==="
df -h /mnt/lab | tail -1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || { echo "torch broken — abort"; exit 1; }

collect(){ local s="$1"; [ -d "$s" ]||return 0; find "$s" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) -size -20M -print0 2>/dev/null|while IFS= read -r -d '' f; do d="$ART/${f#results/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; done; }
disk_free_gb(){ df -BG /mnt/lab|tail -1|awk '{gsub(/G/,"",$4);print $4}'; }

# harm_arm <model> <dataset> <limit> <tag>
harm_arm(){
  local M="$1" DS="$2" LIM="$3" tag="$4"; local msafe; msafe="$(echo "$M"|tr / _)"
  local ACT="results/activations_v3/${tag}"; local GEO="results/harm_v7/${tag}"; mkdir -p "$GEO"
  if [ -f "$ART/harm_v7/${tag}/taxonomy/matched_probe.csv" ]; then echo "$tag done — SKIP"; return 0; fi
  echo "=== HARM ARM $tag ($M on $DS) | free $(disk_free_gb)G ==="
  [ -f "$CACHE/${DS}.jsonl" ] || python -m hypprobe.data.prepare --datasets "$DS" --out "$CACHE" 2>&1 | tee -a results/logs/harm7.log
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor --model "$M" \
    --datasets "$DS" --dtype fp32 --device cuda --limit "$LIM" --chat-mode plain \
    --max-new-tokens 8 --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/harm7.log
  if [ -z "$(find "$ACT" -name '*.pt' 2>/dev/null|head -1)" ]; then echo "$tag: no acts — SKIP"; rm -rf "$ACT"; return 0; fi
  timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.matched_probe --activations "$ACT" \
    --out "$GEO/taxonomy" --dataset "$DS" --target taxonomy --seeds $SEEDS 2>&1 | tee -a results/logs/harm7.log || echo "$tag taxonomy failed"
  timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.matched_probe --activations "$ACT" \
    --out "$GEO/binary" --dataset "$DS" --target depth --seeds $SEEDS 2>&1 | tee -a results/logs/harm7.log || echo "$tag binary failed"
  python -m hypprobe.extract.audit_generations --activations "$ACT" --out "$GEO" 2>&1 | tee -a results/logs/harm7.log || true
  collect "$GEO"; rm -rf "$ACT"; echo "$tag done; free $(disk_free_gb)G"
}

# cross-MODEL: Mistral on Aegis
harm_arm "mistralai/Mistral-7B-v0.3" harm_taxonomy 600 "mistral_aegis"
# cross-DATASET: Qwen on BeaverTails
harm_arm "Qwen/Qwen2.5-7B" harm_beavertails 700 "qwen_beavertails"
# (bonus) cross both: Mistral on BeaverTails
harm_arm "mistralai/Mistral-7B-v0.3" harm_beavertails 700 "mistral_beavertails"

cp results/logs/*7.log "$ART/" 2>/dev/null || true
echo "=== run7 (harm robustness) complete; artifacts in $JOB_OUT/artifacts ==="
