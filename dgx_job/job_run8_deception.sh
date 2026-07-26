#!/usr/bin/env bash
# NAME: deception-run8
#
# Deception probe: does inducing a WRONG answer distort the mid-stack tree?
# Same branching trees under honest / sandbag / distractor instructions (shared
# tree_meta). tree_probe decodes each condition's arm separately. Compare
# tree-decodability (rho_hyp, Δ, peak layer) across conditions:
#   - unchanged under sandbag -> model builds tree faithfully then lies at readout
#     (probe can catch the lie: tree-implied answer vs emitted answer)
#   - drops/flattens under sandbag -> deception corrupts the internal hierarchy
# Both Qwen (base) and DeepSeek-distill (reasoning) — deception may differ by whether
# the model actually reasons. Generation audit records the EMITTED answer so we can
# later check tree-implied vs emitted.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
ulimit -v unlimited 2>/dev/null || true
export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"; mkdir -p "$HF_HOME" results/logs
SEEDS="0 1 2 3 4 5"; CACHE="results/data_cache_v3"; ART="$JOB_OUT/artifacts"; mkdir -p "$ART"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-3600}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-6000}"

echo "=== deception run8 on $(hostname) ==="
df -h /mnt/lab | tail -1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || { echo "torch broken — abort"; exit 1; }
python - <<'PY' || { echo "geometry gate FAILED — abort"; exit 1; }
import sys; from hypprobe.geometry.matched_probe import hypll_distance_check
c=hypll_distance_check(); print("closed-form gate:",c["closed_form_ok"]); sys.exit(0 if c["closed_form_ok"] else 1)
PY

collect(){ local s="$1"; [ -d "$s" ]||return 0; find "$s" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) -size -20M -print0 2>/dev/null|while IFS= read -r -d '' f; do d="$ART/${f#results/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; done; }
disk_free_gb(){ df -BG /mnt/lab|tail -1|awk '{gsub(/G/,"",$4);print $4}'; }

# generous max_new_tokens so the model actually produces a reasoned answer we can
# audit (honest vs sandbag emitted answer). 256 is enough for a short CoT.
python -m hypprobe.data.prepare --datasets deception_trees --out "$CACHE" 2>&1 | tee -a results/logs/decep8.log

for M in "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "Qwen/Qwen2.5-7B"; do
  msafe="$(echo "$M"|tr / _)"; ACT="results/activations_v3/${msafe}__decep"; GEO="results/deception_v8/${msafe}"; mkdir -p "$GEO"
  if [ -f "$ART/deception_v8/${msafe}/tree_probe_verdict.json" ]; then echo "$msafe done — SKIP"; continue; fi
  echo "=== DECEPTION $msafe | free $(disk_free_gb)G ==="
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor --model "$M" \
    --datasets deception_trees --dtype fp32 --device cuda --limit 240 --chat-mode plain \
    --max-new-tokens 256 --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/decep8.log
  if [ -z "$(find "$ACT" -name '*.pt' 2>/dev/null|head -1)" ]; then echo "$msafe: no acts — SKIP"; rm -rf "$ACT"; continue; fi
  # audit records EMITTED answer per condition (honest vs sandbag behaviour)
  python -m hypprobe.extract.audit_generations --activations "$ACT" --out "$GEO" 2>&1 | tee -a results/logs/decep8.log || true
  # tree_probe decodes each condition arm (honest/sandbag/distractor) separately
  timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.tree_probe --activations "$ACT" \
    --out "$GEO" --dataset deception_trees --roles premise last --dims 2 3 5 8 16 \
    --seeds $SEEDS --layer-stride 4 2>&1 | tee -a results/logs/decep8.log || echo "$msafe tree_probe failed"
  collect "$GEO"; rm -rf "$ACT"; echo "$msafe done; free $(disk_free_gb)G"
done

# comparison summary: tree-decodability by condition (does sandbag distort it?)
echo "=== deception summary: decodability honest vs sandbag vs distractor ==="
python - <<'PY' 2>&1 | tee "$ART/deception_summary.txt" || true
import csv, glob, os, numpy as np
from collections import defaultdict
for f in sorted(glob.glob("results/deception_v8/*/tree_probe.csv")):
    model=os.path.basename(os.path.dirname(f)); R=list(csv.DictReader(open(f)))
    for r in R:
        for k in("rho_hyp","delta"): r[k]=float(r[k])
        r["layer"]=int(r["layer"]); r["dim"]=int(r["dim"])
    print(f"\n### {model}")
    for cond in ("honest_b2","sandbag_b2","distractor_b2"):
        sub=[r for r in R if r["arm"]==cond and r["role"]=="premise" and r["dim"]==5]
        if not sub: print(f"  {cond}: no data"); continue
        by=defaultdict(list)
        for r in sub: by[r["layer"]].append(r["rho_hyp"])
        peakL=max(by,key=lambda L:np.mean(by[L])); peak=np.mean(by[peakL])
        print(f"  {cond:14s}: peak decodability rho_hyp={peak:.3f} @L{peakL}")
    print("  -> if sandbag peak << honest peak, deception FLATTENS the tree;")
    print("     if ~equal, model builds tree faithfully & lies only at readout.")
PY
cp results/logs/*8.log "$ART/" 2>/dev/null || true
echo "=== run8 (deception) complete; artifacts in $JOB_OUT/artifacts ==="
