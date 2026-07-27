#!/usr/bin/env bash
# NAME: openq-run9
#
# Close the open questions in one campaign (user: "keep DGX busy, answer every
# question"). DGX runs one job at a time; this batches the ready ones:
#   OQ1 (mechanism): composition_test with the NEW radial-scaling exponent alpha
#        on fresh Qwen2.5-7B tree activations. alpha~0.5 => orthogonal accumulation
#        (resolves the paradox: norm grows w/ depth as sqrt, edges DON'T shrink);
#        alpha~1 aligned; alpha~0 = the falsified shrinking-cone.
#   OQ3 (harm router): does hyperbolic BLOCK harm better + GENERALIZE to unseen
#        categories? nearest-leaf routing hyp vs euclidean vs flat-logreg, blocking
#        F1 + zero-shot-to-held-out-category F1. On Aegis (Qwen, Mistral, Llama)
#        + BeaverTails. HF_TOKEN is exported -> Llama-3.1 now accessible.
# Reuses harm extractions from run6/run7 where present (re-extracts if absent).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
ulimit -v unlimited 2>/dev/null || true
export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"; mkdir -p "$HF_HOME" results/logs
SEEDS="0 1 2 3 4 5"; CACHE="results/data_cache_v3"; ART="$JOB_OUT/artifacts"; mkdir -p "$ART"
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-3600}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-6000}"

echo "=== openq run9 on $(hostname) ==="
df -h /mnt/lab | tail -1
python -c "import torch;print('torch',torch.__version__,torch.cuda.is_available())" || { echo "torch broken — abort"; exit 1; }
python -c "import sklearn;print('sklearn',sklearn.__version__)" || echo "sklearn MISSING — flat baseline will be skipped"

collect(){ local s="$1"; [ -d "$s" ]||return 0; find "$s" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) -size -20M -print0 2>/dev/null|while IFS= read -r -d '' f; do d="$ART/${f#results/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; done; }
disk_free_gb(){ df -BG /mnt/lab|tail -1|awk '{gsub(/G/,"",$4);print $4}'; }

# ---- OQ1: mechanism exponent alpha (fresh Qwen tree extraction) ----
echo "########## OQ1: composition mechanism (radial alpha) ##########"
[ -f "$CACHE/prontoqa_tree.jsonl" ] || python -m hypprobe.data.prepare --datasets prontoqa_tree --out "$CACHE" 2>&1 | tee -a results/logs/oq9.log
MO="Qwen/Qwen2.5-7B"; ms="$(echo "$MO"|tr / _)"; ACT="results/activations_v3/${ms}__tree_oq"; GEO="results/composition_v9/${ms}"; mkdir -p "$GEO"
if [ ! -f "$ART/composition_v9/${ms}/composition_test.csv" ]; then
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor --model "$MO" \
    --datasets prontoqa_tree --dtype fp32 --device cuda --limit 320 --chat-mode plain \
    --max-new-tokens 16 --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/oq9.log
  if [ -n "$(find "$ACT" -name '*.pt' 2>/dev/null|head -1)" ]; then
    timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.composition_test --activations "$ACT" \
      --out "$GEO" --dataset prontoqa_tree 2>&1 | tee -a results/logs/oq9.log || echo "composition failed"
    collect "$GEO"; rm -rf "$ACT"
  fi
fi
echo "free after OQ1: $(disk_free_gb)G"

# ---- OQ3: harm router (blocking + zero-shot), multi-model, multi-dataset ----
echo "########## OQ3: harm router ##########"
router_arm(){
  local M="$1" DS="$2" LIM="$3" tag="$4"; local ms; ms="$(echo "$M"|tr / _)"
  local ACT="results/activations_v3/${tag}__harm"; local GEO="results/harm_router_v9/${tag}"; mkdir -p "$GEO"
  if [ -f "$ART/harm_router_v9/${tag}/harm_router.csv" ]; then echo "$tag done — SKIP"; return 0; fi
  echo "=== ROUTER $tag ($M on $DS) | free $(disk_free_gb)G ==="
  [ -f "$CACHE/${DS}.jsonl" ] || python -m hypprobe.data.prepare --datasets "$DS" --out "$CACHE" 2>&1 | tee -a results/logs/oq9.log
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor --model "$M" \
    --datasets "$DS" --dtype fp32 --device cuda --limit "$LIM" --chat-mode plain \
    --max-new-tokens 8 --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/oq9.log
  if [ -z "$(find "$ACT" -name '*.pt' 2>/dev/null|head -1)" ]; then echo "$tag: no acts (gated?) — SKIP"; rm -rf "$ACT"; return 0; fi
  timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.harm_router --activations "$ACT" \
    --out "$GEO" --dataset "$DS" --seeds $SEEDS 2>&1 | tee -a results/logs/oq9.log || echo "$tag router failed"
  collect "$GEO"; rm -rf "$ACT"; echo "$tag done; free $(disk_free_gb)G"
}
router_arm "Qwen/Qwen2.5-7B"          harm_taxonomy    700 "qwen_aegis"
router_arm "mistralai/Mistral-7B-v0.3" harm_taxonomy    700 "mistral_aegis"
router_arm "meta-llama/Llama-3.1-8B"   harm_taxonomy    700 "llama_aegis"
router_arm "Qwen/Qwen2.5-7B"          harm_beavertails 800 "qwen_beavertails"

# ---- summary ----
echo "=== openq summary ==="
python - <<'PY' 2>&1 | tee "$ART/openq_summary.txt" || true
import csv, glob, os, numpy as np
print("### OQ1 mechanism (radial alpha; ~0.5 orthogonal-accum, ~1 aligned, ~0 shrink-cone)")
for f in glob.glob("results/composition_v9/*/composition_test.csv"):
    R=list(csv.DictReader(open(f)))
    for arm in ("fictional_b2","real_b2"):
        av=[float(r["radial_alpha"]) for r in R if r["arm"]==arm and r["radial_alpha"] not in("","nan")]
        sr=[float(r["shrink_rho"]) for r in R if r["arm"]==arm and r["shrink_rho"] not in("","nan")]
        if av: print(f"  {arm}: alpha={np.mean(av):+.2f}  shrink_rho={np.mean(sr):+.2f}")
print("\n### OQ3 harm router (F1: hyperbolic vs euclidean vs flat; blocking + zero-shot)")
for f in sorted(glob.glob("results/harm_router_v9/*/harm_router.csv")):
    tag=os.path.basename(os.path.dirname(f)); R=list(csv.DictReader(open(f)))
    print(f"  {tag}:")
    for ev in ("blocking","zeroshot"):
        d={}
        for arm in ("hyperbolic","cond_euclidean","flat_logreg"):
            v=[float(r["f1"]) for r in R if r["eval"]==ev and r["arm"]==arm and r["f1"] not in("","nan")]
            if v: d[arm]=np.mean(v)
        if d: print(f"    {ev:9s}: "+" ".join(f"{a.split('_')[0]}={d[a]:.3f}" for a in d)
                     +(f"  [hyp-flat={d.get('hyperbolic',0)-d.get('flat_logreg',0):+.3f}]" if 'flat_logreg' in d and 'hyperbolic' in d else ""))
PY
cp results/logs/*9.log "$ART/" 2>/dev/null || true
echo "=== run9 (open questions) complete; artifacts in $JOB_OUT/artifacts ==="
