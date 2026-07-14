#!/usr/bin/env bash
# NAME: hierarchy-campaign-run3
#
# 18-HOUR RESEARCH CAMPAIGN: build a robust, GENERAL picture of activation
# hierarchy, beyond the single PREREGISTER3 probe (run2, DeepSeek+Qwen 7B on
# ProntoQA is-a trees). Three axes, each shipping artifacts as it completes so
# partial completion still yields results:
#
#   Phase A — SCALE LADDER: Qwen2.5 {1.5B, 3B, 7B, 14B} on prontoqa_tree.
#             Does the low-dim hyperbolic advantage sharpen / shift layer with
#             scale? (7B already done in run2 but re-run here for a clean ladder.)
#   Phase B — RELATION TYPES: Qwen2.5-7B on relation_trees {is_a, part_of,
#             causes, flat_set}. Is hierarchy is-a specific or generic to any
#             structured relation? flat_set is the NEGATIVE CONTROL (Δ must ~0).
#   Phase C — CROSS-FAMILY: Llama-3.1-8B + Mistral-7B on prontoqa_tree. Does the
#             L8-12 / low-dim fingerprint replicate outside the Qwen family?
#
# DISK-SAFE (per-arm extract -> analyse -> ship -> delete). fp32. Small
# max_new_tokens (prompt-side reps). Every model wrapped so one failure (e.g. a
# gated/absent checkpoint) is logged and SKIPPED, never aborting the campaign.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"
mkdir -p "$HF_HOME" results/logs

LIMIT="${LIMIT:-240}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"   # prompt-side reps; only need a short answer
SEEDS="0 1 2 3 4 5"
CACHE="results/data_cache_v3"
ART="$JOB_OUT/artifacts"
mkdir -p "$ART"

echo "=== hierarchy campaign (run3) on $(hostname) ==="
echo "LIMIT=$LIMIT MAX_NEW_TOKENS=$MAX_NEW_TOKENS SEEDS='$SEEDS'"
df -h /mnt/lab | tail -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import hypll; print('hypll import OK')" 2>&1 | tail -1 || echo "hypll absent (closed-form gate still runs)"

# fail-fast geometry gate (seconds)
echo "=== pre-flight: Poincare distance correctness gate ==="
python - <<'PY'
import sys
from hypprobe.geometry.matched_probe import hypll_distance_check
c = hypll_distance_check()
print("closed-form gate:", c["closed_form_ok"], "err", f"{c['closed_form_max_abs_err']:.2e}")
sys.exit(0 if c["closed_form_ok"] else 1)
PY
[ $? -ne 0 ] && { echo "geometry gate FAILED — abort"; exit 1; }

# prepare both datasets once
echo "=== preparing datasets ==="
python -m hypprobe.data.prepare --datasets prontoqa_tree relation_trees --out "$CACHE" \
  2>&1 | tee -a results/logs/prepare_campaign.log

collect() { local s="$1"; [ -d "$s" ] || return 0
  find "$s" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) \
    -size -20M -print0 2>/dev/null | while IFS= read -r -d '' f; do
    dst="$ART/${f#results/}"; mkdir -p "$(dirname "$dst")"; cp "$f" "$dst"; done; }
disk_free_gb() { df -BG /mnt/lab | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }

# run_arm <model> <dataset> <tag> <roles...>
run_arm() {
  local model="$1" dataset="$2" tag="$3"; shift 3; local roles="$*"
  local msafe; msafe="$(echo "$model" | tr '/' '_')"
  local ACT="results/activations_v3/${msafe}__${dataset}"
  local GEO="results/tree_probe_v3/${tag}"
  mkdir -p "$GEO"
  echo "=== ARM $tag ($model on $dataset) | free $(disk_free_gb)G ==="
  python -m hypprobe.extract.hidden_state_extractor --model "$model" \
    --datasets "$dataset" --dtype fp32 --device cuda --limit "$LIMIT" \
    --chat-mode plain --max-new-tokens "$MAX_NEW_TOKENS" \
    --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/extract_campaign.log
  if [ -z "$(find "$ACT" -name '*.pt' 2>/dev/null | head -1)" ]; then
    echo "ARM $tag: no activations (model absent/gated?) — SKIP"; rm -rf "$ACT"; return 0
  fi
  echo "arm $tag size: $(du -sh "$ACT" 2>/dev/null | cut -f1)"
  python -m hypprobe.extract.audit_generations --activations "$ACT" --out "$GEO" \
    2>&1 | tee -a results/logs/audit_campaign.log || echo "audit warnings (see CSV)"
  python -m hypprobe.geometry.tree_probe --activations "$ACT" --out "$GEO" \
    --dataset "$dataset" --roles $roles --dims 2 3 5 8 16 --seeds $SEEDS \
    --layer-stride 4 2>&1 | tee -a results/logs/tree_probe_campaign.log \
    || echo "tree_probe failed for $tag"
  collect "$GEO"
  rm -rf "$ACT"
  echo "arm $tag done; free $(disk_free_gb)G"
}

# Ordered VALUE-FIRST so partial completion still answers the biggest questions,
# and the redundant 7B-on-prontoqa_tree rung is skipped (run2 already has it).

# ---- Phase B: RELATION TYPES + NEGATIVE CONTROL (highest value) ----
# is_a / part_of / causes / flat_set on one model. flat_set (a star, no
# hierarchy) MUST give Δ~0 while is_a gives Δ>0 on the SAME model — the cleanest
# possible proof the rig measures real hierarchy, not an artifact.
echo "########## PHASE B: RELATION TYPES (incl. flat_set negative control) ##########"
run_arm "Qwen/Qwen2.5-7B" relation_trees "relations__Qwen2.5-7B" premise last

# ---- Phase A: SCALE LADDER (Qwen2.5 family) on prontoqa_tree ----
# Endpoints first (1.5B, 14B) so the biggest scale contrast lands even if the
# campaign is cut short; 3B fills the middle; 7B skipped (in run2).
echo "########## PHASE A: SCALE LADDER ##########"
for m in "Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-3B"; do
  msafe="$(echo "$m" | tr '/' '_')"
  run_arm "$m" prontoqa_tree "scaleladder__${msafe}" premise last
done

# ---- Phase C: CROSS-FAMILY on prontoqa_tree (may skip if gated/absent) ----
echo "########## PHASE C: CROSS-FAMILY ##########"
for m in "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"; do
  msafe="$(echo "$m" | tr '/' '_')"
  run_arm "$m" prontoqa_tree "crossfamily__${msafe}" premise last
done

# ---- campaign summary across everything shipped ----
echo "=== campaign summary ==="
python - <<'PY' 2>&1 | tee "$ART/campaign_summary.txt" || true
import json, glob, os
files = sorted(glob.glob("results/tree_probe_v3/*/tree_probe_verdict.json"))
if not files: print("no verdicts yet"); raise SystemExit
for f in files:
    tag = os.path.basename(os.path.dirname(f)); v = json.load(open(f))
    pos = v.get("positions", [])
    print(f"\n### {tag}: {len(pos)} suitable positions")
    # best premise cell per arm-of-interest
    best = {}
    for p in pos:
        k = (p["arm"], p["role"])
        if k not in best or p["mean_delta"] > best[k]["mean_delta"]: best[k] = p
    for (arm, role), p in sorted(best.items()):
        print(f"  {arm:16s} {role:8s} bestL={p['layer']:<3} m{p['dim']} Δ={p['mean_delta']:+.3f} "
              f"slope={p.get('slope')} shuffle={p.get('shuffle_rho'):+.3f}")
    for r in v.get("radial", []):
        print(f"  radial {r['role']:8s} L{r['best_layer']:<3} ρ={r['radial_depth_rho']:+.3f} pass={r['passes']}")
PY
cp results/logs/*campaign*.log "$ART/" 2>/dev/null || true
echo "=== campaign complete; artifacts in $JOB_OUT/artifacts ==="
