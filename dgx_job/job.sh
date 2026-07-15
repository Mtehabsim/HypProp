#!/usr/bin/env bash
# NAME: hierarchy-campaign-run4b
#
# run4b = run4 + reordered scale ladder (14B, 3B, then the flaky 1.5B LAST) so the
# informative rungs land before the rung that hung run3. Content hash differs from
# the earlier run4 push, so the agent treats this as the current job.
#
# run4 = run3 + hang-proofing. run3 completed Phase B (relations, all shipped)
# but then HUNG on a single Qwen2.5-1.5B tree_probe cell (fictional_b1 last L16):
# the child produced no log for >60 min while the agent kept heartbeating, so the
# whole campaign (scale ladder + cross-family) was blocked. Fixes here:
#   - hard `timeout` around extraction (60m) and tree_probe (90m) per model, so a
#     stuck cell kills that arm and the campaign PROCEEDS instead of wedging;
#   - RESUMABILITY: any arm whose verdict already shipped is skipped, so run4 does
#     NOT redo Phase B (relations) or any completed rung — it resumes the ladder.
# NOTE: run4 can only start once run3's hung process is cleared on the DGX (the
# agent won't launch a new job while the old one's process is alive). If run3 is
# still hung, this file is staged and waiting.
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

# RESUME: seed this run's artifacts with any verdicts already shipped by a prior
# campaign run (run3), so run_arm's skip-if-verdict-present check resumes rather
# than redoing completed arms (Phase B relations, etc.).
for prev in dgx_results/hierarchy-campaign-run3-*/artifacts/tree_probe_v3; do
  [ -d "$prev" ] || continue
  echo "=== resume: importing prior verdicts from $prev ==="
  find "$prev" -name "tree_probe_verdict.json" | while read -r v; do
    d="$ART/tree_probe_v3/$(basename "$(dirname "$v")")"; mkdir -p "$d"
    cp "$(dirname "$v")"/*.csv "$(dirname "$v")"/*.json "$(dirname "$v")"/*.md "$d/" 2>/dev/null || true
  done
done

collect() { local s="$1"; [ -d "$s" ] || return 0
  find "$s" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) \
    -size -20M -print0 2>/dev/null | while IFS= read -r -d '' f; do
    dst="$ART/${f#results/}"; mkdir -p "$(dirname "$dst")"; cp "$f" "$dst"; done; }
disk_free_gb() { df -BG /mnt/lab | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }

# Hard wall-clock caps so ONE stuck cell can never wedge the whole campaign
# (run3 hung on a single 1.5B tree_probe cell and blocked all remaining phases).
EXTRACT_TIMEOUT="${EXTRACT_TIMEOUT:-3600}"     # 60 min/model extraction
PROBE_TIMEOUT="${PROBE_TIMEOUT:-5400}"         # 90 min/model tree_probe grid

# run_arm <model> <dataset> <tag> <roles...>
run_arm() {
  local model="$1" dataset="$2" tag="$3"; shift 3; local roles="$*"
  local msafe; msafe="$(echo "$model" | tr '/' '_')"
  local ACT="results/activations_v3/${msafe}__${dataset}"
  local GEO="results/tree_probe_v3/${tag}"
  mkdir -p "$GEO"
  # RESUMABILITY: if this arm's verdict already shipped (prior run), skip it.
  if [ -f "$JOB_OUT/artifacts/tree_probe_v3/${tag}/tree_probe_verdict.json" ] \
     || [ -f "$GEO/tree_probe_verdict.json" ]; then
    echo "=== ARM $tag: verdict already present — SKIP (resume) ==="; return 0
  fi
  echo "=== ARM $tag ($model on $dataset) | free $(disk_free_gb)G ==="
  timeout "$EXTRACT_TIMEOUT" python -m hypprobe.extract.hidden_state_extractor \
    --model "$model" --datasets "$dataset" --dtype fp32 --device cuda --limit "$LIMIT" \
    --chat-mode plain --max-new-tokens "$MAX_NEW_TOKENS" \
    --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/extract_campaign.log
  if [ -z "$(find "$ACT" -name '*.pt' 2>/dev/null | head -1)" ]; then
    echo "ARM $tag: no activations (model absent/gated/timeout?) — SKIP"; rm -rf "$ACT"; return 0
  fi
  echo "arm $tag size: $(du -sh "$ACT" 2>/dev/null | cut -f1)"
  timeout 900 python -m hypprobe.extract.audit_generations --activations "$ACT" --out "$GEO" \
    2>&1 | tee -a results/logs/audit_campaign.log || echo "audit warnings/timeout (see CSV)"
  # THE probe, hard-capped. If it hangs on a cell, timeout kills it, the arm is
  # skipped, and the campaign proceeds to the next model instead of wedging.
  timeout "$PROBE_TIMEOUT" python -m hypprobe.geometry.tree_probe --activations "$ACT" \
    --out "$GEO" --dataset "$dataset" --roles $roles --dims 2 3 5 8 16 --seeds $SEEDS \
    --layer-stride 4 2>&1 | tee -a results/logs/tree_probe_campaign.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 124 ] && echo "tree_probe TIMED OUT for $tag (killed after ${PROBE_TIMEOUT}s) — skipping"
  [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && echo "tree_probe failed for $tag (rc=$rc)"
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
# ORDER: 14B first (biggest scale contrast vs the 7B run2 baseline), then 3B,
# then 1.5B LAST — 1.5B is the rung that hung run3 (fictional_b1 last L16), so if
# it re-hangs its per-stage timeout kills only IT, after the informative rungs are
# already banked. (run4 is resumable, so a re-run only redoes unshipped arms.)
echo "########## PHASE A: SCALE LADDER (14B, 3B, then flaky 1.5B last) ##########"
for m in "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-1.5B"; do
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
