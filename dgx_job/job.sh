#!/usr/bin/env bash
# NAME: tree-probe-run2
#
# run2: run1 extracted + audited both models fine but tree_probe HALTED on the
# HypLL cross-check (max err 2.57e-2). Root-caused locally: our Poincare distance
# is EXACT vs the textbook arcosh closed form (2e-15); the 2.57e-2 gap == our
# |d(c=0.5)-d(c=1.0)|, i.e. a HypLL curvature-CONVENTION difference, not a bug.
# Fix: the hard gate now keys on the dependency-free closed form; HypLL is a
# convention-robust soft cross-check. Added a fail-fast pre-flight so geometry
# regressions abort in seconds, not after extraction.
#
# dgx_agent.sh runs this whenever its content hash changes, with:
#   $JOB_OUT  = dgx_results/<name>-<hash>/   (ship-back dir; >50 MB quarantined)
#   $JOB_ID   = the content hash
#
# PREREGISTER3: what makes activations hierarchical + where hyperbolic helps.
# Replaces the v2 cloud-delta pipeline. The instrument decodes each prompt's
# RETAINED ground-truth is-a tree (from data/prontoqa_tree.py) from concept-token
# representations, comparing a hyperbolic decoder against a capacity- and
# conditioning-matched Euclidean one at several output dimensions, per
# (layer x role). Validated on a CPU positive control before this run: hyperbolic
# beats Euclidean at low-mid dim on a genuine branching tree (Δ up to +0.23,
# monotone in branching), radial-norm tracks depth (ρ=+0.71), noise layer ~0.
#
# KEY DESIGN NOTE vs v2: we only need PROMPT-SIDE concept representations
# (premise/query roles) plus a short answer, so max_new_tokens is small (default
# 24, not 1024). This shrinks each sample's activation file ~10-40x and makes the
# disk-safe sequential loop comfortable.
#
# DISK-SAFE (inherited from v3): process ONE arm at a time: extract -> tree_probe
# + audit -> ship small artifacts -> DELETE raw activations before the next arm.
set -uo pipefail   # NOT -e: a single stage failing must not abort the whole run
cd "$(git rev-parse --show-toplevel)"

# Home dir has no space; caches + activations live on the lab mount.
export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"
mkdir -p "$HF_HOME"

LIMIT="${LIMIT:-240}"                 # prompts per arm cap (dataset has 240 rows)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"  # we read prompt-side reps; only need a short answer
SEEDS="0 1 2 3 4 5"                   # >=6: one-sided signed-rank floor 1/64 clears 0.05
CACHE="results/data_cache_v3"
ART="$JOB_OUT/artifacts"
mkdir -p "$ART" results/logs

echo "=== tree-probe run (PREREGISTER3) on $(hostname) ==="
echo "HF_HOME=$HF_HOME  LIMIT=$LIMIT  MAX_NEW_TOKENS=$MAX_NEW_TOKENS  SEEDS='$SEEDS'"
df -h /mnt/lab | tail -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- optional deps: HypLL (Poincare cross-check) + NLTK (unused here, harmless) ----
# The tree probe HARD-HALTS if HypLL is present but disagrees with our Poincare
# distance; if absent it logs and proceeds on the c->0 unit tests. Install so the
# cross-check actually runs this time (v2 skipped it).
echo "=== ensuring hypll is installed (for the Poincare cross-check) ==="
python -c "import hypll" 2>/dev/null && echo "hypll already present" || {
  pip install --quiet hypll 2>&1 | tail -3 || echo "WARNING: hypll install failed; cross-check will be skipped"
}
python -c "import hypll; print('hypll import OK')" 2>&1 | tail -1

# ---- Phase 0.pre: fail-fast geometry gate (seconds, before any GPU time) ----
# The tree probe hard-gates on our Poincare distance matching the textbook arcosh
# closed form. Run it up front so a geometry regression fails in seconds rather
# than after ~8 min of extraction (as happened in run1, where a HypLL curvature-
# CONVENTION mismatch — not a bug — tripped the old gate post-extraction).
echo "=== pre-flight: Poincare distance correctness gate ==="
python - <<'PY'
import sys
from hypprobe.geometry.matched_probe import hypll_distance_check
c = hypll_distance_check()
print("closed-form gate:", c["closed_form_ok"], "err", f"{c['closed_form_max_abs_err']:.2e}")
print("hypll:", c.get("hypll"),
      ("best '%s' err %.2e" % (c.get("hypll_best_convention"), c.get("hypll_max_abs_err"))
       if c.get("hypll") not in (None, "not installed") else ""))
if not c["closed_form_ok"]:
    print("FATAL: Poincare distance is wrong vs the closed form — aborting.")
    sys.exit(1)
PY
if [ $? -ne 0 ]; then echo "pre-flight geometry gate FAILED — aborting run"; exit 1; fi

# ---- Phase 0: prepare the branching-ontology dataset (tree retained) ----
echo "=== preparing prontoqa_tree (fictional b1/b2/b3 + real, ground-truth tree) ==="
python -m hypprobe.data.prepare --datasets prontoqa_tree --out "$CACHE" \
  2>&1 | tee -a results/logs/prepare_tree.log
python - <<'PY' 2>&1 | tee -a results/logs/prepare_tree.log
import json, collections
rows=[json.loads(l) for l in open("results/data_cache_v3/prontoqa_tree.jsonl")]
arms=collections.Counter((r["tree_meta"]["naming"], r["tree_meta"]["branching"]) for r in rows)
print("prepared arms:", dict(arms), "total", len(rows))
assert rows and all("tree_meta" in r for r in rows), "tree_meta missing — abort"
PY

collect() {   # copy small artifacts from a results subtree into $ART
  local subtree="$1"
  [ -d "$subtree" ] || return 0
  find "$subtree" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' \
      -o -name '*.txt' \) -size -20M -print0 2>/dev/null | while IFS= read -r -d '' f; do
    dst="$ART/${f#results/}"; mkdir -p "$(dirname "$dst")"; cp "$f" "$dst"
  done
}
disk_free_gb() { df -BG /mnt/lab | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }

# ---- per-model loop: extract (plain) -> tree_probe -> ship -> delete ----
# One prompt scaffolding only (plain): the tree lives in the prompt text, so the
# chat template is not needed and would just add a scaffolding confound. Both a
# reasoning-distilled model and a base model are run so a hyperbolic advantage
# can be checked for reasoning-specificity (as in v2, where the effect appeared
# in the base model too).
MODELS=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "Qwen/Qwen2.5-7B")
for model in "${MODELS[@]}"; do
  msafe="$(echo "$model" | tr '/' '_')"
  arm="${msafe}__plain"
  ACT="results/activations_v3/$arm"
  GEO="results/tree_probe_v3/$arm"
  mkdir -p "$GEO"

  free="$(disk_free_gb)"
  echo "=== MODEL $model | free disk ${free}G ==="

  # Stage 0: extract prompt-side reps (+ short answer). Small max_new_tokens.
  echo "--- extract $arm ---"
  python -m hypprobe.extract.hidden_state_extractor --model "$model" \
    --datasets prontoqa_tree --dtype fp32 --device cuda --limit "$LIMIT" \
    --chat-mode plain --max-new-tokens "$MAX_NEW_TOKENS" \
    --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/extract_tree.log
  if [ ! -d "$ACT" ] || [ -z "$(find "$ACT" -name '*.pt' 2>/dev/null | head -1)" ]; then
    echo "MODEL $arm: no activations produced; skipping analysis"
    rm -rf "$ACT"; continue
  fi
  echo "arm $arm activations size: $(du -sh "$ACT" 2>/dev/null | cut -f1)"

  # Stage 1: generation audit (records warnings; never aborts the arm)
  echo "--- audit $arm ---"
  python -m hypprobe.extract.audit_generations --activations "$ACT" --out "$GEO" \
    2>&1 | tee -a results/logs/audit_tree.log || echo "audit flagged warnings (see CSV)"

  # Stage 2: THE tree probe (concept alignment + matched-capacity decode + dim
  # sweep + radial fingerprint + shuffled-tree null + 4-gate verdict)
  echo "--- tree_probe $arm ---"
  python -m hypprobe.geometry.tree_probe --activations "$ACT" --out "$GEO" \
    --dataset prontoqa_tree --roles premise query last \
    --dims 2 3 5 8 16 --seeds $SEEDS --layer-stride 4 \
    2>&1 | tee -a results/logs/tree_probe.log || echo "tree_probe failed for $arm"

  # ship this arm's small artifacts, then FREE the disk before the next model
  collect "$GEO"
  echo "--- deleting raw activations for $arm to free disk ---"
  rm -rf "$ACT"
  echo "arm $arm done; free disk now $(disk_free_gb)G"
done

# ---- cross-model summary: suitable positions + dose-response, per model ----
echo "=== cross-model tree-probe summary ==="
python - <<'PY' 2>&1 | tee "$ART/tree_probe_cross_model.txt" || true
import json, glob, os
files = glob.glob("results/tree_probe_v3/*/tree_probe_verdict.json")
if not files:
    print("no tree_probe_verdict.json found")
for f in sorted(files):
    arm = os.path.basename(os.path.dirname(f))
    v = json.load(open(f))
    print(f"\n### {arm}")
    pos = v.get("positions", [])
    print(f"  suitable positions (G1&G2&G3): {len(pos)}")
    for p in pos[:20]:
        print(f"    {p['arm']:16s} {p['role']:8s} L{p['layer']:<3} m{p['dim']:<3} "
              f"Δ={p['mean_delta']:+.3f} p={p.get('wilcoxon_p')} slope={p.get('slope')} "
              f"shuffle={p.get('shuffle_rho'):+.3f}")
    print("  dose-response (Δ by branching 1/2/3):")
    for d in v.get("dose_response", []):
        print(f"    {d['role']:8s} {d['max_delta_by_branching']} "
              f"monotone={d['monotone_nondecreasing']} b1_ok={d['negative_control_b1_ok']} "
              f"POSITIVE={d['dose_response_positive']}")
    print("  radial norm<->depth:")
    for r in v.get("radial", []):
        print(f"    {r['role']:8s} bestL={r['best_layer']:<3} ρ={r['radial_depth_rho']:+.3f} "
              f"passes={r['passes']}")
print("\nINTERPRETATION: a 'suitable position' passing all gates = a (layer,role,dim)")
print("where hyperbolic geometry recovers the ground-truth tree better than matched")
print("Euclidean AND the advantage grows at low dim AND survives the shuffled-tree")
print("null. A positive dose-response (Δ grows with branching, ~0 at b=1) is the")
print("mechanistic evidence that the model builds a genuine branching hierarchy.")
PY

cp results/logs/*_tree.log "$ART/" 2>/dev/null || true
cp results/logs/tree_probe.log "$ART/" 2>/dev/null || true
echo "=== job complete; artifacts in $JOB_OUT/artifacts ==="
