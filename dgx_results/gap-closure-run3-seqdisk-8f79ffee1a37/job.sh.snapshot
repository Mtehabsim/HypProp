#!/usr/bin/env bash
# NAME: gap-closure-run3-seqdisk
#
# dgx_agent.sh runs this whenever its content hash changes, with:
#   $JOB_OUT  = dgx_results/<name>-<hash>/   (ship-back dir; >50 MB quarantined)
#   $JOB_ID   = the content hash
#
# DISK-SAFE architecture (v3): the fp32 all-layer activations are HUGE
# (~200-450 MB/sample at 1024 tokens) and the lab mount is ~98% full with only
# ~280 GB free. So instead of extracting all 4 arms up front (peak ~250-550 GB),
# we process ONE arm at a time: extract -> run every per-arm analysis stage on
# it -> ship the small artifacts -> DELETE the raw activations before the next
# arm. Peak disk = one arm (~60 GB). The cross-model H2 note is assembled at the
# end from the tiny per-arm H1 CSVs (which persist).
set -uo pipefail   # NOT -e: a single stage failing must not abort the whole run
cd "$(git rev-parse --show-toplevel)"

# Home dir has no space; caches + activations live on the lab mount.
export HF_HOME="/mnt/lab/Mo/hyperbolic1/.hf_cache"
mkdir -p "$HF_HOME"

LIMIT="${LIMIT:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SEEDS="0 1 2 3 4"
CACHE="results/data_cache_v2"
ART="$JOB_OUT/artifacts"
mkdir -p "$ART"

echo "=== gap-closure v3 (disk-safe, sequential arms) on $(hostname) ==="
echo "HF_HOME=$HF_HOME  LIMIT=$LIMIT  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
df -h /mnt/lab | tail -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- Phase 0.a: prepare data once (variants included; hard-fails if vacuous) ----
echo "=== preparing prontoqa (+ non-vacuous variants) ==="
python -m hypprobe.data.prepare --datasets prontoqa --variants --out "$CACHE" \
  2>&1 | tee -a results/logs/prepare_v3.log

collect() {   # copy small artifacts from a results subtree into $ART
  local subtree="$1"
  [ -d "$subtree" ] || return 0
  find "$subtree" -type f \( -name '*.csv' -o -name '*.md' -o -name '*.json' \
      -o -name '*.txt' \) -size -20M -print0 2>/dev/null | while IFS= read -r -d '' f; do
    dst="$ART/${f#results/}"; mkdir -p "$(dirname "$dst")"; cp "$f" "$dst"
  done
}

disk_free_gb() { df -BG /mnt/lab | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }

# ---- per-arm loop: extract -> analyze -> ship -> delete ----
MODELS=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "Qwen/Qwen2.5-7B")
for model in "${MODELS[@]}"; do
  msafe="$(echo "$model" | tr '/' '_')"
  for mode in plain chat; do
    cm="$mode"; [ "$mode" = "chat" ] && cm="auto"   # base model w/o template -> falls back loudly
    arm="${msafe}__${mode}"
    ACT="results/activations_v2/$arm"
    GEO="results/geometry_v2/$arm"
    DET="results/determinants_v2/$arm"
    mkdir -p "$GEO" "$DET"

    free="$(disk_free_gb)"
    echo "=== ARM $arm | free disk ${free}G ==="
    if [ "${free:-0}" -lt 70 ]; then
      echo "WARNING: only ${free}G free before extracting $arm; proceeding but watch for ENOSPC"
    fi

    # Stage 0: extract this arm only
    echo "--- extract $arm ---"
    python -m hypprobe.extract.hidden_state_extractor --model "$model" \
      --datasets prontoqa --dtype fp32 --device cuda --limit "$LIMIT" \
      --chat-mode "$cm" --max-new-tokens "$MAX_NEW_TOKENS" \
      --cache "$CACHE" --out "$ACT" 2>&1 | tee -a results/logs/extract_v3.log
    if [ ! -d "$ACT" ] || [ -z "$(find "$ACT" -name '*.pt' 2>/dev/null | head -1)" ]; then
      echo "ARM $arm: no activations produced; skipping analysis for this arm"
      rm -rf "$ACT"; continue
    fi
    echo "arm $arm activations size: $(du -sh "$ACT" 2>/dev/null | cut -f1)"

    # Stage 1: generation audit (never aborts the arm; records warnings)
    echo "--- audit $arm ---"
    python -m hypprobe.extract.audit_generations --activations "$ACT" --out "$GEO" \
      2>&1 | tee -a results/logs/audit_v3.log || echo "audit flagged warnings (see CSV)"

    # Stage 2: Atlas forensics (ceiling + candidate sweep)
    echo "--- atlas_forensics $arm ---"
    python -m hypprobe.geometry.atlas_forensics --activations "$ACT" --out "$GEO" \
      2>&1 | tee -a results/logs/forensics_v3.log || echo "forensics failed for $arm"

    # Stage 3: rung0_v2 (calibration, balanced background, span-relative, paired H1)
    echo "--- rung0_v2 $arm ---"
    python -m hypprobe.geometry.rung0_v2 --activations "$ACT" --out "$GEO" \
      --project-root . 2>&1 | tee -a results/logs/rung0_v3.log || echo "rung0_v2 failed for $arm"

    # Stage 4: matched-conditioning probe (THE decisive one)
    echo "--- matched_probe $arm ---"
    python -m hypprobe.geometry.matched_probe --activations "$ACT" --out "$GEO" \
      --dataset prontoqa --target depth --seeds $SEEDS \
      2>&1 | tee -a results/logs/matched_probe_v3.log || echo "matched_probe failed for $arm"

    # Stage 5: determinants v2 (real nulls, powered order test, last-token adj.)
    echo "--- determinants_v2 $arm ---"
    python -m hypprobe.geometry.determinants_v2 --activations "$ACT" --out "$DET" \
      --source generated 2>&1 | tee -a results/logs/determinants_v3.log \
      || echo "determinants_v2 failed for $arm"

    # ship this arm's small artifacts, then FREE the disk before the next arm
    collect "$GEO"; collect "$DET"
    echo "--- deleting raw activations for $arm to free disk ---"
    rm -rf "$ACT"
    echo "arm $arm done; free disk now $(disk_free_gb)G"
  done
done

# ---- cross-arm H2: direction consistency from the persisted H1 CSVs ----
echo "=== cross-arm H2 summary (from per-arm h1_paired_v2.csv) ==="
python - <<'PY' 2>&1 | tee "$ART/h2_cross_arm_summary.txt" || true
import csv, glob, os
rows = []
for f in glob.glob("results/geometry_v2/*/h1_paired_v2.csv"):
    arm = os.path.basename(os.path.dirname(f))
    for r in csv.DictReader(open(f)):
        r["arm"] = arm
        rows.append(r)
if not rows:
    print("no h1_paired_v2.csv found")
else:
    print(f"{'arm':40s} {'model':45s} {'median_diff':>12s} {'wilcoxon_p':>12s} verdict")
    for r in sorted(rows, key=lambda x: x["arm"]):
        print(f"{r.get('arm',''):40s} {r.get('model',''):45s} "
              f"{str(r.get('median_diff','')):>12s} {str(r.get('wilcoxon_p','')):>12s} "
              f"{r.get('verdict','')}")
    print("\nH2 (per PREREGISTER2): reasoning-model gap should exceed base-model gap.")
    print("With one reasoning/base pair per chat regime, report DIRECTION only; a")
    print("causal 'tuning amplifies compression' claim needs >=2 consistent pairs.")
PY

cp results/logs/*_v3.log "$ART/" 2>/dev/null || true
echo "=== job complete; artifacts in $JOB_OUT/artifacts ==="
