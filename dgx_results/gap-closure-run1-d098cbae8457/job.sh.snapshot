#!/usr/bin/env bash
# NAME: gap-closure-run1
#
# The current DGX job. dgx_agent.sh runs this file whenever its content hash
# changes, with:
#   $JOB_OUT  = dgx_results/<name>-<hash>/   (put EVERYTHING to ship back here;
#               files > 50 MB are quarantined, so never copy activations)
#   $JOB_ID   = the content hash
#
# This job = the ordered gap-closure pipeline (PREREGISTER2.md stages 0-5),
# then copies the small verdict/CSV artifacts into $JOB_OUT.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== gap-closure run starting on $(hostname) ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Smoke-scale knob: bump LIMIT once the wiring is proven.
export LIMIT="${LIMIT:-300}"

./run_gaps.sh

# ---- ship back ONLY the small artifacts ----
echo "=== collecting artifacts ==="
mkdir -p "$JOB_OUT/artifacts"
# verdicts, CSVs, manifests, audit — everything small under geometry_v2 + determinants_v2
for d in results/geometry_v2 results/determinants_v2; do
  [ -d "$d" ] || continue
  (cd "$(dirname "$d")" && find "$(basename "$d")" -type f \
      \( -name '*.csv' -o -name '*.md' -o -name '*.json' -o -name '*.txt' \) \
      -size -20M -exec cp --parents {} "$OLDPWD/$JOB_OUT/artifacts/" \;)
done
# the run log tail for quick reading
tail -200 results/logs/run_gaps.log > "$JOB_OUT/run_gaps.log.tail" 2>/dev/null || true
for f in results/logs/matched_probe.log results/logs/audit_v2.log; do
  [ -f "$f" ] && cp "$f" "$JOB_OUT/artifacts/" || true
done

echo "=== job complete ==="
