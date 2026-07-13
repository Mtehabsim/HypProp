#!/usr/bin/env bash
# dgx_agent.sh — git-bridge job runner. Start ONCE on the DGX; leave it running.
#
#   tmux new -s dgxagent          # (or screen / nohup)
#   ./dgx_agent.sh
#
# Protocol (single source of truth = git):
#   1. Loop: git pull --rebase --autostash every POLL_SECS.
#   2. The current job is dgx_job/job.sh. Its identity is the sha256 of its
#      CONTENT (first 12 hex chars) -> a job runs exactly once per version.
#   3. If dgx_results/<name>-<hash>/{DONE,FAILED} does not exist in the repo,
#      run the job with:  JOB_OUT=dgx_results/<name>-<hash>  bash dgx_job/job.sh
#      capturing stdout+stderr to $JOB_OUT/job.log.
#   4. While the job runs, every HEARTBEAT_SECS the agent commits+pushes the
#      log so progress is visible remotely.
#   5. On exit it writes STATUS.json + DONE (exit 0) or FAILED (else),
#      commits ONLY the $JOB_OUT directory, and pushes.
#
# The laptop side edits dgx_job/job.sh, pushes, and later `git pull`s the
# results. Jobs must write everything they want shipped back into $JOB_OUT
# (small files: CSVs, verdicts, logs — the agent refuses files > 50 MB so the
# repo never swallows activations).
#
# The agent survives job failures, network hiccups, and updates to ITSELF
# (it re-execs when dgx_agent.sh changes on a pull).

set -u
cd "$(dirname "$0")"

POLL_SECS="${POLL_SECS:-60}"
HEARTBEAT_SECS="${HEARTBEAT_SECS:-600}"
MAX_FILE_MB="${MAX_FILE_MB:-50}"
JOB_FILE="dgx_job/job.sh"
LOCK_FILE=".dgx_agent.pid"

log() { echo "[agent $(date '+%m-%d %H:%M:%S')] $*"; }

_sha() {  # portable sha256 -> first 12 chars
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -c1-12
  else shasum -a 256 "$1" | cut -c1-12; fi
}

self_hash="$(_sha "$0")"

# ---- single-instance lock ----
if [ -f "$LOCK_FILE" ] && kill -0 "$(cat "$LOCK_FILE")" 2>/dev/null; then
  log "another agent (pid $(cat "$LOCK_FILE")) is already running; exiting"
  exit 1
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

git_sync() {  # pull without dying on transient failures
  git pull --rebase --autostash 2>&1 | tail -1 || {
    git rebase --abort 2>/dev/null
    log "pull failed (network/conflict) — will retry next cycle"
    return 1
  }
}

commit_push() {  # commit_push <path> <message>  — commits ONLY <path>
  local path="$1" msg="$2"
  # size guard: quarantine anything huge so a job can't bloat the repo
  find "$path" -type f -size +"${MAX_FILE_MB}"M 2>/dev/null | while read -r f; do
    log "OVERSIZED (> ${MAX_FILE_MB}MB), not committing: $f"
    mv "$f" "${f}.oversize.local"   # *.local left untracked on the DGX
    echo "moved aside (oversize): $f" >> "$path/oversize.txt"
  done
  git add -- "$path" 2>/dev/null
  if git diff --cached --quiet -- "$path"; then return 0; fi
  git commit -q -m "$msg" -- "$path" || return 1
  git push -q 2>/dev/null || { git pull --rebase --autostash -q 2>/dev/null; git push -q; } \
    || log "push failed — changes are committed locally; will push next cycle"
}

run_job() {
  local hash="$1" out="$2" name="$3"
  mkdir -p "$out"
  cp "$JOB_FILE" "$out/job.sh.snapshot"        # exact code that ran
  {
    echo "host: $(hostname)"
    echo "started: $(date -Iseconds)"
    echo "job: $name ($hash)"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null
  } > "$out/STATUS.txt"
  commit_push "$out" "dgx: START $name ($hash)"

  log "running job $name ($hash) -> $out"
  local start_ts=$(date +%s)
  ( JOB_OUT="$out" JOB_ID="$hash" bash "$JOB_FILE" ) > "$out/job.log" 2>&1 &
  local pid=$!

  local last_hb=$start_ts
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    if (( $(date +%s) - last_hb >= HEARTBEAT_SECS )); then
      commit_push "$out" "dgx: heartbeat $name ($(( ($(date +%s) - start_ts) / 60 ))m elapsed)"
      last_hb=$(date +%s)
    fi
  done
  wait "$pid"; local code=$?
  local mins=$(( ($(date +%s) - start_ts) / 60 ))

  printf '{"job":"%s","hash":"%s","exit_code":%d,"minutes":%d,"finished":"%s"}\n' \
    "$name" "$hash" "$code" "$mins" "$(date -Iseconds)" > "$out/STATUS.json"
  if [ "$code" -eq 0 ]; then
    touch "$out/DONE"
    commit_push "$out" "dgx: DONE $name ($hash, ${mins}m)"
    log "job $name DONE (${mins}m)"
  else
    touch "$out/FAILED"
    commit_push "$out" "dgx: FAILED $name ($hash, exit $code, ${mins}m) — see job.log"
    log "job $name FAILED exit=$code (${mins}m)"
  fi
}

log "agent started (poll ${POLL_SECS}s, heartbeat ${HEARTBEAT_SECS}s); watching $JOB_FILE"
while true; do
  git_sync || { sleep "$POLL_SECS"; continue; }

  # re-exec if the agent itself was updated by the pull
  if [ "$(_sha "$0")" != "$self_hash" ]; then
    log "dgx_agent.sh changed upstream — re-executing myself"
    rm -f "$LOCK_FILE"
    exec "$0"
  fi

  if [ -f "$JOB_FILE" ]; then
    hash="$(_sha "$JOB_FILE")"
    name="$(grep -m1 '^# NAME:' "$JOB_FILE" | sed 's/^# NAME:[[:space:]]*//' | tr -cs 'A-Za-z0-9._-' '-' | sed 's/-$//')"
    name="${name:-job}"
    out="dgx_results/${name}-${hash}"
    if [ ! -e "$out/DONE" ] && [ ! -e "$out/FAILED" ]; then
      run_job "$hash" "$out" "$name"
    fi
  fi
  sleep "$POLL_SECS"
done
