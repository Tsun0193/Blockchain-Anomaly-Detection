#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/dispatch_tmux_3settings.sh <session_baseline> <session_xavier_only> <session_graphnorm_xavier> [python_bin]

Example:
  bash scripts/dispatch_tmux_3settings.sh sess0 sess1 sess2 .venv/bin/python

Description:
  Dispatch 3 jobs to 3 existing tmux sessions:
    - baseline         on cuda:0
    - xavier_only      on cuda:1
    - graphnorm_xavier on cuda:2
USAGE
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 1
fi

SESSION_BASELINE="$1"
SESSION_XAVIER_ONLY="$2"
SESSION_GRAPHNORM_XAVIER="$3"
PYTHON_BIN="${4:-.venv/bin/python}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

declare -a SESSIONS=("$SESSION_BASELINE" "$SESSION_XAVIER_ONLY" "$SESSION_GRAPHNORM_XAVIER")
declare -a SETTINGS=("baseline" "xavier_only" "graphnorm_xavier")
declare -a GPUS=(0 1 2)

for SESSION in "${SESSIONS[@]}"; do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session not found: $SESSION" >&2
    exit 1
  fi
done

TS="$(date +%Y%m%d_%H%M%S)"
for IDX in 0 1 2; do
  SESSION="${SESSIONS[$IDX]}"
  SETTING="${SETTINGS[$IDX]}"
  GPU="${GPUS[$IDX]}"
  LOG_FILE="$LOG_DIR/${SETTING}_gpu${GPU}_${TS}.log"

  CMD="cd '$REPO_ROOT' && bash scripts/run_setting_models.sh --setting '$SETTING' --gpu-id '$GPU' --python '$PYTHON_BIN' 2>&1 | tee '$LOG_FILE'"
  tmux send-keys -t "$SESSION" "$CMD" C-m

  echo "Dispatched: setting=$SETTING gpu=$GPU session=$SESSION"
  echo "  log: $LOG_FILE"
done

echo "All jobs dispatched."
