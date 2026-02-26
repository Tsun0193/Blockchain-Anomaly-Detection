#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_setting_models.sh --setting <baseline|xavier_only|graphnorm_xavier> --gpu-id <0|1|2|3> [--python <python_bin>] [--models GCN,GAT,SAGE]

Description:
  Run training + test for multiple models under ONE experiment setting.
  This script is intended to run inside one tmux session.
USAGE
}

SETTING=""
GPU_ID=""
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODELS_CSV="GCN,GAT,SAGE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setting)
      SETTING="$2"
      shift 2
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --models)
      MODELS_CSV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SETTING" || -z "$GPU_ID" ]]; then
  echo "Both --setting and --gpu-id are required." >&2
  usage
  exit 1
fi

case "$SETTING" in
  baseline|xavier_only|graphnorm_xavier) ;;
  *)
    echo "Invalid setting: $SETTING" >&2
    exit 1
    ;;
esac

if [[ ! "$GPU_ID" =~ ^[0-3]$ ]]; then
  echo "--gpu-id must be in 0..3, got: $GPU_ID" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found/executable: $PYTHON_BIN" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models parsed from --models." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bcad_${SETTING}_gpu${GPU_ID}_XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

TRAIN_CFG_SRC="config/training.yaml"
TEST_CFG_SRC="config/testing.yaml"
MODEL_CFG_SRC="config/model.yaml"
TRAIN_CFG_TMP="$TMP_DIR/training.yaml"
TEST_CFG_TMP="$TMP_DIR/testing.yaml"
MODEL_CFG_TMP="$TMP_DIR/model.yaml"

cp "$TRAIN_CFG_SRC" "$TRAIN_CFG_TMP"
cp "$TEST_CFG_SRC" "$TEST_CFG_TMP"
cp "$MODEL_CFG_SRC" "$MODEL_CFG_TMP"

"$PYTHON_BIN" - "$TRAIN_CFG_TMP" "$TEST_CFG_TMP" "$SETTING" "$GPU_ID" <<'PY'
import sys
import yaml

train_path, test_path, setting, gpu_id = sys.argv[1:5]
gpu_id = int(gpu_id)

for cfg_path in (train_path, test_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    runtime = cfg.setdefault("runtime", {})
    runtime["gpu_id"] = gpu_id
    runtime.setdefault("available_cuda_ids", [0, 1, 2, 3])
    runtime["device"] = f"cuda:{gpu_id}"
    experiment = cfg.setdefault("experiment", {})
    experiment["setting"] = setting
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
PY

RESULTS_ROOT="$("$PYTHON_BIN" - "$TRAIN_CFG_TMP" "$SETTING" <<'PY'
import sys
import yaml

cfg_path, setting = sys.argv[1:3]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
presets = (cfg.get("experiment") or {}).get("presets") or {}
selected = presets.get(setting) or {}
print(selected.get("results_root", "results"))
PY
)"

PLOTS_DIR="plots/${SETTING}"
mkdir -p "$PLOTS_DIR"

echo "============================================="
echo "Setting      : $SETTING"
echo "GPU          : cuda:$GPU_ID"
echo "Models       : ${MODELS[*]}"
echo "Results root : $RESULTS_ROOT"
echo "Plots dir    : $PLOTS_DIR"
echo "Temp config  : $TMP_DIR"
echo "============================================="

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo ">>> [${SETTING}] Training model: ${MODEL}"

  cp "$MODEL_CFG_SRC" "$MODEL_CFG_TMP"
  "$PYTHON_BIN" - "$MODEL_CFG_TMP" "$MODEL" <<'PY'
import sys
import yaml

cfg_path, model_name = sys.argv[1:3]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("model", {})
cfg["model"]["type"] = model_name
with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  "$PYTHON_BIN" scripts/train.py \
    --training-config "$TRAIN_CFG_TMP" \
    --model-config "$MODEL_CFG_TMP"

  echo ">>> [${SETTING}] Evaluating model: ${MODEL}"
  "$PYTHON_BIN" scripts/test.py \
    --testing-config "$TEST_CFG_TMP" \
    --model-config "$MODEL_CFG_TMP"

  echo ">>> [${SETTING}] Plot Optuna diagnostics: ${MODEL}"
  "$PYTHON_BIN" scripts/plot_optuna_diagnostics.py \
    --results-dir "$RESULTS_ROOT" \
    --model "$MODEL" \
    --out-dir "${PLOTS_DIR}/${MODEL}"
done

echo ""
echo ">>> [${SETTING}] Plot best-trial curves across models"
"$PYTHON_BIN" scripts/plot_best_trial_curves.py \
  --results-dir "$RESULTS_ROOT" \
  --models "${MODELS[@]}" \
  --config-name "$SETTING" \
  --out-train "${PLOTS_DIR}/best_trial_train_loss.png" \
  --out-val "${PLOTS_DIR}/best_trial_val_score.png"

echo "Done. All outputs are in:"
echo "  - ${RESULTS_ROOT}/<MODEL>/"
echo "  - ${PLOTS_DIR}/"
