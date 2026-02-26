# Blockchain Anomaly Detection (GCN/GAT/GraphSAGE)

Leakage-free GNN training pipeline for Elliptic Bitcoin AML data with Optuna tuning.

Current experiment settings:
- `baseline`
- `xavier_only`
- `graphnorm_xavier`

Model family:
- `GCN`
- `GAT`
- `SAGE` (GraphSAGE)

## 1) Environment setup

This repo includes `pyproject.toml` and `uv.lock`.

```bash
# from repo root
uv sync
```

If you do not use `uv`, install dependencies with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install kagglehub matplotlib optuna pandas pyyaml scikit-learn seaborn torch torch-geometric tqdm
```

## 2) Prepare dataset (Elliptic)

Expected folder:
- `datasets/elliptic/elliptic_txs_features.csv`
- `datasets/elliptic/elliptic_txs_edgelist.csv`
- `datasets/elliptic/elliptic_txs_classes.csv`

Download via `kagglehub`:

```bash
.venv/bin/python - <<'PY'
import kagglehub
import shutil
from pathlib import Path

src = Path(kagglehub.dataset_download("ellipticco/elliptic-data-set"))
dst = Path("datasets/elliptic")
dst.mkdir(parents=True, exist_ok=True)

for name in ["elliptic_txs_features.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_classes.csv"]:
    shutil.copy2(src / name, dst / name)

print("Downloaded to:", dst.resolve())
PY
```

## 3) Configuration overview

Main config files:
- `config/training.yaml`
- `config/testing.yaml`
- `config/model.yaml`

Important knobs in `training.yaml` / `testing.yaml`:
- `runtime.gpu_id`: choose GPU index (`0..3`)
- `experiment.setting`: one of `baseline`, `xavier_only`, `graphnorm_xavier`
- `training.n_trials`: Optuna trial count
- `optimizer.weight_decay`: fixed default currently `5e-4`

`model.yaml` controls which model is run (`GCN`/`GAT`/`SAGE`).

## 4) Reproduce experiments

### Option A: 3 tmux sessions (recommended for parallel run)

In three different tmux sessions:

```bash
# tmux session 1 (cuda:0)
bash scripts/run_setting_models.sh --setting baseline --gpu-id 0 --python .venv/bin/python
```

```bash
# tmux session 2 (cuda:1)
bash scripts/run_setting_models.sh --setting xavier_only --gpu-id 1 --python .venv/bin/python
```

```bash
# tmux session 3 (cuda:2)
bash scripts/run_setting_models.sh --setting graphnorm_xavier --gpu-id 2 --python .venv/bin/python
```

Each command will:
1. Train + tune `GCN`, `GAT`, `SAGE` with Optuna.
2. Save best checkpoint per model.
3. Run final test evaluation.
4. Generate per-setting training/diagnostic plots.

### Option B: dispatch to existing tmux sessions automatically

```bash
# Example: three existing sessions named sess0 sess1 sess2
bash scripts/dispatch_tmux_3settings.sh sess0 sess1 sess2 .venv/bin/python
```

Logs are written to `logs/`.

## 5) Output structure

Results by setting:
- `results/<MODEL>/...` for `baseline`
- `results_no_graphnorm/<MODEL>/...` for `xavier_only`
- `results_with_graphnorm/<MODEL>/...` for `graphnorm_xavier`

Per model files:
- `<MODEL>_training_results.json`
- `results_TI.csv` (AUC, AP)
- `results_TD.csv` (Precision/Recall/F1 at percentiles)

Checkpoints by setting:
- `checkpoints/<MODEL>/<model>_best.pt`
- `checkpoints_no_graphnorm/<MODEL>/<model>_best.pt`
- `checkpoints_with_graphnorm/<MODEL>/<model>_best.pt`

Plots by setting:
- `plots/baseline/`
- `plots/xavier_only/`
- `plots/graphnorm_xavier/`

## 6) Generate plots and summary table after training

### Rebuild all main plots

```bash
for setting in baseline xavier_only graphnorm_xavier; do
  case "$setting" in
    baseline) res_dir="results" ;;
    xavier_only) res_dir="results_no_graphnorm" ;;
    graphnorm_xavier) res_dir="results_with_graphnorm" ;;
  esac

  .venv/bin/python scripts/plot_best_trial_curves.py \
    --results-dir "$res_dir" \
    --models GCN GAT SAGE \
    --config-name "$setting" \
    --out-train "plots/$setting/best_trial_train_loss.png" \
    --out-val "plots/$setting/best_trial_val_score.png"

  for model in GCN GAT SAGE; do
    .venv/bin/python scripts/plot_optuna_diagnostics.py \
      --results-dir "$res_dir" \
      --model "$model" \
      --out-dir "plots/$setting/$model"
  done

  .venv/bin/python scripts/plot_model_metrics_pr90.py \
    --results-dir "$res_dir" \
    --models GCN GAT SAGE \
    --out "plots/model_comparison_metrics_pr90_${setting}.png"
done
```

### Build final comparison table across 3 settings

```bash
.venv/bin/python scripts/summarize_best_results_table.py \
  --root-dir . \
  --out-csv plots/best_results_summary.csv \
  --out-md plots/best_results_summary.md \
  --out-png plots/best_results_summary.png
```

## 7) Methodology notes (leakage-free)

- Validation and test masks are enforced disjoint.
- Optuna objective uses validation AUPRC only.
- Test metric is not used for trial selection.
- Held-out test is reported after selecting best validation hyperparameters.


### Data License
This project uses the Elliptic dataset from Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set.
We do not redistribute the raw dataset. Users must download the data from Kaggle and comply with the dataset’s original license and Kaggle terms.
