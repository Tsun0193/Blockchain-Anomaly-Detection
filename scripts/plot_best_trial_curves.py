import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_history(entry):
    if entry is None:
        return {}
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        try:
            return json.loads(entry)
        except Exception:
            try:
                return ast.literal_eval(entry)
            except Exception:
                return {}
    return {}


def load_best_trial(training_json_path: Path):
    payload = json.loads(training_json_path.read_text())
    trials = payload.get("Study", [])
    complete_trials = [t for t in trials if t.get("value") is not None]
    if not complete_trials:
        raise ValueError(f"No completed trials found in {training_json_path}")
    best_trial = max(complete_trials, key=lambda t: float(t["value"]))
    history = parse_history(best_trial.get("user_attrs_history") or best_trial.get("history"))
    return best_trial, history


def safe_series(history, *keys):
    for key in keys:
        values = history.get(key)
        if values is not None and len(values) > 0:
            return np.asarray([float(v) for v in values], dtype=float)
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot training/validation curves from the BEST Optuna trial (no mean/std)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per-model result folders (e.g., results/GCN/GCN_training_results.json).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["GCN", "GAT", "SAGE"],
        help="Model names to include.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="baseline",
        help="Label used in figure title.",
    )
    parser.add_argument(
        "--out-train",
        type=Path,
        default=Path("plots/best_trial_train_loss.png"),
        help="Output image path for train loss figure.",
    )
    parser.add_argument(
        "--out-val",
        type=Path,
        default=Path("plots/best_trial_val_score.png"),
        help="Output image path for validation score figure.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_val.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    plot_data = {}
    missing_val_loss_models = []
    for model_name in args.models:
        training_json_path = args.results_dir / model_name / f"{model_name}_training_results.json"
        if not training_json_path.exists():
            missing.append(str(training_json_path))
            continue

        best_trial, history = load_best_trial(training_json_path)
        train_loss = safe_series(history, "train_loss")
        val_score = safe_series(history, "val_score", "val_auc", "val_loss")
        val_loss_overlay = safe_series(history, "val_loss")
        if val_loss_overlay is None:
            missing_val_loss_models.append(model_name)
        plot_data[model_name] = {
            "train_loss": train_loss,
            "val_score": val_score,
            "val_loss_overlay": val_loss_overlay,
        }

        print(
            f"{model_name}: best_trial={best_trial.get('number')} "
            f"value={float(best_trial.get('value')):.6f}"
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    fig_train, ax_train = plt.subplots(figsize=(8, 6), dpi=120)
    ax_train_val = ax_train.twinx()
    train_handles = []
    val_handles = []
    for model_name in args.models:
        model_blob = plot_data.get(model_name, {})
        train_series = model_blob.get("train_loss")
        val_series = model_blob.get("val_loss_overlay")

        if train_series is None:
            continue

        line_train = ax_train.plot(
            np.arange(train_series.size),
            train_series,
            label=f"{model_name} train_loss",
            linewidth=2,
        )[0]
        train_handles.append(line_train)

        if val_series is not None:
            line_val = ax_train_val.plot(
                np.arange(val_series.size),
                val_series,
                linestyle="--",
                linewidth=1.8,
                color=line_train.get_color(),
                label=f"{model_name} val_loss",
            )[0]
            val_handles.append(line_val)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train loss")
    ax_train_val.set_ylabel("Validation loss (dashed)")
    all_handles = train_handles + val_handles
    if all_handles:
        ax_train.legend(all_handles, [h.get_label() for h in all_handles], fontsize=9)
    fig_train.tight_layout()
    fig_train.savefig(args.out_train)
    plt.close(fig_train)

    fig_val, ax_val = plt.subplots(figsize=(8, 6), dpi=120)
    for model_name in args.models:
        series = plot_data.get(model_name, {}).get("val_score")
        if series is not None:
            ax_val.plot(np.arange(series.size), series, label=model_name, linewidth=2)
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation score")
    ax_val.legend()
    fig_val.tight_layout()
    fig_val.savefig(args.out_val)
    plt.close(fig_val)

    print(f"Saved figure: {args.out_train}")
    print(f"Saved figure: {args.out_val}")
    if missing_val_loss_models:
        print("Validation loss is missing for models below; dashed val_loss not drawn:")
        for model_name in missing_val_loss_models:
            print(f"  - {model_name}")
        print("Re-run training after val_loss logging is enabled to get dashed val_loss.")
    if missing:
        print("Missing files:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
