import argparse
import ast
import json
from pathlib import Path


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


def infer_model_from_yaml(model_cfg_path: Path):
    if not model_cfg_path.exists():
        return None
    for line in model_cfg_path.read_text().splitlines():
        txt = line.strip()
        if txt.startswith("type:"):
            return txt.split(":", 1)[1].strip().strip("\"'")
    return None


def load_training_json(results_dir: Path, model: str = None, training_json: Path = None):
    if training_json is not None:
        payload = json.loads(training_json.read_text())
        model_name = payload.get("Task", training_json.stem.replace("_training_results", ""))
        return model_name, training_json, payload

    if model is None:
        model = infer_model_from_yaml(Path("config/model.yaml")) or "GCN"

    training_path = results_dir / model / f"{model}_training_results.json"
    if not training_path.exists():
        raise FileNotFoundError(f"Training result not found: {training_path}")
    payload = json.loads(training_path.read_text())
    return model, training_path, payload


def extract_trials(payload):
    trials = payload.get("Study", [])
    complete_trials = [t for t in trials if t.get("value") is not None]
    complete_trials.sort(key=lambda t: int(t.get("number", 0)))
    return complete_trials


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to render plots. Install it with: pip install matplotlib"
        ) from exc
    return plt


def save_lr_plot(plt, out_path: Path, trial_numbers, lrs):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(trial_numbers, lrs, marker="o", markersize=2.5, linewidth=0.8, color="#E5A000")
    ax.set_yscale("log")
    ax.set_title("Learning Rate per Trial", fontsize=10)
    ax.set_xlabel("Trial", fontsize=8)
    ax.set_ylabel("Learning Rate", fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_perf_plot(plt, out_path: Path, trial_numbers, values):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(trial_numbers, values, marker="o", markersize=2.5, linewidth=0.8, color="#1f77b4")
    ax.set_title("Optuna Trial Performance", fontsize=10)
    ax.set_xlabel("Trial Number", fontsize=8)
    ax.set_ylabel("Objective Value (AUC/score)", fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_best_convergence_plot(plt, out_path: Path, best_trial):
    history = parse_history(best_trial.get("user_attrs_history") or best_trial.get("history"))
    train_loss = [float(v) for v in history.get("train_loss", [])]
    val_loss = [float(v) for v in history.get("val_loss", [])] if history.get("val_loss") else None
    val_score = [float(v) for v in history.get("val_score", [])] if history.get("val_score") else None

    if not train_loss:
        raise ValueError("Best trial has no train_loss in history.")

    # Overlay val_loss if available. Fall back to val_score for backward compatibility.
    if val_loss:
        val_series = val_loss
        val_label = "Validation Loss"
    else:
        val_series = val_score
        val_label = "Validation Score"

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax1.plot(range(len(train_loss)), train_loss, color="#2f3fd0", linewidth=0.9, label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=8)
    ax1.set_ylabel("Train Loss", color="#2f3fd0", fontsize=8)
    ax1.tick_params(axis="y", labelcolor="#2f3fd0", labelsize=7)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.grid(True, alpha=0.3)

    if val_series:
        ax2 = ax1.twinx()
        ax2.plot(
            range(len(val_series)),
            val_series,
            color="#D9A520",
            linewidth=0.9,
            linestyle="--" if val_loss else "-",
            label=val_label,
        )
        ax2.set_ylabel(val_label, color="#D9A520", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#D9A520", labelsize=7)

    title = f"Best Trial {best_trial.get('number')} (Value={float(best_trial.get('value')):.4f})"
    ax1.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 3 Optuna diagnostic charts from training results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--model", type=str, default=None, help="Model name (GCN/GAT/SAGE).")
    parser.add_argument(
        "--training-json",
        type=Path,
        default=None,
        help="Direct path to *_training_results.json (overrides --results-dir/--model).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    plt = ensure_matplotlib()
    model_name, training_path, payload = load_training_json(args.results_dir, args.model, args.training_json)
    trials = extract_trials(payload)
    if not trials:
        raise ValueError(f"No completed trials in {training_path}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_numbers = [int(t.get("number", i)) for i, t in enumerate(trials)]
    values = [float(t.get("value")) for t in trials]
    lrs = []
    for t in trials:
        lr = t.get("params_lr")
        if lr is None:
            lr = (t.get("params") or {}).get("lr")
        lrs.append(float(lr) if lr is not None else float("nan"))

    best_trial = max(trials, key=lambda t: float(t.get("value")))

    out_lr = out_dir / f"{model_name}_optuna_lr_per_trial.png"
    out_perf = out_dir / f"{model_name}_optuna_trial_performance.png"
    out_conv = out_dir / f"{model_name}_optuna_best_trial_convergence.png"

    save_lr_plot(plt, out_lr, trial_numbers, lrs)
    save_perf_plot(plt, out_perf, trial_numbers, values)
    save_best_convergence_plot(plt, out_conv, best_trial)

    print(f"Training source: {training_path}")
    print(f"Saved: {out_lr}")
    print(f"Saved: {out_perf}")
    print(f"Saved: {out_conv}")


if __name__ == "__main__":
    main()
