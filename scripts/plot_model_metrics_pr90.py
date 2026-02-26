import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_model_stats(results_dir: Path, model: str):
    model_dir = results_dir / model
    ti_path = model_dir / "results_TI.csv"
    td_path = model_dir / "results_TD.csv"
    tr_path = model_dir / f"{model}_training_results.json"

    if not ti_path.exists():
        raise FileNotFoundError(f"Missing file: {ti_path}")
    if not td_path.exists():
        raise FileNotFoundError(f"Missing file: {td_path}")
    if not tr_path.exists():
        raise FileNotFoundError(f"Missing file: {tr_path}")

    ti = pd.read_csv(ti_path)
    td = pd.read_csv(td_path)
    tr = json.loads(tr_path.read_text())

    auc_prc = float(tr.get("AUC_PRC", tr.get("BestValidationAUPRC", np.nan)))
    mean_auc = float(ti["AUC"].mean())
    mean_ap = float(ti["AP"].mean())
    f1_90 = float(td["F1_Score_90"].mean()) if "F1_Score_90" in td.columns else np.nan
    f1_99 = float(td["F1_Score_99"].mean()) if "F1_Score_99" in td.columns else np.nan
    f1_999 = float(td["F1_Score_99.9"].mean()) if "F1_Score_99.9" in td.columns else np.nan

    pr90 = pd.DataFrame(
        {
            "Recall_90": td["Recall_90"] if "Recall_90" in td.columns else np.nan,
            "Precision_90": td["Precision_90"] if "Precision_90" in td.columns else np.nan,
        }
    ).dropna()

    return {
        "model": model,
        "metrics": [auc_prc, mean_auc, mean_ap, f1_90, f1_99, f1_999],
        "pr90": pr90,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot model comparison metrics + Precision/Recall@90 scatter."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+", default=["GCN", "GAT", "SAGE"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/model_comparison_metrics_pr90.png"),
    )
    parser.add_argument(
        "--title-left",
        type=str,
        default="Model Comparison Across Metrics",
    )
    parser.add_argument(
        "--title-right",
        type=str,
        default="Precision vs Recall @ 90",
    )
    args = parser.parse_args()

    all_stats = [load_model_stats(args.results_dir, m) for m in args.models]
    metric_names = ["AUC_PRC", "Mean_AUC", "Mean_AP", "F1@90", "F1@99", "F1@99.9"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

    x = np.arange(len(metric_names))
    width = 0.24 if len(all_stats) <= 3 else 0.8 / max(len(all_stats), 1)

    for i, stat in enumerate(all_stats):
        offset = (i - (len(all_stats) - 1) / 2.0) * width
        ax1.bar(x + offset, stat["metrics"], width=width, label=stat["model"], alpha=0.95)

    ax1.set_title(args.title_left, fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=40, ha="right")
    ax1.set_xlabel("Metric")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(title="Model")

    markers = ["o", "x", "s", "d", "^", "v"]
    for i, stat in enumerate(all_stats):
        if stat["pr90"].empty:
            continue
        ax2.scatter(
            stat["pr90"]["Recall_90"],
            stat["pr90"]["Precision_90"],
            s=14,
            alpha=0.8,
            marker=markers[i % len(markers)],
            label=stat["model"],
        )

    ax2.set_title(args.title_right, fontsize=11)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(title="Model")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
