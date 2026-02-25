import argparse
import csv
from pathlib import Path


CONFIG_SPECS = [
    ("Baseline (AUC / AUPRC)", "results"),
    ("Xavier Only", "results_no_graphnorm"),
    ("GraphNorm + Xavier", "results_with_graphnorm"),
]

MODEL_SPECS = [
    ("GCN", "GCN"),
    ("GAT", "GAT"),
    ("GraphSAGE", "SAGE"),
]


def compute_means_from_results_ti(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        auc_sum = 0.0
        ap_sum = 0.0
        n = 0
        for row in reader:
            if "AUC" not in row or "AP" not in row:
                continue
            auc_sum += float(row["AUC"])
            ap_sum += float(row["AP"])
            n += 1
    if n == 0:
        return None
    return {
        "auc_mean": auc_sum / n,
        "ap_mean": ap_sum / n,
        "n_rows": n,
    }


def fmt_pair(auc, ap):
    return f"{auc:.4f} / {ap:.4f}"


def build_summary(root_dir: Path):
    rows = []
    missing = []
    for model_display, model_dir in MODEL_SPECS:
        values = []
        for cfg_display, cfg_rel_dir in CONFIG_SPECS:
            p = root_dir / cfg_rel_dir / model_dir / "results_TI.csv"
            if not p.exists():
                values.append(None)
                missing.append(str(p))
                continue
            stats = compute_means_from_results_ti(p)
            values.append(stats)
        rows.append((model_display, values))
    return rows, missing


def write_csv(summary_rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Baseline_AUC",
                "Baseline_AUPRC",
                "XavierOnly_AUC",
                "XavierOnly_AUPRC",
                "GraphNormXavier_AUC",
                "GraphNormXavier_AUPRC",
                "Best_Config_By_AUPRC",
            ]
        )
        for model_display, values in summary_rows:
            best_idx = -1
            best_ap = float("-inf")
            for i, stats in enumerate(values):
                if stats is None:
                    continue
                if stats["ap_mean"] > best_ap:
                    best_ap = stats["ap_mean"]
                    best_idx = i

            row = [model_display]
            for stats in values:
                if stats is None:
                    row.extend(["", ""])
                else:
                    row.extend([f"{stats['auc_mean']:.6f}", f"{stats['ap_mean']:.6f}"])
            row.append(CONFIG_SPECS[best_idx][0] if best_idx >= 0 else "")
            writer.writerow(row)


def write_markdown(summary_rows, out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Model"] + [cfg for cfg, _ in CONFIG_SPECS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    for model_display, values in summary_rows:
        best_idx = -1
        best_ap = float("-inf")
        for i, stats in enumerate(values):
            if stats is None:
                continue
            if stats["ap_mean"] > best_ap:
                best_ap = stats["ap_mean"]
                best_idx = i

        cells = [model_display]
        for i, stats in enumerate(values):
            if stats is None:
                cells.append("N/A")
                continue
            cell = fmt_pair(stats["auc_mean"], stats["ap_mean"])
            if i == best_idx:
                cell = f"**{cell}**"
            cells.append(cell)

        lines.append("| " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n")


def render_png_table(summary_rows, out_png: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for --out-png. Install with: pip install matplotlib") from exc

    headers = ["Model"] + [cfg for cfg, _ in CONFIG_SPECS]
    table_rows = []
    best_positions = set()

    for r_idx, (model_display, values) in enumerate(summary_rows):
        best_idx = -1
        best_ap = float("-inf")
        for i, stats in enumerate(values):
            if stats is None:
                continue
            if stats["ap_mean"] > best_ap:
                best_ap = stats["ap_mean"]
                best_idx = i

        row = [model_display]
        for i, stats in enumerate(values):
            row.append("N/A" if stats is None else fmt_pair(stats["auc_mean"], stats["ap_mean"]))
            if i == best_idx and stats is not None:
                best_positions.add((r_idx + 1, i + 1))  # +1 because row 0 is header
        table_rows.append(row)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 2.4), dpi=160)
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)

    # Header bold
    for c in range(len(headers)):
        tbl[(0, c)].get_text().set_fontweight("bold")

    # Highlight best cell per row
    for pos in best_positions:
        if pos in tbl.get_celld():
            tbl[pos].get_text().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize mean AUC/AUPRC across configs and mark best config per model."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Root folder containing results/, results_no_graphnorm/, results_with_graphnorm/.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("plots/best_results_summary.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("plots/best_results_summary.md"))
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("plots/best_results_summary.png"),
        help="Output PNG table. Set empty string to skip image render.",
    )
    args = parser.parse_args()

    summary_rows, missing = build_summary(args.root_dir)
    write_csv(summary_rows, args.out_csv)
    write_markdown(summary_rows, args.out_md)

    if str(args.out_png).strip():
        render_png_table(summary_rows, args.out_png)
        print(f"Saved: {args.out_png}")

    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_md}")

    if missing:
        print("Missing results_TI.csv files:")
        for p in sorted(set(missing)):
            print(f"  - {p}")

    # Also print markdown-like table in terminal.
    print("\nSummary (best by AUPRC per row):")
    print((args.out_md).read_text())


if __name__ == "__main__":
    main()
