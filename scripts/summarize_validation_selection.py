#!/usr/bin/env python
"""
Summarize validation-based pipeline selection results.

Inputs:
  - experiments/<name>/results/results.json (or incremental_selection.json)
  - Baseline experiment result JSONs for the four EEGNet pipelines

Outputs:
  - results/comparisons/validation_selection_summary.json
  - results/comparisons/validation_selection_summary.txt (human-readable)
  - results/figures/fig4_selection_summary.png
  - A LaTeX table snippet printed to stdout / saved to file
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIPELINE_EXPERIMENTS = {
    "LR_noEA": "experiments/XWStroke_Full50_EEGNet_LR_noEA_noAug/results/results.json",
    "LR_EA": "experiments/XWStroke_Full50_EEGNet_LR_EA/results/results.json",
    "Aff_noEA": "experiments/XWStroke_Full50_EEGNet_AffAlign_noEA_noAug/results/results.json",
    "Aff_EA": "experiments/XWStroke_Full50_EEGNet_EA_AffAlign_NoMP/results/results.json",
}


def load_accs(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return {r["test_subject_id"]: r["test_acc"] for r in data["subjects"]}


def cohen_d_paired(x, y):
    """Cohen's d for paired samples using the standard deviation of differences."""
    diff = np.array(x) - np.array(y)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd != 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-results", required=True, type=str,
                        help="Path to validation-based selection results JSON")
    parser.add_argument("--out-dir", default="results/comparisons", type=str)
    parser.add_argument("--fig-dir", default="results/figures", type=str)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load selection results
    with open(args.selection_results, "r") as f:
        sel = json.load(f)

    selected_pipeline = {r["target_subject_id"]: r["selected_pipeline"] for r in sel["subjects"]}
    selected_test_acc = {r["target_subject_id"]: r["final"]["test_acc"] for r in sel["subjects"]}
    subject_ids = sorted(selected_pipeline.keys())

    # Load baseline accuracies
    baseline_accs = {name: load_accs(path) for name, path in PIPELINE_EXPERIMENTS.items()}

    # Build dataframe
    rows = []
    for sid in subject_ids:
        row = {"subject_id": sid}
        for pipe in PIPELINE_EXPERIMENTS:
            row[f"acc_{pipe}"] = baseline_accs[pipe][sid]
        row["selected_pipeline"] = selected_pipeline[sid]
        row["selected_acc"] = selected_test_acc[sid]
        row["oracle_acc"] = max(row[f"acc_{pipe}"] for pipe in PIPELINE_EXPERIMENTS)
        # Best overall fixed pipeline = Aff_EA
        row["best_fixed_pipeline_acc"] = row["acc_Aff_EA"]
        row["gain_vs_oracle"] = row["selected_acc"] - row["oracle_acc"]
        row["gain_vs_best_fixed_pipeline"] = row["selected_acc"] - row["best_fixed_pipeline_acc"]
        rows.append(row)
    df = pd.DataFrame(rows)

    # Summary statistics
    summary = {}
    for pipe in PIPELINE_EXPERIMENTS:
        vals = df[f"acc_{pipe}"].values
        summary[pipe] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)), "n": len(vals)}

    selected_vals = df["selected_acc"].values
    summary["selected"] = {
        "mean": float(selected_vals.mean()),
        "std": float(selected_vals.std(ddof=1)),
        "n": len(selected_vals),
    }
    summary["oracle"] = {
        "mean": float(df["oracle_acc"].mean()),
        "std": float(df["oracle_acc"].std(ddof=1)),
        "n": len(df),
    }

    # Paired tests: selected vs oracle, vs best fixed pipeline, and vs each pipeline
    tests = {}
    for label, col in [("oracle", "oracle_acc"),
                       ("best_fixed_pipeline", "best_fixed_pipeline_acc")] + \
                      [(p, f"acc_{p}") for p in PIPELINE_EXPERIMENTS]:
        t, p = stats.ttest_rel(selected_vals, df[col].values)
        d = cohen_d_paired(selected_vals, df[col].values)
        tests[label] = {"t": float(t), "p": float(p), "cohen_d": float(d),
                        "mean_diff": float(np.mean(selected_vals - df[col].values))}
    summary["paired_tests"] = tests

    # Selection frequency
    freq = df["selected_pipeline"].value_counts().to_dict()
    summary["selection_frequency"] = {k: int(v) for k, v in freq.items()}

    # Save JSON
    with open(out_dir / "validation_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable text
    lines = []
    lines.append("=== Validation-based pipeline selection summary ===")
    lines.append(f"Subjects: {len(df)}")
    lines.append("")
    lines.append("Mean accuracy ± SD:")
    for pipe in PIPELINE_EXPERIMENTS:
        lines.append(f"  {pipe:10s}: {summary[pipe]['mean']*100:5.2f}% ± {summary[pipe]['std']*100:4.2f}%")
    lines.append(f"  {'Selected':10s}: {summary['selected']['mean']*100:5.2f}% ± {summary['selected']['std']*100:4.2f}%")
    lines.append(f"  {'Oracle':10s}: {summary['oracle']['mean']*100:5.2f}% ± {summary['oracle']['std']*100:4.2f}%")
    lines.append(f"  {'Aff_EA':10s}: {summary['Aff_EA']['mean']*100:5.2f}% ± {summary['Aff_EA']['std']*100:4.2f}% (best fixed pipeline)")
    lines.append("")
    lines.append("Paired comparison with validation-based selection:")
    for label, res in tests.items():
        name_map = {"best_fixed_pipeline": "best fixed (Aff_EA)", "oracle": "oracle"}
        display = name_map.get(label, label)
        lines.append(f"  vs {display:20s}: Δ={res['mean_diff']*100:+.2f}%, t={res['t']:+.2f}, p={res['p']:.4f}, d={res['cohen_d']:.2f}")
    lines.append("")
    lines.append("Selected pipeline frequency:")
    for pipe, count in sorted(freq.items()):
        lines.append(f"  {pipe:10s}: {count} ({count/len(df)*100:.1f}%)")

    text = "\n".join(lines)
    with open(out_dir / "validation_selection_summary.txt", "w") as f:
        f.write(text + "\n")
    print(text)

    # LaTeX table snippet (single column, to be adapted to IEEE 	able* if needed)
    latex = r"""
\begin{table}[t]
\caption{Validation-based pipeline selection vs. fixed EEGNet pipelines (XWStroke, $N$=""" + f"{len(df)}" + r""").}
\label{tab:selection}
\centering
\begin{tabular}{lc}
\toprule
Method & Accuracy (\%) \\
\midrule
"""
    for pipe in PIPELINE_EXPERIMENTS:
        marker = "*" if pipe == "Aff_EA" else ""
        latex += f"{pipe}{marker} & {summary[pipe]['mean']*100:.2f} $\\pm$ {summary[pipe]['std']*100:.2f} \\\\\n"
    latex += f"\\midrule\n"
    latex += f"Validation-based selection & {summary['selected']['mean']*100:.2f} $\\pm$ {summary['selected']['std']*100:.2f} \\\\\n"
    latex += f"Oracle upper bound & {summary['oracle']['mean']*100:.2f} $\\pm$ {summary['oracle']['std']*100:.2f} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(out_dir / "validation_selection_table.tex", "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {out_dir / 'validation_selection_table.tex'}")

    # Figure: per-subject selected accuracy and gain vs best fixed
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    x = np.arange(len(df))
    palette = {"LR_noEA": "#1f77b4", "LR_EA": "#ff7f0e",
               "Aff_noEA": "#2ca02c", "Aff_EA": "#d62728"}

    # (a) accuracies
    ax = axes[0]
    for pipe in PIPELINE_EXPERIMENTS:
        ax.scatter(x, df[f"acc_{pipe}"] * 100, c=palette[pipe], s=15, alpha=0.5, label=pipe)
    ax.scatter(x, df["selected_acc"] * 100, c="black", s=30, marker="x", label="Selected", zorder=5)
    ax.plot(x, df["oracle_acc"] * 100, "k--", linewidth=1, alpha=0.4, label="Oracle")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-subject accuracy: selected vs. fixed pipelines")
    ax.legend(ncol=3, loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) gain vs best fixed
    ax = axes[1]
    colors = [palette[p] for p in df["selected_pipeline"]]
    ax.bar(x, df["gain_vs_best_fixed_pipeline"] * 100, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Subject index")
    ax.set_ylabel("Gain vs. Aff+EA (%)")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = fig_dir / "fig4_selection_summary.png"
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_dir / "fig4_selection_summary.pdf", dpi=300)
    plt.close()
    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
