#!/usr/bin/env python
"""
Clinical association analysis for validation-based pipeline selection.

Inputs:
  - experiments/<name>/results/results.json (from validation_based_pipeline_selection.py)
  - results/stratified/stratified_analysis_detailed.csv

Outputs:
  - results/comparisons/validation_selection_clinical_associations.json
  - results/figures/fig3_clinical_associations.png
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
import seaborn as sns
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


def cohen_d(x, y):
    """Compute Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-results", required=True, type=str,
                        help="Path to validation-based selection results.json")
    parser.add_argument("--metadata", default="results/stratified/stratified_analysis_detailed.csv", type=str)
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

    # Load baseline accuracies
    baseline_accs = {name: load_accs(path) for name, path in PIPELINE_EXPERIMENTS.items()}
    subject_ids = sorted(selected_pipeline.keys())

    # Load metadata
    meta = pd.read_csv(args.metadata)
    meta.columns = [c.lstrip('\ufeff') for c in meta.columns]
    if "Subject" in meta.columns and "subject_id" not in meta.columns:
        meta["subject_id"] = meta["Subject"]
    meta = meta.set_index("subject_id")

    # Ensure normalized duration category
    def norm_duration(d):
        d = str(d)
        return "Chronic(>3mo)" if "hronic" in d or ">" in d else "Acute(≤3mo)"
    meta["duration_norm"] = meta["Duration_Category"].apply(norm_duration)

    # Build dataframe
    rows = []
    for sid in subject_ids:
        row = {"subject_id": sid}
        for pipe in PIPELINE_EXPERIMENTS:
            row[f"acc_{pipe}"] = baseline_accs[pipe][sid]
        row["selected_pipeline"] = selected_pipeline[sid]
        row["selected_acc"] = selected_test_acc[sid]
        row["nihss"] = meta.loc[sid, "NIHSS"]
        row["duration"] = meta.loc[sid, "duration_norm"]
        row["location"] = meta.loc[sid, "StrokeLocation_Category"]
        row["affected_side"] = meta.loc[sid, "ParalysisSide"]
        rows.append(row)
    df = pd.DataFrame(rows)

    # Compute deltas vs LR_noEA
    for pipe in ["LR_EA", "Aff_noEA", "Aff_EA"]:
        df[f"delta_{pipe}"] = df[f"acc_{pipe}"] - df["acc_LR_noEA"]

    # EA benefit: max(LR_EA, Aff_EA) - max(LR_noEA, Aff_noEA)
    df["ea_benefit"] = df[["acc_LR_EA", "acc_Aff_EA"]].max(axis=1) - df[["acc_LR_noEA", "acc_Aff_noEA"]].max(axis=1)
    # Aff benefit: max(Aff_noEA, Aff_EA) - max(LR_noEA, LR_EA)
    df["aff_benefit"] = df[["acc_Aff_noEA", "acc_Aff_EA"]].max(axis=1) - df[["acc_LR_noEA", "acc_LR_EA"]].max(axis=1)

    # Statistical tests
    results = {}

    # 1. Correlation: NIHSS vs Aff benefit / EA benefit
    r_nihss_aff, p_nihss_aff = stats.pearsonr(df["nihss"], df["aff_benefit"])
    r_nihss_ea, p_nihss_ea = stats.pearsonr(df["nihss"], df["ea_benefit"])
    results["nihss_correlation"] = {
        "aff_benefit": {"r": float(r_nihss_aff), "p": float(p_nihss_aff)},
        "ea_benefit": {"r": float(r_nihss_ea), "p": float(p_nihss_ea)},
    }

    # 2. Duration group differences
    chronic = df[df["duration"] == "Chronic(>3mo)"]
    acute = df[df["duration"] == "Acute(≤3mo)"]
    results["duration"] = {}
    for benefit in ["aff_benefit", "ea_benefit"]:
        t, p = stats.ttest_ind(chronic[benefit], acute[benefit])
        d = cohen_d(chronic[benefit], acute[benefit])
        results["duration"][benefit] = {
            "chronic_mean": float(chronic[benefit].mean()),
            "acute_mean": float(acute[benefit].mean()),
            "t": float(t), "p": float(p), "cohen_d": float(d)
        }

    # 3. Location group differences (Subcortical vs others)
    subcortical = df[df["location"] == "Subcortical"]
    other_loc = df[df["location"] != "Subcortical"]
    results["location_subcortical"] = {}
    for benefit in ["aff_benefit", "ea_benefit"]:
        t, p = stats.ttest_ind(subcortical[benefit], other_loc[benefit])
        d = cohen_d(subcortical[benefit], other_loc[benefit])
        results["location_subcortical"][benefit] = {
            "subcortical_mean": float(subcortical[benefit].mean()),
            "other_mean": float(other_loc[benefit].mean()),
            "t": float(t), "p": float(p), "cohen_d": float(d)
        }

    # 4. Selected pipeline distribution by clinical groups
    crosstab_duration = pd.crosstab(df["selected_pipeline"], df["duration"])
    crosstab_location = pd.crosstab(df["selected_pipeline"], df["location"])
    results["selected_by_duration"] = crosstab_duration.to_dict()
    results["selected_by_location"] = crosstab_location.to_dict()

    # Save JSON
    with open(out_dir / "validation_selection_clinical_associations.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) NIHSS vs Aff benefit
    ax = axes[0, 0]
    sns.regplot(x="nihss", y="aff_benefit", data=df, ax=ax, color="steelblue",
                scatter_kws={"alpha": 0.6}, line_kws={"label": f"r={r_nihss_aff:.3f}, p={p_nihss_aff:.3f}"})
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("NIHSS")
    ax.set_ylabel("Aff+Align benefit (vs. LR)")
    ax.set_title("(a) NIHSS vs. Aff+Align Benefit")
    ax.legend()

    # (b) NIHSS vs EA benefit
    ax = axes[0, 1]
    sns.regplot(x="nihss", y="ea_benefit", data=df, ax=ax, color="darkorange",
                scatter_kws={"alpha": 0.6}, line_kws={"label": f"r={r_nihss_ea:.3f}, p={p_nihss_ea:.3f}"})
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("NIHSS")
    ax.set_ylabel("EA benefit (vs. noEA)")
    ax.set_title("(b) NIHSS vs. EA Benefit")
    ax.legend()

    # (c) Duration group boxplot
    ax = axes[1, 0]
    plot_df = pd.melt(df, id_vars=["duration"], value_vars=["aff_benefit", "ea_benefit"],
                      var_name="Benefit", value_name="Accuracy Gain")
    plot_df["Benefit"] = plot_df["Benefit"].map({"aff_benefit": "Aff+Align", "ea_benefit": "EA"})
    sns.boxplot(x="Benefit", y="Accuracy Gain", hue="duration", data=plot_df, ax=ax, palette="Set2")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("(c) Pipeline Benefit by Stroke Duration")

    # (d) Selected pipeline by location
    ax = axes[1, 1]
    crosstab_location_pct = crosstab_location.div(crosstab_location.sum(axis=0), axis=1)
    crosstab_location_pct.T.plot(kind="bar", stacked=True, ax=ax, colormap="Set3")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Lesion Location")
    ax.set_title("(d) Selected Pipeline by Lesion Location")
    ax.legend(title="Selected Pipeline", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_clinical_associations.png", dpi=300)
    plt.savefig(out_dir / "fig3_clinical_associations.pdf", dpi=300)
    plt.close()

    # Print summary
    print("=== Clinical association summary ===")
    print(f"NIHSS vs Aff benefit: r={r_nihss_aff:.3f}, p={p_nihss_aff:.4f}")
    print(f"NIHSS vs EA benefit:  r={r_nihss_ea:.3f}, p={p_nihss_ea:.4f}")
    print(f"Duration (chronic vs acute) Aff benefit: d={results['duration']['aff_benefit']['cohen_d']:.3f}, "
          f"p={results['duration']['aff_benefit']['p']:.4f}")
    print(f"Subcortical vs other Aff benefit: d={results['location_subcortical']['aff_benefit']['cohen_d']:.3f}, "
          f"p={results['location_subcortical']['aff_benefit']['p']:.4f}")
    print(f"Figure saved to: {out_dir / 'fig3_clinical_associations.png'}")


if __name__ == "__main__":
    main()
