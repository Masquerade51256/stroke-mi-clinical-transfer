#!/usr/bin/env python
"""
Evaluate simple lesion-informed pipeline assignment rules.

For each subject, assign a pipeline based on lesion location (subcortical vs. other)
and compare against fixed baselines, oracle, and clinical-nearest validation selection.

Inputs:
    - experiments/XWStroke_Full50_EEGNet_*/results/results.json (4 pipelines)
    - experiments/XWStroke_ValBased_ClinicalNearest/results/results.json
    - results/stratified/stratified_analysis_detailed.csv

Outputs:
    - results/comparisons/lesion_informed_rule_summary.json
    - results/comparisons/lesion_informed_rule_summary.txt
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIPELINE_EXPERIMENTS = {
    "LR_noEA": "experiments/XWStroke_Full50_EEGNet_LR_noEA_noAug/results/results.json",
    "LR_EA": "experiments/XWStroke_Full50_EEGNet_LR_EA/results/results.json",
    "Aff_noEA": "experiments/XWStroke_Full50_EEGNet_AffAlign_noEA_noAug/results/results.json",
    "Aff_EA": "experiments/XWStroke_Full50_EEGNet_EA_AffAlign_NoMP/results/results.json",
}

CLINICAL_NEAREST = "experiments/XWStroke_ValBased_ClinicalNearest/results/results.json"
METADATA = "results/stratified/stratified_analysis_detailed.csv"


def load_accs(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return {r["test_subject_id"]: r["test_acc"] for r in data["subjects"]}


def cohen_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd != 0 else 0.0


def main():
    out_dir = Path("results/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline accuracies
    accs = {name: load_accs(path) for name, path in PIPELINE_EXPERIMENTS.items()}
    subject_ids = sorted(next(iter(accs.values())).keys())

    # Load clinical metadata
    meta = pd.read_csv(METADATA)
    meta["subject_id"] = meta["subject_id"].astype(int)
    meta = meta.set_index("subject_id")

    # Load clinical-nearest validation selection
    with open(CLINICAL_NEAREST) as f:
        sel = json.load(f)
    selected_acc = {r["target_subject_id"]: r["final"]["test_acc"] for r in sel["subjects"]}

    # Define lesion-informed rules
    is_subcortical = {sid: (meta.loc[sid, "StrokeLocation_Category"] == "Subcortical") for sid in subject_ids}

    rules = {
        "Subcortical -> Aff+EA, Other -> LR_noEA": lambda sid: "Aff_EA" if is_subcortical[sid] else "LR_noEA",
        "Subcortical -> Aff+EA, Other -> LR_EA": lambda sid: "Aff_EA" if is_subcortical[sid] else "LR_EA",
        "Subcortical -> Aff+EA, Other -> Aff+EA (everyone Aff+EA)": lambda sid: "Aff_EA",
        "Subcortical -> LR_noEA, Other -> Aff+EA (reverse)": lambda sid: "LR_noEA" if is_subcortical[sid] else "Aff_EA",
    }

    # Compute oracle and best fixed
    oracle_acc = [max(accs[p][sid] for p in accs) for sid in subject_ids]
    best_fixed = [accs["Aff_EA"][sid] for sid in subject_ids]
    clinical_nearest = [selected_acc[sid] for sid in subject_ids]

    rows = []
    print("=" * 70)
    print("Lesion-informed pipeline assignment rules")
    print("=" * 70)

    # Best fixed pipeline
    mean = np.mean(best_fixed) * 100
    std = np.std(best_fixed) * 100
    print(f"{'Best fixed (Aff+EA)':45s} {mean:6.2f}% ± {std:5.2f}%")
    rows.append({"method": "Best fixed (Aff+EA)", "mean": mean, "std": std})

    # Clinical-nearest validation selection
    mean = np.mean(clinical_nearest) * 100
    std = np.std(clinical_nearest) * 100
    t, p = stats.ttest_rel(clinical_nearest, best_fixed)
    d = cohen_d_paired(clinical_nearest, best_fixed)
    print(f"{'Clinical-nearest validation':45s} {mean:6.2f}% ± {std:5.2f}%  vs fixed: Δ={mean - np.mean(best_fixed)*100:+.2f}%, p={p:.4f}, d={d:.2f}")
    rows.append({"method": "Clinical-nearest validation", "mean": mean, "std": std, "vs_fixed_p": p, "vs_fixed_d": d})

    # Oracle
    mean = np.mean(oracle_acc) * 100
    std = np.std(oracle_acc) * 100
    print(f"{'Oracle (per-subject best)':45s} {mean:6.2f}% ± {std:5.2f}%")
    rows.append({"method": "Oracle", "mean": mean, "std": std})

    # Lesion-informed rules
    for rule_name, rule_fn in rules.items():
        rule_accs = [accs[rule_fn(sid)][sid] for sid in subject_ids]
        mean = np.mean(rule_accs) * 100
        std = np.std(rule_accs) * 100
        t, p = stats.ttest_rel(rule_accs, best_fixed)
        d = cohen_d_paired(rule_accs, best_fixed)
        print(f"{rule_name:45s} {mean:6.2f}% ± {std:5.2f}%  vs fixed: Δ={mean - np.mean(best_fixed)*100:+.2f}%, p={p:.4f}, d={d:.2f}")
        rows.append({"method": rule_name, "mean": mean, "std": std, "vs_fixed_p": p, "vs_fixed_d": d})

    # Save summary
    summary = {
        "subject_ids": subject_ids,
        "rules": rows,
        "n_subcortical": int(sum(is_subcortical.values())),
        "n_other": int(len(subject_ids) - sum(is_subcortical.values())),
    }
    with open(out_dir / "lesion_informed_rule_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "lesion_informed_rule_summary.txt", "w") as f:
        f.write("Lesion-informed pipeline assignment rules\n")
        f.write("=" * 70 + "\n")
        for r in rows:
            line = f"{r['method']:45s} {r['mean']:6.2f}% ± {r['std']:5.2f}%"
            if "vs_fixed_p" in r:
                line += f"  vs fixed: p={r['vs_fixed_p']:.4f}, d={r['vs_fixed_d']:.2f}"
            f.write(line + "\n")

    print("=" * 70)
    print(f"Saved summary to {out_dir / 'lesion_informed_rule_summary.json'}")


if __name__ == "__main__":
    main()
