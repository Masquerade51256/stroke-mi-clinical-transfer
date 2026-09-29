#!/usr/bin/env python
"""
Compare clinical-nearest vs. random validation-based pipeline selection,
baselines, oracle, and lesion-informed rule.

Outputs a LaTeX-ready table snippet and JSON summary.
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

SELECTION_EXPERIMENTS = {
    "Clinical-nearest validation": "experiments/XWStroke_ValBased_ClinicalNearest/results/results.json",
    "Random validation (5 seeds)": "experiments/XWStroke_ValBased_Random5/results/results.json",
}

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

    # Load selection results
    sel_accs = {}
    for name, path in SELECTION_EXPERIMENTS.items():
        with open(path) as f:
            data = json.load(f)
        sel_accs[name] = {r["target_subject_id"]: r["final"]["test_acc"] for r in data["subjects"]}

    # Oracle and best fixed
    oracle = [max(accs[p][sid] for p in accs) for sid in subject_ids]
    best_fixed = [accs["Aff_EA"][sid] for sid in subject_ids]

    # Lesion-informed rule
    meta = pd.read_csv(METADATA)
    meta["subject_id"] = meta["subject_id"].astype(int)
    meta = meta.set_index("subject_id")
    is_subcortical = {sid: (meta.loc[sid, "StrokeLocation_Category"] == "Subcortical") for sid in subject_ids}
    lesion_rule = [accs["Aff_EA"][sid] if is_subcortical[sid] else accs["LR_EA"][sid] for sid in subject_ids]

    rows = []
    print("=" * 80)
    print("Comparison of validation strategies and baselines")
    print("=" * 80)

    # Best fixed
    mean = np.mean(best_fixed) * 100
    std = np.std(best_fixed) * 100
    print(f"{'Best fixed (Aff+EA)':40s} {mean:6.2f}% ± {std:5.2f}%")
    rows.append({"method": "Best fixed (Aff+EA)", "mean": mean, "std": std})

    # Selection strategies
    for name in SELECTION_EXPERIMENTS:
        vals = [sel_accs[name][sid] for sid in subject_ids]
        mean = np.mean(vals) * 100
        std = np.std(vals) * 100
        t, p = stats.ttest_rel(vals, best_fixed)
        d = cohen_d_paired(vals, best_fixed)
        print(f"{name:40s} {mean:6.2f}% ± {std:5.2f}%  vs fixed: Δ={mean - np.mean(best_fixed)*100:+.2f}%, p={p:.4f}, d={d:.2f}")
        rows.append({"method": name, "mean": mean, "std": std, "vs_fixed_p": p, "vs_fixed_d": d})

    # Clinical-nearest vs random
    cn = [sel_accs["Clinical-nearest validation"][sid] for sid in subject_ids]
    rnd = [sel_accs["Random validation (5 seeds)"][sid] for sid in subject_ids]
    t, p = stats.ttest_rel(cn, rnd)
    d = cohen_d_paired(cn, rnd)
    print(f"\nClinical-nearest vs Random: Δ={ (np.mean(cn)-np.mean(rnd))*100:+.2f}%, t={t:.2f}, p={p:.4f}, d={d:.2f}")

    # Oracle
    mean = np.mean(oracle) * 100
    std = np.std(oracle) * 100
    print(f"\n{'Oracle (per-subject best)':40s} {mean:6.2f}% ± {std:5.2f}%")
    rows.append({"method": "Oracle", "mean": mean, "std": std})

    # Lesion-informed
    mean = np.mean(lesion_rule) * 100
    std = np.std(lesion_rule) * 100
    t, p = stats.ttest_rel(lesion_rule, best_fixed)
    d = cohen_d_paired(lesion_rule, best_fixed)
    print(f"{'Lesion-informed rule (sub->Aff+EA, other->LR+EA)':40s} {mean:6.2f}% ± {std:5.2f}%  vs fixed: Δ={mean - np.mean(best_fixed)*100:+.2f}%, p={p:.4f}, d={d:.2f}")
    rows.append({"method": "Lesion-informed rule", "mean": mean, "std": std, "vs_fixed_p": p, "vs_fixed_d": d})

    print("=" * 80)

    # Save JSON
    summary = {
        "subject_ids": subject_ids,
        "methods": rows,
        "clinical_nearest_vs_random": {
            "mean_diff_pp": float((np.mean(cn) - np.mean(rnd)) * 100),
            "t": float(t),
            "p": float(p),
            "d": float(d),
        },
    }
    with open(out_dir / "validation_strategy_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save LaTeX table snippet
    with open(out_dir / "validation_strategy_table.tex", "w") as f:
        f.write("% Add to main.tex within a table environment\n")
        for r in rows:
            line = f"{r['method']} & {r['mean']:.2f}$\\pm${r['std']:.2f}"
            if "vs_fixed_p" in r:
                sig = "*" if r["vs_fixed_p"] < 0.05 else "ns"
                line += f" & {r['mean'] - rows[0]['mean']:+.2f} & {r['vs_fixed_p']:.3f} & {r['vs_fixed_d']:.2f}"
            else:
                line += " & — & — & —"
            f.write(line + " \\\\\n")

    print(f"Saved summary to {out_dir / 'validation_strategy_comparison.json'}")
    print(f"Saved LaTeX table to {out_dir / 'validation_strategy_table.tex'}")


if __name__ == "__main__":
    main()
