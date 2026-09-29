#!/usr/bin/env python
"""
Aggregate alternative-backbone 4-pipeline LOSO results and compare against
the EEGNet reference (architecture robustness check).

Scans experiments/*/config.yaml for runs named
  XWStroke_Full50_<Model>_{LR_noEA,LR_EA,AffAlign_noEA,EA_AffAlign}
(default <Model>=ShallowConvNet) and pairs them with the existing EEGNet
4-pipeline runs.

Outputs (results/comparisons/):
  - altbackbone_4pipeline_per_subject.csv
  - altbackbone_summary.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy import stats

EXP_GLOB = "experiments/*/config.yaml"
EEGNET_RUNS = {
    "LR_noEA": "experiments/XWStroke_Full50_EEGNet_LR_noEA_noAug",
    "LR_EA": "experiments/XWStroke_Full50_EEGNet_LR_EA",
    "AffAlign_noEA": "experiments/XWStroke_Full50_EEGNet_AffAlign_noEA_noAug",
    "EA_AffAlign": "experiments/XWStroke_Full50_EEGNet_EA_AffAlign_NoMP",
}
OUT_DIR = "results/comparisons"


def load_per_subject(exp_dir):
    results_path = os.path.join(exp_dir, "results", "results.json")
    if not os.path.isfile(results_path):
        return None
    with open(results_path) as f:
        res = json.load(f)
    subjects = res["subjects"]
    if len(subjects) != 50:
        print(f"WARNING: {exp_dir} has {len(subjects)}/50 subjects, skipping")
        return None
    return {s["test_subject_id"]: s["test_acc"] * 100 for s in subjects}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ShallowConvNet")
    args = parser.parse_args()
    model = args.model

    run_re = re.compile(
        rf"XWStroke_Full50_{re.escape(model)}_(?P<pipe>LR_noEA|LR_EA|AffAlign_noEA|EA_AffAlign)$"
    )

    # Discover alternative-backbone runs
    alt_runs = {}
    for cfg_path in sorted(glob.glob(EXP_GLOB)):
        exp_dir = os.path.dirname(cfg_path)
        try:
            with open(cfg_path) as f:
                name = (yaml.safe_load(f).get("experiment") or {}).get("name", "")
        except Exception:
            continue
        m = run_re.fullmatch(name)
        if m:
            alt_runs[m.group("pipe")] = exp_dir

    # Historical runs whose config names predate the naming convention
    # (ADFCNN EA pipelines already match the paper protocol; reuse as-is).
    FIXED_RUNS = {
        "ADFCNN": {
            "LR_EA": "experiments/XWStroke_Full50_ADFCNN_EA_LR_Run",
            "EA_AffAlign": "experiments/XWStroke_Full50_ADFCNN_EA_AffAlign_Run",
        },
    }
    for pipe, exp_dir in FIXED_RUNS.get(model, {}).items():
        alt_runs.setdefault(pipe, exp_dir)

    rows, summary = [], {"model": model, "pipelines": {}}
    for pipe in ["LR_noEA", "LR_EA", "AffAlign_noEA", "EA_AffAlign"]:
        alt_dir = alt_runs.get(pipe)
        ref_dir = EEGNET_RUNS[pipe]
        if not alt_dir:
            print(f"MISSING: no {model} run for {pipe}")
            continue
        alt = load_per_subject(alt_dir)
        ref = load_per_subject(ref_dir)
        if alt is None or ref is None:
            continue

        sids = sorted(alt)
        alt_accs = np.array([alt[s] for s in sids])
        ref_accs = np.array([ref[s] for s in sids])
        t, p = stats.ttest_rel(alt_accs, ref_accs)

        for s in sids:
            rows.append({"model": model, "pipeline": pipe, "test_subject_id": s,
                         "acc": alt[s], "eegnet_acc": ref[s]})

        summary["pipelines"][pipe] = {
            f"{model}_mean": float(alt_accs.mean()),
            f"{model}_std": float(alt_accs.std()),
            "eegnet_mean": float(ref_accs.mean()),
            "eegnet_std": float(ref_accs.std()),
            "paired_t": float(t),
            "paired_p": float(p),
            "alt_dir": alt_dir,
            "eegnet_dir": ref_dir,
        }
        print(f"{pipe:15s} {model}: {alt_accs.mean():.2f}% ± {alt_accs.std():.2f}%   "
              f"EEGNet: {ref_accs.mean():.2f}% ± {ref_accs.std():.2f}%   "
              f"paired t={t:+.2f}, p={p:.4f}")

    # Best fixed pipeline per backbone
    if summary["pipelines"]:
        best_alt = max(summary["pipelines"].items(),
                       key=lambda kv: kv[1][f"{model}_mean"])
        best_ref = max(summary["pipelines"].items(),
                       key=lambda kv: kv[1]["eegnet_mean"])
        summary["best_pipeline_alt"] = best_alt[0]
        summary["best_pipeline_eegnet"] = best_ref[0]
        print(f"\nBest fixed pipeline ({model}): {best_alt[0]} "
              f"({best_alt[1][f'{model}_mean']:.2f}%)")
        print(f"Best fixed pipeline (EEGNet): {best_ref[0]} "
              f"({best_ref[1]['eegnet_mean']:.2f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"altbackbone_{model.lower()}"
    if rows:
        csv_path = os.path.join(OUT_DIR, f"{stem}_4pipeline_per_subject.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
        print(f"Wrote {csv_path}")
    json_path = os.path.join(OUT_DIR, f"{stem}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
