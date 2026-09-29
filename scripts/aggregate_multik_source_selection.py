#!/usr/bin/env python
"""
Aggregate multi-K clinical source-selection LOSO results.

Scans experiments/*/config.yaml for runs named
  XWStroke_Full50_EEGNet_LR_{noEA,EA}_SourceSel_K{5,10,15,20,25}
and appends the existing full-source (K=49) runs as the reference row.

For each run also computes the Pearson correlation between each target
subject's mean clinical distance to its selected sources and its test
accuracy (using the same distance implementation as training:
utils/clinical_source_selection with unit weights).

Outputs (results/comparisons/):
  - multik_source_selection_summary.csv
  - multik_source_selection_per_subject.csv
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from utils.clinical_source_selection import (
    load_clinical_metadata,
    select_source_subjects,
    _clinical_distance,
)

EXP_GLOB = "experiments/*/config.yaml"
RUN_NAME_RE = re.compile(
    r"XWStroke_Full50_EEGNet_LR_(?P<tag>noEA|EA)_SourceSel_K(?P<k>\d+)"
)
K49_RUNS = {
    "noEA": "experiments/XWStroke_Full50_EEGNet_LR_noEA_noAug",
    "EA": "experiments/XWStroke_Full50_EEGNet_LR_EA",
}
METADATA_PATH = "results/stratified/stratified_analysis_detailed.csv"
OUT_DIR = "results/comparisons"


def mean_source_distance(test_sid, train_sids, df):
    """Mean unit-weight clinical distance from target to its sources."""
    idx = df.set_index("subject_id")
    target = idx.loc[test_sid]
    t_prof = {
        "location": target["StrokeLocation_Category"],
        "duration": target["duration_norm"],
        "nihss": target["NIHSS"],
        "nihss_std": target["nihss_std"],
    }
    dists = []
    for sid in train_sids:
        src = idx.loc[sid]
        s_prof = {
            "location": src["StrokeLocation_Category"],
            "duration": src["duration_norm"],
            "nihss": src["NIHSS"],
            "nihss_std": target["nihss_std"],
        }
        dists.append(_clinical_distance(t_prof, s_prof, 1.0, 1.0, 1.0))
    return float(np.mean(dists))


def collect_run(exp_dir, tag, k, df):
    results_path = os.path.join(exp_dir, "results", "results.json")
    if not os.path.isfile(results_path):
        return None, None
    with open(results_path) as f:
        res = json.load(f)
    subjects = res["subjects"]
    if len(subjects) != 50:
        print(f"WARNING: {exp_dir} has {len(subjects)}/50 subjects, skipping")
        return None, None

    accs = np.array([s["test_acc"] for s in subjects])
    all_ids = sorted(df["subject_id"].tolist())

    def resolve_train_subjects(s):
        ts = s["train_subjects"]
        if isinstance(ts, list):
            return ts
        # Newer runs store only the count; the selection is deterministic
        # (distance-based), so recompute the same top-K source set.
        candidates = [sid for sid in all_ids if sid != s["test_subject_id"]]
        if isinstance(ts, int) and ts >= len(candidates):
            return candidates
        return select_source_subjects(
            target_subject_id=s["test_subject_id"],
            candidate_source_ids=candidates,
            metadata_df=df,
            k=k,
        )

    dists = np.array([
        mean_source_distance(s["test_subject_id"], resolve_train_subjects(s), df)
        for s in subjects
    ])
    r, p = stats.pearsonr(dists, accs)

    summary = {
        "K": k,
        "pipeline": f"LR+{tag}",
        "mean_accuracy": accs.mean() * 100,
        "std_accuracy": accs.std() * 100,
        "clinical_distance_r": r,
        "p_value": p,
        "experiment_dir": exp_dir,
    }
    per_subject = pd.DataFrame({
        "K": k,
        "pipeline": f"LR+{tag}",
        "test_subject_id": [s["test_subject_id"] for s in subjects],
        "test_acc": accs * 100,
        "mean_source_distance": dists,
    })
    return summary, per_subject


def main():
    df = load_clinical_metadata(METADATA_PATH)

    # Discover multi-K runs (latest dir per (tag, k) with complete results)
    runs = {}
    for cfg_path in sorted(glob.glob(EXP_GLOB)):
        exp_dir = os.path.dirname(cfg_path)
        try:
            with open(cfg_path) as f:
                name = (yaml.safe_load(f).get("experiment") or {}).get("name", "")
        except Exception:
            continue
        m = RUN_NAME_RE.fullmatch(name)
        if m:
            runs[(m.group("tag"), int(m.group("k")))] = exp_dir

    summaries, per_subject_frames = [], []
    for (tag, k), exp_dir in sorted(runs.items(), key=lambda x: (x[0][0], x[0][1])):
        s, ps = collect_run(exp_dir, tag, k, df)
        if s:
            summaries.append(s)
            per_subject_frames.append(ps)
            print(f"K={k:2d} LR+{tag}: {s['mean_accuracy']:.2f}% ± {s['std_accuracy']:.2f}%  "
                  f"r={s['clinical_distance_r']:+.3f} (p={s['p_value']:.3f})  [{exp_dir}]")

    # K=49 full-source reference rows
    for tag, exp_dir in K49_RUNS.items():
        s, ps = collect_run(exp_dir, tag, 49, df)
        if s:
            summaries.append(s)
            per_subject_frames.append(ps)
            print(f"K=49 LR+{tag}: {s['mean_accuracy']:.2f}% ± {s['std_accuracy']:.2f}%  "
                  f"r={s['clinical_distance_r']:+.3f} (p={s['p_value']:.3f})  [{exp_dir}]")

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_df = pd.DataFrame(summaries).sort_values(["pipeline", "K"])
    summary_path = os.path.join(OUT_DIR, "multik_source_selection_summary.csv")
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"\nWrote {summary_path}")

    if per_subject_frames:
        ps_df = pd.concat(per_subject_frames)
        ps_path = os.path.join(OUT_DIR, "multik_source_selection_per_subject.csv")
        ps_df.to_csv(ps_path, index=False, float_format="%.4f")
        print(f"Wrote {ps_path}")


if __name__ == "__main__":
    main()
