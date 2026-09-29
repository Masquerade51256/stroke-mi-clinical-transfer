#!/usr/bin/env python
"""
Pilot analysis for clinical source-subject selection.

Post-hoc analysis: for each LOSO target subject, compute the average clinical
similarity of the source cohort (or the top-K most similar sources) and test
whether this predicts the gain from Aff+Align over LR labels.
"""

import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default='experiments/XWStroke_Full50_LR/results/results.json')
    parser.add_argument('--aff', default='experiments/XWStroke_Full50_LOSO_Affected_Aligned/results/results.json')
    parser.add_argument('--strat', default='results/stratified/stratified_analysis_detailed.csv')
    args = parser.parse_args()

    with open(args.lr) as f:
        lr = json.load(f)
    with open(args.aff) as f:
        aff = json.load(f)

    lr_accs = {r['test_subject_id']: r['test_acc'] for r in lr['subjects']}
    aff_accs = {r['test_subject_id']: r['test_acc'] for r in aff['subjects']}
    delta = {sid: aff_accs[sid] - lr_accs[sid] for sid in lr_accs}

    strat = pd.read_csv(args.strat)
    strat['subject_id'] = strat['subject_id'].astype(int)
    strat = strat.set_index('subject_id')

    loc_map = {'Brainstem': 0, 'Subcortical': 1, 'Cortical': 2, 'Mixed': 3}
    strat['loc_code'] = strat['StrokeLocation_Category'].map(loc_map)
    strat['chronic'] = (strat['Duration_Category'] == 'Chronic(>3mo)').astype(int)
    strat['nihss_norm'] = (strat['NIHSS'] - strat['NIHSS'].mean()) / strat['NIHSS'].std()

    features = strat[['loc_code', 'chronic', 'nihss_norm']].values
    sids = strat.index.values
    n = len(sids)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sum(np.abs(features[i] - features[j]))
    similarity = 1 - dist / dist.max()

    print("=" * 80)
    print("Source Selection Pilot: Post-hoc Similarity Analysis")
    print("=" * 80)

    records = []
    for idx, test_sid in enumerate(sids):
        source_mask = np.ones(n, dtype=bool)
        source_mask[idx] = False

        records.append({
            'test_subject': test_sid,
            'delta': delta[test_sid],
            'avg_similarity': np.mean(similarity[idx, source_mask]),
            'source_subcortical_pct': (strat.loc[sids[source_mask], 'loc_code'] == 1).mean(),
            'source_chronic_pct': strat.loc[sids[source_mask], 'chronic'].mean(),
            'source_low_nihss_pct': (strat.loc[sids[source_mask], 'NIHSS'] <= 4).mean(),
        })

    df = pd.DataFrame(records)

    for var in ['avg_similarity', 'source_subcortical_pct', 'source_chronic_pct', 'source_low_nihss_pct']:
        r, p = stats.pearsonr(df[var], df['delta'])
        print(f"{var:30s}: r={r:+.3f}, p={p:.4f}")

    print("\nTop-K Similar Source Analysis")
    print("-" * 80)
    for K in [5, 10, 15, 20, 25]:
        top_sim_means = []
        for idx in range(n):
            sims = similarity[idx].copy()
            sims[idx] = -1
            top_k_idx = np.argsort(sims)[-K:]
            top_sim_means.append(sims[top_k_idx].mean())
        r, p = stats.pearsonr(top_sim_means, df['delta'])
        print(f"Top-{K:2d} avg similarity: r={r:+.3f}, p={p:.4f}")


if __name__ == '__main__':
    main()
