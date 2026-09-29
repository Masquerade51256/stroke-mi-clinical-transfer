#!/usr/bin/env python
"""
Oracle analysis for a conditional LR vs. Aff+Align strategy.

For each subject we already know whether LR or Aff+Align performed better.
This script estimates the upper bound of a perfect selector and tests
simple clinical rules for choosing the strategy.
"""

import json
import argparse
import numpy as np
import pandas as pd


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
    sids = sorted(lr_accs.keys())

    oracle_acc = np.mean([max(lr_accs[sid], aff_accs[sid]) for sid in sids])
    lr_mean = np.mean([lr_accs[sid] for sid in sids])
    aff_mean = np.mean([aff_accs[sid] for sid in sids])

    print("Oracle strategy upper bound")
    print("-" * 60)
    print(f"LR-only:  {lr_mean*100:.2f}%")
    print(f"Aff-only: {aff_mean*100:.2f}%")
    print(f"Oracle:   {oracle_acc*100:.2f}%  (+{oracle_acc-lr_mean:.4f} over LR, +{oracle_acc-aff_mean:.4f} over Aff)")

    strat = pd.read_csv(args.strat)
    strat['subject_id'] = strat['subject_id'].astype(int)
    strat['delta'] = strat['subject_id'].map(lambda sid: aff_accs[sid] - lr_accs[sid])
    strat = strat.set_index('subject_id')

    rules = [
        ("Chronic", strat['Duration_Category'] == 'Chronic(>3mo)'),
        ("Subcortical", strat['StrokeLocation_Category'] == 'Subcortical'),
        ("Low NIHSS (≤4)", strat['NIHSS'] <= 4),
        ("Chronic & Subcortical",
         (strat['Duration_Category'] == 'Chronic(>3mo)') & (strat['StrokeLocation_Category'] == 'Subcortical')),
        ("Chronic & Low NIHSS",
         (strat['Duration_Category'] == 'Chronic(>3mo)') & (strat['NIHSS'] <= 4)),
        ("Subcortical & Low NIHSS",
         (strat['StrokeLocation_Category'] == 'Subcortical') & (strat['NIHSS'] <= 4)),
        ("Chronic & Subcortical & Low NIHSS",
         (strat['Duration_Category'] == 'Chronic(>3mo)') &
         (strat['StrokeLocation_Category'] == 'Subcortical') &
         (strat['NIHSS'] <= 4)),
    ]

    print("\nSimple clinical conditional rules")
    print("-" * 60)
    for name, mask in rules:
        accs = []
        n_aff = 0
        for sid in sids:
            if mask.loc[sid]:
                accs.append(aff_accs[sid])
                n_aff += 1
            else:
                accs.append(lr_accs[sid])
        print(f"{name:35s} (use Aff in {n_aff:2d}/{len(sids)}): {np.mean(accs)*100:.2f}%")


if __name__ == '__main__':
    main()
