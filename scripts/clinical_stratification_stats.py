#!/usr/bin/env python
"""
Statistical validation of clinical-stratification effects for Aff+Align vs LR.

Outputs:
  - paired t-test p-values for each subgroup
  - Benjamini-Hochberg FDR and Bonferroni corrected p-values
  - OLS main-effects and interaction models
  - combined "favorable" profile analysis
"""

import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.multitest import multipletests


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
    strat['delta'] = strat['subject_id'].map(delta)

    print("=" * 80)
    print("A. Paired t-tests and Multiple-Comparison Correction")
    print("=" * 80)

    subgroups = []
    pvals = []

    # Overall
    _, p = stats.ttest_rel(
        [aff_accs[sid] for sid in sorted(lr_accs)],
        [lr_accs[sid] for sid in sorted(lr_accs)]
    )
    subgroups.append('Overall'); pvals.append(p)

    # Location
    for cat in ['Brainstem', 'Cortical', 'Mixed', 'Subcortical']:
        subjects = strat[strat['StrokeLocation_Category'] == cat]['subject_id'].values
        _, p = stats.ttest_rel([aff_accs[sid] for sid in subjects],
                               [lr_accs[sid] for sid in subjects])
        subgroups.append(f'Location_{cat}'); pvals.append(p)

    # Duration
    for cat in ['Acute(≤3mo)', 'Chronic(>3mo)']:
        subjects = strat[strat['Duration_Category'] == cat]['subject_id'].values
        _, p = stats.ttest_rel([aff_accs[sid] for sid in subjects],
                               [lr_accs[sid] for sid in subjects])
        subgroups.append(f'Duration_{cat}'); pvals.append(p)

    # NIHSS
    strat['NIHSS_Category'] = pd.cut(strat['NIHSS'], bins=[-1, 4, 100], labels=['Low(≤4)', 'High(>4)'])
    for cat in ['Low(≤4)', 'High(>4)']:
        subjects = strat[strat['NIHSS_Category'] == cat]['subject_id'].values
        _, p = stats.ttest_rel([aff_accs[sid] for sid in subjects],
                               [lr_accs[sid] for sid in subjects])
        subgroups.append(f'NIHSS_{cat}'); pvals.append(p)

    reject_fdr, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    reject_bonf, pvals_bonf, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

    for sg, p, pf, pb, r in zip(subgroups, pvals, pvals_fdr, pvals_bonf, reject_fdr):
        print(f"{sg:25s} raw={p:.4f}  FDR={pf:.4f}  Bonf={pb:.4f}  sig(FDR)={'YES' if r else 'NO'}")

    print("\n" + "=" * 80)
    print("B. OLS Main-Effects Model: Delta ~ Location + Duration + NIHSS")
    print("=" * 80)

    strat['Location_Subcortical'] = (strat['StrokeLocation_Category'] == 'Subcortical').astype(int)
    strat['Location_Cortical'] = (strat['StrokeLocation_Category'] == 'Cortical').astype(int)
    strat['Location_Brainstem'] = (strat['StrokeLocation_Category'] == 'Brainstem').astype(int)
    strat['Chronic'] = (strat['Duration_Category'] == 'Chronic(>3mo)').astype(int)
    strat['NIHSS_Low'] = (strat['NIHSS'] <= 4).astype(int)

    X = strat[['Location_Subcortical', 'Location_Cortical', 'Location_Brainstem',
               'Chronic', 'NIHSS_Low']].copy()
    X = add_constant(X)
    y = strat['delta']
    model = OLS(y, X, missing='drop').fit()
    print(model.summary())

    print("\n" + "=" * 80)
    print("C. OLS Interaction Model")
    print("=" * 80)

    X['Subcortical_x_Chronic'] = X['Location_Subcortical'] * X['Chronic']
    X['Subcortical_x_LowNIHSS'] = X['Location_Subcortical'] * X['NIHSS_Low']
    X['Chronic_x_LowNIHSS'] = X['Chronic'] * X['NIHSS_Low']
    model2 = OLS(y, X, missing='drop').fit()
    print(model2.summary())

    print("\n" + "=" * 80)
    print("D. Combined 'Favorable' Profile")
    print("=" * 80)

    strat['Favorable'] = (
        (strat['StrokeLocation_Category'] == 'Subcortical') &
        (strat['Duration_Category'] == 'Chronic(>3mo)') &
        (strat['NIHSS'] <= 4)
    ).astype(int)
    fav = strat[strat['Favorable'] == 1]
    other = strat[strat['Favorable'] == 0]

    fav_lr = [lr_accs[sid] for sid in fav['subject_id']]
    fav_aff = [aff_accs[sid] for sid in fav['subject_id']]
    other_lr = [lr_accs[sid] for sid in other['subject_id']]
    other_aff = [aff_accs[sid] for sid in other['subject_id']]

    print(f"Favorable (n={len(fav)}):       LR {np.mean(fav_lr)*100:.2f}%  "
          f"Aff+Align {np.mean(fav_aff)*100:.2f}%  "
          f"diff={np.mean(fav_aff)-np.mean(fav_lr):+.3f}")
    print(f"Non-favorable (n={len(other)}): LR {np.mean(other_lr)*100:.2f}%  "
          f"Aff+Align {np.mean(other_aff)*100:.2f}%  "
          f"diff={np.mean(other_aff)-np.mean(other_lr):+.3f}")

    fav_gain = [aff_accs[sid] - lr_accs[sid] for sid in fav['subject_id']]
    other_gain = [aff_accs[sid] - lr_accs[sid] for sid in other['subject_id']]
    t, p = stats.ttest_ind(fav_gain, other_gain)
    print(f"Difference in gains: t={t:.3f}, p={p:.4f}")


if __name__ == '__main__':
    main()
