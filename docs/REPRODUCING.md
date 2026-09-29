# Reproducing the Paper Experiments

This guide maps every experiment in the paper to the exact commands that
produce it. All commands are run from the repository root.

> **Runtime note.** A single full-50-subject LOSO run (100 epochs) takes
> roughly 4–6 hours on a modern GPU. The full reproduction involves ~20 such
> runs. All experiments use seed 42 (see each config).

## 0. Setup

1. Install dependencies (see README).
2. Download the dataset from
   [figshare:21679035](https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035)
   and place it under `src/datasets/21679035/` as described in the README
   (`participants.tsv` at the dataset root, `.mat` files under `sourcedata/`).

## 1. Fixed pipelines — ADFCNN (Tasks A/B/C)

These produce the reference Left/Right vs. Affected/Aligned results used by
the stratified analyses and the oracle bound.

```bash
# Task A (Left/Right) and Task B (Affected/unaffected), run sequentially:
python scripts/run_full50_loso_comparison.py
#   -> experiments/XWStroke_Full50_LR/
#   -> experiments/XWStroke_Full50_Affected/

# Task C (Affected/unaffected + hemisphere alignment):
python scripts/run_full50_loso_with_alignment.py
#   -> experiments/XWStroke_Full50_LOSO_Affected_Aligned/
```

## 2. Clinical metadata table + stratified analysis

Derives per-subject clinical categories (lesion location, duration) from
`participants.tsv`, computes lateralization indices, and exports the metadata
CSV required by all selection experiments.

```bash
python scripts/run_stratified_analysis.py
#   -> results/stratified/stratified_analysis_detailed.csv   (metadata table)
#   -> results/stratified/...                                (figures + report)
```

Statistical validation of the subgroup effects (paired tests, FDR/Bonferroni,
OLS interaction models):

```bash
python scripts/clinical_stratification_stats.py
```

## 3. Oracle upper bound

```bash
python scripts/conditional_strategy_oracle.py
```

Expected headline number: an oracle that picks the better of LR vs.
Aff+Align per subject reaches **55.68%** mean accuracy.

## 4. Fixed pipelines — EEGNet (the four canonical pipelines)

```bash
python train.py --config configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_noaug.yaml
#   -> experiments/XWStroke_Full50_EEGNet_LR_noEA_noAug/
python train.py --config configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml
#   -> experiments/XWStroke_Full50_EEGNet_LR_EA/
python train.py --config configs/experiment/xwstroke_eegnet_full_loso_affalign_noEA_noaug.yaml
#   -> experiments/XWStroke_Full50_EEGNet_AffAlign_noEA_noAug/
python train.py --config configs/experiment/xwstroke_eegnet_full_loso_ea_affalign_no_mp.yaml
#   -> experiments/XWStroke_Full50_EEGNet_EA_AffAlign_NoMP/
```

The best fixed pipeline (Aff+EA) should give **51.32%** mean test accuracy.

## 5. Validation-based pipeline selection

Requires the metadata CSV from step 2 and (for comparison) the four fixed
pipelines from step 4.

```bash
# Clinically nearest validation subject:
python scripts/validation_based_pipeline_selection.py \
    --val-strategy clinical_nearest \
    --experiment-name XWStroke_ValBased_ClinicalNearest

# Random validation subject, averaged over 5 seeds:
python scripts/validation_based_pipeline_selection.py \
    --val-strategy random --n-random-seeds 5 \
    --experiment-name XWStroke_ValBased_Random5

# Fixed validation subject (ablation):
python scripts/validation_based_pipeline_selection.py \
    --val-strategy fixed \
    --experiment-name XWStroke_ValBased_Fixed
```

Expected headline numbers: clinical-nearest **49.16%** vs. best fixed
pipeline **51.32%** (paired *p* = 0.030); random validation **50.71%**.

Summaries, tables and figures:

```bash
python scripts/summarize_validation_selection.py          # summary + Fig. 4
python scripts/compare_validation_strategies.py           # LaTeX-ready table
python scripts/analyze_validation_selection_clinical.py   # Fig. 3
python scripts/plot_validation_test_correlation.py        # Fig. 5
```

The validation-vs-test correlation analysis should give *r* = 0.040,
*p* = 0.574 (no usable correlation).

## 6. Clinical-similarity source-subject selection

### ADFCNN, Aff+Align pipeline (K = 10, K = 15, same-location-only)

```bash
python train.py --config configs/experiment/xwstroke_adfcnn_full_loso_affalign_source_selection_k10.yaml
python train.py --config configs/experiment/xwstroke_adfcnn_full_loso_affalign_source_selection_k15.yaml
python train.py --config configs/experiment/xwstroke_adfcnn_full_loso_affalign_source_selection_same_loc.yaml
```

### EEGNet multi-K sensitivity analysis (K ∈ {5, 10, 15, 20, 25})

```bash
python scripts/generate_multik_source_selection_configs.py
# generates configs/experiment/xwstroke_eegnet_full_loso_{lr_noEA,lr_ea}_source_sel_k{K}.yaml

# then run each generated config, e.g.:
python train.py --config configs/experiment/xwstroke_eegnet_full_loso_lr_ea_source_sel_k10.yaml
# ... (10 runs in total)

python scripts/aggregate_multik_source_selection.py
```

Post-hoc check of whether source-cohort clinical similarity predicts the
benefit of Aff+Align over LR:

```bash
python scripts/source_selection_pilot.py
```

Expected conclusion: clinical-similarity source selection does **not**
improve decoding performance at any K.

## 7. Lesion-informed rule analysis (post-hoc)

Tests whether a simple lesion-location rule (subcortical → Aff+EA) recovers
part of the oracle headroom:

```bash
python scripts/analyze_lesion_informed_rule.py
```

## 8. Architecture robustness (alternative backbones)

Repeats the four canonical pipelines with ShallowConvNet / DeepConvNet to
confirm the findings are not backbone-specific:

```bash
python scripts/generate_altbackbone_loso_configs.py --model ShallowConvNet
python scripts/generate_altbackbone_loso_configs.py --model DeepConvNet

# run the 8 generated configs (4 pipelines x 2 backbones), e.g.:
python train.py --config configs/experiment/xwstroke_shallowconvnet_full_loso_lr_ea.yaml
# ...

python scripts/aggregate_altbackbone_loso.py --model ShallowConvNet
python scripts/aggregate_altbackbone_loso.py --model DeepConvNet
```

## Expected key results

| Quantity                                   | Value   |
|--------------------------------------------|---------|
| Best fixed pipeline (Aff+EA)               | 51.32%  |
| Clinical-nearest validation selection      | 49.16%  |
| Random validation selection (5 seeds)      | 50.71%  |
| Oracle upper bound                         | 55.68%  |
| Val–test correlation (clinical-nearest)    | r=0.040, p=0.574 |
| Clinical-nearest vs. best fixed (paired)   | p=0.030 |

Small deviations (±0.1–0.3 pp) can occur across hardware/library versions;
the ordering and significance pattern should be stable.
