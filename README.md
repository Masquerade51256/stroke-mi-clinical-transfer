# The Limits of Clinical Metadata for Cross-Subject Stroke Motor Imagery Decoding

Official code release for the paper:

> **The Limits of Clinical Metadata for Cross-Subject Stroke Motor Imagery Decoding: Evidence from Anatomy-Aware Relabelling and Alignment**
> *IEEE International Conference on Biomedical and Health Informatics (BHI) 2026 — accepted.*

## Overview

Cross-subject transfer learning is a promising way to reduce subject-specific
calibration in stroke motor-imagery (MI) BCIs. An intuitive idea is to use
**clinical metadata** (lesion location, stroke duration, NIHSS severity) to
guide source-subject selection or validation. This repository contains the
code used to systematically test that assumption on a public dataset of
**50 acute stroke patients**.

**Main findings:**

- Clinical-similarity-based source-subject selection does **not** improve
  decoding performance.
- Validation-based pipeline selection using the clinically nearest subject
  (**49.16%**) significantly **underperforms** the best fixed pipeline
  (**51.32%**, *p* = 0.030) and offers no clear advantage over random
  validation (**50.71%**).
- Validation accuracy on the clinically nearest subject is essentially
  **uncorrelated** with target test accuracy (*r* = 0.040, *p* = 0.574).
- Post-hoc analyses show patients with **subcortical lesions** benefit
  disproportionately from anatomy-aware (affected/unaffected) relabelling and
  Euclidean Alignment (EA), whereas NIHSS and stroke duration are not
  predictive.
- An oracle selector reaches **55.68%**, indicating substantial
  patient-specific headroom — but coarse clinical labels are insufficient as
  direct decision rules. EEG-based validation proxies are needed instead.

## The four decoding pipelines

All experiments use leave-one-subject-out (LOSO) cross-validation over the
50 patients, crossing two design choices:

| Pipeline  | Labels                              | Euclidean Alignment |
|-----------|-------------------------------------|---------------------|
| LR+noEA   | Left / Right hand                   | No                  |
| LR+EA     | Left / Right hand                   | Yes                 |
| Aff+noEA  | Affected / Unaffected (hemisphere-aligned) | No            |
| Aff+EA    | Affected / Unaffected (hemisphere-aligned) | Yes           |

Backbones: **EEGNet** (main), **ADFCNN**, plus **ShallowConvNet** and
**DeepConvNet** for the architecture-robustness analysis.

## Installation

```bash
git clone https://github.com/Masquerade51256/stroke-mi-clinical-transfer.git
cd stroke-mi-clinical-transfer

conda create -n stroke-mi python=3.10
conda activate stroke-mi

# Install PyTorch for your CUDA setup, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Data preparation

This project uses the public dataset:

> Liu, H. et al. *An EEG motor imagery dataset for brain computer interface in
> acute stroke patients.* Scientific Data 11, 131 (2024).
> [figshare:21679035](https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035)

Download `sourcedata.zip` (raw per-subject `.mat` files) and the participant
characteristics table from the link above, and arrange them as:

```
src/datasets/21679035/
├── participants.tsv                  # patient characteristics (BIDS format)
└── sourcedata/
    ├── sub-01/
    │   └── sub-01_task-motor-imagery_eeg.mat
    ├── sub-02/
    │   └── ...
    └── sub-50/
```

No other preprocessing step is required — filtering, resampling, windowing,
Euclidean Alignment, and hemisphere alignment are all handled by the
framework at load time, driven by the YAML configs.

## Quick start

Run one full 50-subject LOSO experiment:

```bash
python train.py --config configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml
```

Results are written to `experiments/<EXPERIMENT_NAME>/results/results.json`.

A complete, step-by-step mapping from each experiment in the paper to the
exact commands is given in [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Repository structure

```
├── train.py                    # Main entry point (YAML-config driven)
├── experiments/run_experiment.py
├── core/                       # Registry, config system, experiment manager
├── data/                       # Dataset loaders (BIDS/EDF/.mat)
├── models/                     # EEGNet, ADFCNN, ShallowConvNet, DeepConvNet
├── preprocessing/              # Filter banks, resampling, EA, augmentation
├── trainers/                   # LOSO / streaming-LOSO / supervised trainers
├── utils/                      # Clinical similarity, paths, logging, plots
├── configs/
│   ├── dataset/                # Label scheme x EA variants of XWStroke
│   └── experiment/             # Full-50 LOSO experiment configs
├── scripts/                    # Experiment drivers & analysis scripts
└── docs/REPRODUCING.md         # Paper-experiment reproduction guide
```

## Citation

If you use this code, please cite our paper and the dataset:

```bibtex
@inproceedings{liu2026limits,
  title     = {The Limits of Clinical Metadata for Cross-Subject Stroke Motor
               Imagery Decoding: Evidence from Anatomy-Aware Relabelling and
               Alignment},
  author    = {Liu, Yuehan and Ye, Linjia and Kuang, Mengfei and Duan, Shengcai
               and Tao, Wei and Yang, Yi and Wan, Feng},
  booktitle = {Proc. IEEE Int. Conf. Biomedical and Health Informatics (BHI)},
  year      = {2026},
  note      = {Accepted}
}

@article{liu2024stroke,
  title   = {An EEG motor imagery dataset for brain computer interface in
             acute stroke patients},
  author  = {Liu, Haijie and Wei, Penghu and Wang, Haochong and others},
  journal = {Scientific Data},
  volume  = {11},
  pages   = {131},
  year    = {2024}
}
```

## License

This project is released under the [MIT License](LICENSE).
The dataset is distributed separately by its original authors under its own
terms; please refer to the figshare page for details.
