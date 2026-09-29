#!/usr/bin/env python
"""
Generate 4-pipeline LOSO configs for an alternative backbone
(architecture robustness check).

Usage:
    python scripts/generate_altbackbone_loso_configs.py --model ShallowConvNet

Bases (the four canonical EEGNet pipeline configs):
  LR+noEA : xwstroke_eegnet_full_loso_lr_noEA_noaug.yaml
  LR+EA   : xwstroke_eegnet_full_loso_lr_ea.yaml
  Aff+noEA: xwstroke_eegnet_full_loso_affalign_noEA_noaug.yaml
  Aff+EA  : xwstroke_eegnet_full_loso_ea_affalign_no_mp.yaml

Training hyperparameters are kept identical to EEGNet (Adam, lr=1e-3,
wd=1e-2, batch=512, epochs=100, mixed_precision=false) so the comparison
isolates the backbone.
"""

import argparse
import yaml

PIPELINES = {
    "lr_noEA": "configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_noaug.yaml",
    "lr_ea": "configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml",
    "affalign_noEA": "configs/experiment/xwstroke_eegnet_full_loso_affalign_noEA_noaug.yaml",
    "ea_affalign": "configs/experiment/xwstroke_eegnet_full_loso_ea_affalign_no_mp.yaml",
}

OUT_TEMPLATE = "configs/experiment/xwstroke_{model_lower}_full_loso_{pipe}.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ShallowConvNet",
                        help="Registered model name (e.g. ShallowConvNet, DeepConvNet)")
    parser.add_argument("--model-args", default="{}",
                        help="YAML dict string for model.args, e.g. \"{drop_out: 0.5}\"")
    args = parser.parse_args()

    model_args = yaml.safe_load(args.model_args) or {}

    for pipe, base_path in PIPELINES.items():
        with open(base_path) as f:
            cfg = yaml.safe_load(f)

        tag = pipe.replace("lr_noEA", "LR_noEA").replace("lr_ea", "LR_EA") \
                  .replace("affalign_noEA", "AffAlign_noEA").replace("ea_affalign", "EA_AffAlign")
        cfg["experiment"]["name"] = f"XWStroke_Full50_{args.model}_{tag}"
        cfg["experiment"]["description"] = (
            f"Full 50-subject LOSO with {args.model}, {tag} "
            f"(architecture robustness check, hyperparameters matched to EEGNet)"
        )
        cfg["model"] = {"type": args.model, "args": dict(model_args)}

        out_path = OUT_TEMPLATE.format(model_lower=args.model.lower(), pipe=pipe)
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
