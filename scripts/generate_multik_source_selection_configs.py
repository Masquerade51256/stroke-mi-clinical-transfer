#!/usr/bin/env python
"""
Generate EEGNet clinical source-selection LOSO configs for the multi-K
sensitivity analysis.

Base configs:
  - configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_noaug.yaml  (LR+noEA)
  - configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml          (LR+EA)

For each pipeline and each K in K_VALUES, emit a config that adds a
`trainer.args.source_selection` block and shrinks subject_buffer_size to K.
K=49 (all sources) is intentionally excluded: those are the existing
LR+noEA / LR+EA full-source LOSO results.
"""

import yaml

K_VALUES = [5, 10, 15, 20, 25]

PIPELINES = {
    "lr_noEA": "configs/experiment/xwstroke_eegnet_full_loso_lr_noEA_noaug.yaml",
    "lr_ea": "configs/experiment/xwstroke_eegnet_full_loso_lr_ea.yaml",
}

OUT_TEMPLATE = "configs/experiment/xwstroke_eegnet_full_loso_{pipe}_source_sel_k{k}.yaml"


def main():
    for pipe, base_path in PIPELINES.items():
        with open(base_path) as f:
            base = yaml.safe_load(f)

        for k in K_VALUES:
            cfg = dict(base)
            ea_tag = "EA" if pipe == "lr_ea" else "noEA"
            cfg["experiment"] = dict(base["experiment"])
            cfg["experiment"]["name"] = f"XWStroke_Full50_EEGNet_LR_{ea_tag}_SourceSel_K{k}"
            cfg["experiment"]["description"] = (
                f"Full 50-subject LOSO with EEGNet, LR labels, {ea_tag}, "
                f"top-{k} clinical source selection (multi-K sensitivity analysis)"
            )

            trainer_args = dict(base["trainer"]["args"])
            trainer_args["subject_buffer_size"] = k
            trainer_args["source_selection"] = {
                "enabled": True,
                "metadata_path": "results/stratified/stratified_analysis_detailed.csv",
                "k": k,
                "location_weight": 1.0,
                "duration_weight": 1.0,
                "nihss_weight": 1.0,
                "require_same_location": False,
            }
            cfg["trainer"] = {"type": base["trainer"]["type"], "args": trainer_args}

            out_path = OUT_TEMPLATE.format(pipe=pipe, k=k)
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
