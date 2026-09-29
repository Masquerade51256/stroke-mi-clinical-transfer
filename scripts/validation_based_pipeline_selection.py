#!/usr/bin/env python
r"""
Validation-based pipeline selection for cross-subject stroke EEG decoding.

For each target subject:
  1. Select a validation subject from the source pool.
  2. Train each of the 4 candidate EEGNet pipelines on source \ {val, target}
     and evaluate on the validation subject.
  3. Select the pipeline with the highest validation accuracy.
  4. Re-train the selected pipeline on source \ {target} and evaluate on target.

Candidate pipelines:
  - LR+noEA  : Left/Right labels, no Euclidean Alignment
  - LR+EA    : Left/Right labels, with EA
  - Aff+noEA : Affected/Aligned labels, no EA
  - Aff+EA   : Affected/Aligned labels, with EA
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.registry import DATASETS, MODELS, TRAINERS
from data import DATASETS as DATASET_REGISTRY
from models import register_all_models
from trainers import TRAINERS as TRAINER_REGISTRY
from trainers.streaming_loso_trainer import StreamingLOSOTrainer
from utils.clinical_source_selection import load_clinical_metadata, _clinical_distance
from utils.logging import setup_logger
from utils.path_manager import PathManager


PIPELINES = {
    "LR_noEA": "configs/dataset/XWStroke_noEA_noaug.yaml",
    "LR_EA": "configs/dataset/XWStroke_EA.yaml",
    "Aff_noEA": "configs/dataset/XWStroke_affected_aligned_noEA_noaug.yaml",
    "Aff_EA": "configs/dataset/XWStroke_EA_affected_aligned.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-based pipeline selection")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/experiment/XWStroke_EEGNet_Full50.yaml",
        help="Base experiment config (model/training params)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="XWStroke_ValidationBasedSelection",
        help="Experiment output directory name",
    )
    parser.add_argument(
        "--val-strategy",
        type=str,
        default="clinical_nearest",
        choices=["clinical_nearest", "random", "fixed"],
        help="Strategy for choosing the validation subject",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="results/stratified/stratified_analysis_detailed.csv",
        help="Clinical metadata CSV for clinical_nearest strategy",
    )
    parser.add_argument(
        "--n-random-seeds",
        type=int,
        default=1,
        help="Number of random validation-subject draws to average (random strategy)",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Override subject list (default: 1..50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from incremental selection results if available",
    )
    parser.add_argument(
        "--skip-final-retrain",
        action="store_true",
        help="Skip final re-training (debug only)",
    )
    return parser.parse_args()


def load_dataset_info(dataset_config_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Load dataset class and info dict from a dataset config file."""
    dataset_cfg = Config.fromfile(dataset_config_path).to_dict()
    dataset_type = dataset_cfg["dataset"]["name"]
    dataset_cls = DATASET_REGISTRY.get(dataset_type)
    return dataset_cls, dataset_cfg


def select_validation_subject(
    target_subject_id: int,
    candidate_source_ids: List[int],
    strategy: str,
    metadata_df: Optional[pd.DataFrame],
    seed_offset: int = 0,
    base_seed: int = 42,
) -> int:
    """Select a validation subject from candidate sources for a target subject."""
    if strategy == "clinical_nearest":
        if metadata_df is None:
            raise ValueError("clinical_nearest strategy requires clinical metadata")
        df = metadata_df.set_index("subject_id")
        target = df.loc[target_subject_id]
        target_profile = {
            "location": target["StrokeLocation_Category"],
            "duration": target["duration_norm"],
            "nihss": target["NIHSS"],
            "nihss_std": target["nihss_std"],
        }
        best_sid, best_dist = None, float("inf")
        for sid in candidate_source_ids:
            if sid not in df.index:
                continue
            src = df.loc[sid]
            src_profile = {
                "location": src["StrokeLocation_Category"],
                "duration": src["duration_norm"],
                "nihss": src["NIHSS"],
                "nihss_std": target["nihss_std"],
            }
            dist = _clinical_distance(target_profile, src_profile)
            if dist < best_dist:
                best_dist = dist
                best_sid = sid
        if best_sid is None:
            best_sid = candidate_source_ids[0]
        return best_sid

    elif strategy == "random":
        rng = random.Random(base_seed + target_subject_id + seed_offset)
        return rng.choice(candidate_source_ids)

    elif strategy == "fixed":
        # Use the first candidate as a deterministic placeholder
        return candidate_source_ids[0]

    else:
        raise ValueError(f"Unknown validation strategy: {strategy}")


def create_trainer_for_pipeline(
    base_config: Config,
    dataset_config_path: str,
    device: torch.device,
    paths: PathManager,
    logger: logging.Logger,
) -> StreamingLOSOTrainer:
    """Create a StreamingLOSOTrainer configured for a specific pipeline."""
    config_dict = base_config.to_dict()
    config_dict["data"]["info_path"] = dataset_config_path
    # Ensure mixed precision is disabled (workaround for RTX 5090 SIGILL)
    config_dict.setdefault("trainer", {}).setdefault("args", {})["mixed_precision"] = False
    cfg = Config(config_dict)

    trainer = StreamingLOSOTrainer(
        model=None,
        config=cfg,
        device=device,
        paths=paths,
        logger=logger,
    )
    # Disable the trainer's own incremental resume; we handle it at selection level
    trainer._resume_enabled = False
    return trainer


def run_pipeline_validation(
    trainer: StreamingLOSOTrainer,
    pipeline_name: str,
    target_subject_id: int,
    validation_subject_id: int,
    all_source_ids: List[int],
    dataset_cls: Any,
    dataset_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Train one pipeline with a held-out validation subject and return metrics."""
    train_subjects = [s for s in all_source_ids if s != target_subject_id]
    result, _ = trainer._train_loso_round_streaming(
        test_subject_id=target_subject_id,
        train_subject_ids=train_subjects,
        dataset_cls=dataset_cls,
        dataset_info=dataset_info,
        val_subject_id=validation_subject_id,
    )
    return {
        "pipeline": pipeline_name,
        "target_subject_id": target_subject_id,
        "validation_subject_id": validation_subject_id,
        "val_acc": result["val_acc"],
        "test_acc": result["test_acc"],  # leakage-free test monitoring
        "best_val_acc": result["best_val_acc"],
        "best_val_epoch": result["best_val_epoch"],
        "stopped_early": result["stopped_early"],
        "actual_epochs": result["actual_epochs"],
    }


def run_pipeline_final(
    trainer: StreamingLOSOTrainer,
    pipeline_name: str,
    target_subject_id: int,
    all_source_ids: List[int],
    dataset_cls: Any,
    dataset_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Train the selected pipeline on all sources except target and evaluate on target."""
    train_subjects = [s for s in all_source_ids if s != target_subject_id]
    result, _ = trainer._train_loso_round_streaming(
        test_subject_id=target_subject_id,
        train_subject_ids=train_subjects,
        dataset_cls=dataset_cls,
        dataset_info=dataset_info,
        val_subject_id=None,  # no validation split for final evaluation
    )
    return {
        "pipeline": pipeline_name,
        "target_subject_id": target_subject_id,
        "test_acc": result["test_acc"],
        "test_loss": result["test_loss"],
        "train_subjects": result["train_subjects"],
        "test_samples": result["test_samples"],
    }


def main():
    args = parse_args()

    # Register models and trainers
    register_all_models()

    # Load base config
    if not Path(args.base_config).exists():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    base_config = Config.fromfile(args.base_config)

    # Override device
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # Setup paths and logging
    paths = PathManager(args.experiment_name, base_config.get("paths.root_dir", "./experiments"))
    paths.create_directories()
    logger = setup_logger(
        name=args.experiment_name,
        log_file=paths.log_file,
        level=base_config.get("logging.level", "INFO"),
        console=base_config.get("logging.console", True),
    )
    logger.info(f"Validation-based pipeline selection: strategy={args.val_strategy}")
    logger.info(f"Output directory: {paths.exp_dir}")

    # Set seeds
    seed = base_config.get("experiment.seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Subject list
    if args.subjects:
        subject_ids = sorted(args.subjects)
    else:
        subject_ids = list(range(1, 51))

    # Load clinical metadata if needed
    metadata_df = None
    if args.val_strategy == "clinical_nearest":
        if not Path(args.metadata).exists():
            raise FileNotFoundError(f"Metadata not found: {args.metadata}")
        metadata_df = load_clinical_metadata(args.metadata)
        logger.info(f"Loaded clinical metadata: {len(metadata_df)} subjects")

    # Pre-load dataset class/info for each pipeline
    pipeline_data = {}
    for name, cfg_path in PIPELINES.items():
        dataset_cls, dataset_info = load_dataset_info(cfg_path)
        pipeline_data[name] = (dataset_cls, dataset_info)
        logger.info(f"Pipeline '{name}' -> {cfg_path}")

    # Incremental results file
    incremental_path = paths.results_dir / "incremental_selection.json"
    completed_targets = set()
    all_target_results: List[Dict[str, Any]] = []
    if args.resume and incremental_path.exists():
        try:
            with open(incremental_path, "r", encoding="utf-8") as f:
                inc = json.load(f)
            all_target_results = inc.get("subjects", [])
            completed_targets = {r["target_subject_id"] for r in all_target_results}
            logger.info(f"Resuming: {len(completed_targets)} targets already completed")
        except Exception as e:
            logger.warning(f"Failed to load incremental results: {e}")

    # Main loop
    for target_subject_id in subject_ids:
        if target_subject_id in completed_targets:
            logger.info(f"Skipping target {target_subject_id} (already completed)")
            continue

        start_time = time.time()
        candidate_sources = [s for s in subject_ids if s != target_subject_id]

        # Determine validation subject(s)
        if args.val_strategy == "random" and args.n_random_seeds > 1:
            validation_subject_ids = [
                select_validation_subject(
                    target_subject_id, candidate_sources, args.val_strategy,
                    metadata_df, seed_offset=i, base_seed=seed
                )
                for i in range(args.n_random_seeds)
            ]
        else:
            validation_subject_ids = [
                select_validation_subject(
                    target_subject_id, candidate_sources, args.val_strategy,
                    metadata_df, base_seed=seed
                )
            ]

        logger.info(f"\n{'='*60}")
        logger.info(f"Target {target_subject_id} | Validation subject(s): {validation_subject_ids}")
        logger.info(f"{'='*60}")

        # Evaluate each pipeline on validation subject(s)
        pipeline_val_scores: Dict[str, List[float]] = {name: [] for name in PIPELINES}
        pipeline_validation_details: List[Dict[str, Any]] = []

        for pipeline_name, (dataset_cls, dataset_info) in pipeline_data.items():
            trainer = create_trainer_for_pipeline(
                base_config, PIPELINES[pipeline_name], device, paths, logger
            )
            for val_subject_id in validation_subject_ids:
                detail = run_pipeline_validation(
                    trainer=trainer,
                    pipeline_name=pipeline_name,
                    target_subject_id=target_subject_id,
                    validation_subject_id=val_subject_id,
                    all_source_ids=subject_ids,
                    dataset_cls=dataset_cls,
                    dataset_info=dataset_info,
                )
                pipeline_val_scores[pipeline_name].append(detail["val_acc"])
                pipeline_validation_details.append(detail)
                logger.info(
                    f"  [{pipeline_name}] val@{val_subject_id}: "
                    f"val_acc={detail['val_acc']:.4f}, test_monitor={detail['test_acc']:.4f}"
                )

        # Select best pipeline by mean validation accuracy
        mean_val_scores = {
            name: float(np.mean(scores)) for name, scores in pipeline_val_scores.items()
        }
        best_pipeline = max(mean_val_scores, key=mean_val_scores.get)
        logger.info(
            f"  Selected pipeline: {best_pipeline} "
            f"(mean val acc = {mean_val_scores[best_pipeline]:.4f})"
        )

        # Final re-train on target
        if args.skip_final_retrain:
            final_result = {
                "pipeline": best_pipeline,
                "target_subject_id": target_subject_id,
                "test_acc": None,
                "skipped_final": True,
            }
        else:
            dataset_cls, dataset_info = pipeline_data[best_pipeline]
            trainer = create_trainer_for_pipeline(
                base_config, PIPELINES[best_pipeline], device, paths, logger
            )
            final_result = run_pipeline_final(
                trainer=trainer,
                pipeline_name=best_pipeline,
                target_subject_id=target_subject_id,
                all_source_ids=subject_ids,
                dataset_cls=dataset_cls,
                dataset_info=dataset_info,
            )
            logger.info(
                f"  Final [{best_pipeline}] target {target_subject_id}: "
                f"test_acc={final_result['test_acc']:.4f}"
            )

        target_result = {
            "target_subject_id": target_subject_id,
            "validation_subject_ids": validation_subject_ids,
            "val_strategy": args.val_strategy,
            "pipeline_val_scores": mean_val_scores,
            "pipeline_validation_details": pipeline_validation_details,
            "selected_pipeline": best_pipeline,
            "final": final_result,
            "elapsed_seconds": time.time() - start_time,
        }
        all_target_results.append(target_result)

        # Save incremental results
        incremental_data = {
            "subjects": all_target_results,
            "val_strategy": args.val_strategy,
            "pipeline_configs": PIPELINES,
            "overall": {
                "mean_test_acc": float(np.mean([
                    r["final"]["test_acc"] for r in all_target_results
                    if r["final"].get("test_acc") is not None
                ])),
                "std_test_acc": float(np.std([
                    r["final"]["test_acc"] for r in all_target_results
                    if r["final"].get("test_acc") is not None
                ])),
            },
        }
        tmp_path = incremental_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(incremental_data, f, indent=2)
        tmp_path.replace(incremental_path)

    # Final summary
    test_accs = [r["final"]["test_acc"] for r in all_target_results if r["final"].get("test_acc") is not None]
    final_summary = {
        "subjects": all_target_results,
        "val_strategy": args.val_strategy,
        "pipeline_configs": PIPELINES,
        "overall": {
            "mean_test_acc": float(np.mean(test_accs)),
            "std_test_acc": float(np.std(test_accs)),
            "min_test_acc": float(np.min(test_accs)),
            "max_test_acc": float(np.max(test_accs)),
            "n_subjects": len(test_accs),
        },
    }
    results_path = paths.results_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Validation-based pipeline selection complete")
    logger.info(f"Mean test accuracy: {final_summary['overall']['mean_test_acc']*100:.2f}% ± "
                f"{final_summary['overall']['std_test_acc']*100:.2f}%")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
