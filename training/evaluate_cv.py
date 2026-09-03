#!/usr/bin/env python3
"""Evaluate nnU-Net held-out fold predictions with the fastMONAI VS metrics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FASTMONAI_ROOT = REPOSITORY_ROOT.parent / "fastMONAI"
DEFAULT_PLANS = "nnUNetResEncUNetLPlans"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_TOLERANCES = (0.5, 1.0, 2.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score nnU-Net fold validation predictions in original NIfTI spacing. "
            "No postprocessing is applied."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--nnunet-raw", type=Path)
    parser.add_argument("--nnunet-results", type=Path)
    parser.add_argument("--fastmonai-root", type=Path, default=DEFAULT_FASTMONAI_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <trained-model>/cv_metrics.",
    )
    parser.add_argument(
        "--allow-missing-folds",
        action="store_true",
        help="Evaluate available folds instead of failing when a requested fold is absent.",
    )
    return parser


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return (np.asarray(image.dataobj) > 0).astype(np.float32), spacing


def compute_case_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    metrics,
) -> dict[str, object]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    prediction_tensor = torch.from_numpy(prediction)
    target_tensor = torch.from_numpy(target)
    prediction_5d = prediction_tensor[None, None]
    target_5d = target_tensor[None, None]
    surface = metrics.calculate_surface_metrics(
        prediction,
        target,
        spacing,
        nsd_tolerances_mm=DEFAULT_TOLERANCES,
    )
    return {
        "dsc": float(metrics.calculate_dsc(prediction_tensor, target_tensor).item()),
        "sensitivity": float(
            metrics.calculate_confusion_metrics(
                prediction_5d, target_5d, "sensitivity"
            ).nanmean().item()
        ),
        "precision": float(
            metrics.calculate_confusion_metrics(
                prediction_5d, target_5d, "precision"
            ).nanmean().item()
        ),
        "ldr": float(
            metrics.calculate_lesion_detection_rate(
                prediction_5d, target_5d
            ).nanmean().item()
        ),
        "rve": float(
            metrics.calculate_signed_rve(prediction_5d, target_5d).nanmean().item()
        ),
        "assd_mm": surface["assd_mm"],
        "hd95_mm": surface["hd95_mm"],
        **{
            f"nsd_tau{tolerance}_mm": surface[f"nsd_tau{tolerance}_mm"]
            for tolerance in DEFAULT_TOLERANCES
        },
        "surface_status": surface["status"],
        "spacing_mm": "x".join(f"{value:.6g}" for value in surface["spacing_mm"]),
    }


def summarize(frame: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    rows = []
    groups = [
        ("all", frame),
        *[(f"fold_{fold}", group) for fold, group in frame.groupby("fold")],
    ]
    for group_name, group in groups:
        for metric_name in metric_names:
            values = group[metric_name].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "group": group_name,
                    "metric": metric_name,
                    "mean": float(finite.mean()) if len(finite) else float("nan"),
                    "std": float(finite.std()) if len(finite) else float("nan"),
                    "finite_cases": int(len(finite)),
                    "total_cases": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = _parser().parse_args()
    fastmonai_root = args.fastmonai_root.expanduser().resolve()
    if not fastmonai_root.is_dir():
        raise SystemExit(f"fastMONAI checkout not found: {fastmonai_root}")
    sys.path.insert(0, str(fastmonai_root))
    from fastMONAI import vision_metrics as metrics

    raw_setting = args.nnunet_raw or os.environ.get("nnUNet_raw")
    results_setting = args.nnunet_results or os.environ.get("nnUNet_results")
    nnunet_raw = (
        Path(raw_setting).expanduser().resolve()
        if raw_setting
        else REPOSITORY_ROOT / "nnUNet_data" / "nnUNet_raw"
    )
    nnunet_results = (
        Path(results_setting).expanduser().resolve()
        if results_setting
        else REPOSITORY_ROOT / "nnUNet_data" / "nnUNet_results"
    )
    labels_dir = nnunet_raw / args.dataset / "labelsTr"
    model_dir = (
        nnunet_results
        / args.dataset
        / f"{args.trainer}__{args.plans}__{args.configuration}"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else model_dir / "cv_metrics"
    )
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not model_dir.is_dir():
        raise SystemExit(f"Trained model directory not found: {model_dir}")

    rows = []
    missing_folds = []
    for fold in args.folds:
        validation_dir = model_dir / f"fold_{fold}" / "validation"
        predictions = sorted(validation_dir.glob("*.nii.gz"))
        if not predictions:
            missing_folds.append(fold)
            continue
        print(f"fold {fold}: {len(predictions)} predictions")
        for prediction_path in predictions:
            case_id = prediction_path.name.removesuffix(".nii.gz")
            target_path = labels_dir / f"{case_id}.nii.gz"
            if not target_path.is_file():
                raise FileNotFoundError(f"Ground-truth mask not found: {target_path}")
            prediction, _ = load_mask(prediction_path)
            target, spacing = load_mask(target_path)
            rows.append(
                {
                    "fold": fold,
                    "case_id": case_id,
                    **compute_case_metrics(prediction, target, spacing, metrics),
                }
            )

    if missing_folds and not args.allow_missing_folds:
        raise SystemExit(
            f"No validation predictions for requested folds: {missing_folds}. "
            "Use --allow-missing-folds only for an intentional partial evaluation."
        )
    if not rows:
        raise SystemExit("No validation predictions were found")

    frame = pd.DataFrame(rows).sort_values(["fold", "case_id"]).reset_index(drop=True)
    if frame["case_id"].duplicated().any():
        duplicated = sorted(
            frame.loc[frame["case_id"].duplicated(keep=False), "case_id"].unique()
        )
        raise RuntimeError(
            f"Cases occurred in more than one validation fold: {duplicated[:10]}"
        )
    metric_names = [
        "dsc",
        "sensitivity",
        "precision",
        "ldr",
        "rve",
        "assd_mm",
        "hd95_mm",
        *[f"nsd_tau{tolerance}_mm" for tolerance in DEFAULT_TOLERANCES],
    ]
    summary = summarize(frame, metric_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_case.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    overall = summary[summary["group"] == "all"]
    print(f"\nEvaluated {len(frame)} cases")
    print(overall[["metric", "mean", "std", "finite_cases"]].to_string(index=False))
    print(f"\nWrote {output_dir / 'per_case.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
