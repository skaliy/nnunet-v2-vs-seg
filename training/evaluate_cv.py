#!/usr/bin/env python3
"""Evaluate nnU-Net held-out fold predictions with fastMONAI's metric panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANS = "nnUNetResEncUNetLPlans"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_TOLERANCES = (0.5, 1.0, 2.0)
METRIC_NAMES = [
    "dsc",
    "sensitivity",
    "precision",
    "ldr",
    "rve",
    "assd_mm",
    "hd95_mm",
    *[f"nsd_tau{tolerance}_mm" for tolerance in DEFAULT_TOLERANCES],
]


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
    parser.add_argument("--nnunet-preprocessed", type=Path)
    parser.add_argument("--nnunet-results", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <trained-model>/cv_metrics.",
    )
    parser.add_argument(
        "--allow-missing-folds",
        action="store_true",
        help=(
            "Evaluate the folds that have predictions instead of failing when a "
            "requested fold has none. A fold with only part of its validation "
            "cases is always an error."
        ),
    )
    return parser


def _workspace_dir(argument: Path | None, variable: str) -> Path:
    setting = argument or os.environ.get(variable)
    if setting:
        return Path(setting).expanduser().resolve()
    return REPOSITORY_ROOT / "nnUNet_data" / variable


def load_splits(raw_dataset: Path, preprocessed_dataset: Path) -> list[dict[str, list[str]]]:
    """The splits nnU-Net trained with: the preprocessed copy, else the raw one."""
    trained = preprocessed_dataset / "splits_final.json"
    prepared = raw_dataset / "splits_final.json"
    if trained.is_file() and prepared.is_file() and trained.read_bytes() != prepared.read_bytes():
        raise SystemExit(f"splits_final.json differs between {trained} and {prepared}")
    for path in (trained, prepared):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(f"splits_final.json not found in {preprocessed_dataset} or {raw_dataset}")


def check_fold_predictions(fold: int, predicted: list[str], expected: list[str]) -> None:
    """A fold counts only when its predictions are exactly its validation cases."""
    missing = sorted(set(expected) - set(predicted))
    unexpected = sorted(set(predicted) - set(expected))
    if missing or unexpected:
        raise SystemExit(
            f"fold {fold} validation is not complete: {len(predicted)} predictions for "
            f"{len(expected)} validation cases; missing {missing[:5]}, "
            f"not in this fold {unexpected[:5]}"
        )


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return (np.asarray(image.dataobj) > 0).astype(np.float32), spacing


def compute_case_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    calculate_all_metrics,
) -> dict[str, object]:
    """One case's metrics from fastMONAI, in the column order of per_case.csv."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    panel = calculate_all_metrics(
        prediction, target, spacing, nsd_tolerances_mm=DEFAULT_TOLERANCES
    )
    return {
        **{name: float(panel[name]) for name in METRIC_NAMES},
        "surface_status": panel["surface_status"],
        "spacing_mm": "x".join(f"{value:.6g}" for value in panel["spacing_mm"]),
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


def main(argv: list[str] | None = None, calculate_all_metrics=None) -> int:
    args = _parser().parse_args(argv)
    if calculate_all_metrics is None:
        from fastMONAI.vision_metrics import calculate_all_metrics

    raw_dataset = _workspace_dir(args.nnunet_raw, "nnUNet_raw") / args.dataset
    preprocessed_dataset = (
        _workspace_dir(args.nnunet_preprocessed, "nnUNet_preprocessed") / args.dataset
    )
    labels_dir = raw_dataset / "labelsTr"
    model_dir = (
        _workspace_dir(args.nnunet_results, "nnUNet_results")
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
    splits = load_splits(raw_dataset, preprocessed_dataset)
    unknown_folds = [fold for fold in args.folds if not 0 <= fold < len(splits)]
    if unknown_folds:
        raise SystemExit(f"splits_final.json has {len(splits)} folds; no fold {unknown_folds}")

    # Check every requested fold before scoring any, so that an incomplete
    # fold stops the run at once instead of after the earlier folds are scored.
    fold_predictions = []
    missing_folds = []
    for fold in args.folds:
        validation_dir = model_dir / f"fold_{fold}" / "validation"
        predictions = sorted(validation_dir.glob("*.nii.gz"))
        if not predictions:
            missing_folds.append(fold)
            continue
        check_fold_predictions(
            fold,
            [path.name.removesuffix(".nii.gz") for path in predictions],
            splits[fold]["val"],
        )
        fold_predictions.append((fold, predictions))
    if missing_folds and not args.allow_missing_folds:
        raise SystemExit(
            f"No validation predictions for requested folds: {missing_folds}. "
            "Use --allow-missing-folds only for an intentional partial evaluation."
        )
    if not fold_predictions:
        raise SystemExit("No validation predictions were found")

    rows = []
    for fold, predictions in fold_predictions:
        print(f"fold {fold}: {len(predictions)} predictions", flush=True)
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
                    **compute_case_metrics(prediction, target, spacing, calculate_all_metrics),
                }
            )

    frame =pd.DataFrame(rows).sort_values(["fold", "case_id"]).reset_index(drop=True)
    if frame["case_id"].duplicated().any():
        duplicated = sorted(
            frame.loc[frame["case_id"].duplicated(keep=False), "case_id"].unique()
        )
        raise RuntimeError(
            f"Cases occurred in more than one validation fold: {duplicated[:10]}"
        )
    summary = summarize(frame, METRIC_NAMES)
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
