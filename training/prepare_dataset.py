#!/usr/bin/env python3
"""Build a reproducible nnU-Net v2 dataset from the authoritative VS index.

Relative image and mask paths are resolved from ``--source-root``. By default,
indexes stored in a ``data/`` directory resolve from its project parent; other
indexes resolve from the directory containing the CSV. The destination
must not already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"case_id", "t1_img_path", "t1_seg_path", "fold"}
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: Path
    mask_path: Path
    fold: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an nnU-Net raw dataset, preserving the canonical fastMONAI "
            "five-fold assignments. Existing datasets are never overwritten."
        )
    )
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--label-mode",
        choices=("binary", "largest-component"),
        required=True,
        help="Binarize masks only, or additionally retain only the largest component.",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        required=True,
        help="Dataset index containing case IDs, image/mask paths, and folds.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Base for relative paths in the CSV. Defaults to the CSV's project "
            "directory (the parent of data/ when applicable)."
        ),
    )
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        help=(
            "nnUNet_raw destination. Defaults to $nnUNet_raw or "
            "<repository>/nnUNet_data/nnUNet_raw."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel mask conversion workers (default: up to 16).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of hard-linking them when possible.",
    )
    return parser


def _default_source_root(data_csv: Path) -> Path:
    return data_csv.parent.parent if data_csv.parent.name == "data" else data_csv.parent


def _resolve_source(value: str, source_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (source_root / path).resolve()


def load_cases(data_csv: Path, source_root: Path) -> list[Case]:
    """Load and fully validate the dataset index before any output is written."""

    if not data_csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {data_csv}")
    frame = pd.read_csv(data_csv)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset CSV is missing columns: {missing_columns}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("case_id, image path, mask path, and fold must all be present")

    case_ids = frame["case_id"].astype(str)
    if case_ids.duplicated().any():
        duplicates = sorted(case_ids[case_ids.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate case_id values: {duplicates[:10]}")

    cases: list[Case] = []
    missing_files: list[Path] = []
    for row in frame.itertuples(index=False):
        case_id = str(row.case_id).strip()
        if not case_id or "/" in case_id or "\\" in case_id:
            raise ValueError(f"Invalid case_id: {case_id!r}")
        try:
            fold = int(row.fold)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid fold for {case_id}: {row.fold!r}") from error

        image_path = _resolve_source(str(row.t1_img_path), source_root)
        mask_path = _resolve_source(str(row.t1_seg_path), source_root)
        if not image_path.is_file():
            missing_files.append(image_path)
        if not mask_path.is_file():
            missing_files.append(mask_path)
        cases.append(Case(case_id, image_path, mask_path, fold))

    if missing_files:
        shown = "\n  ".join(map(str, missing_files[:10]))
        raise FileNotFoundError(
            f"Missing source files ({len(missing_files)} total; first 10):\n  {shown}"
        )
    if len(set(case.fold for case in cases)) < 2:
        raise ValueError("At least two folds are required to create cross-validation splits")
    return cases


def make_splits(cases: list[Case]) -> list[dict[str, list[str]]]:
    """Convert the index's fold labels into nnU-Net split order."""

    splits = []
    for validation_fold in sorted(set(case.fold for case in cases)):
        splits.append(
            {
                "train": [case.case_id for case in cases if case.fold != validation_fold],
                "val": [case.case_id for case in cases if case.fold == validation_fold],
            }
        )
    return splits


def convert_mask(source: Path, destination: Path, label_mode: str) -> int:
    """Binarize one mask, optionally retaining only its largest component."""

    image = sitk.ReadImage(str(source))
    binary = (sitk.GetArrayFromImage(image) > 0).astype(np.uint8)
    components, component_count = label(binary)
    if label_mode == "largest-component" and component_count > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        binary = (components == int(sizes.argmax())).astype(np.uint8)

    converted = sitk.GetImageFromArray(binary)
    converted.CopyInformation(image)
    sitk.WriteImage(converted, str(destination))
    return int(component_count)


def _place_image(source: Path, destination: Path, copy_images: bool) -> str:
    if copy_images:
        shutil.copy2(source, destination)
        return "copied"
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def build_dataset(
    *,
    cases: list[Case],
    destination: Path,
    dataset_name: str,
    label_mode: str,
    workers: int,
    copy_images: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build into a sibling staging directory, then atomically publish it."""

    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. "
            "Choose another dataset ID or remove it explicitly after checking its contents."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        images_dir = stage / "imagesTr"
        labels_dir = stage / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()

        placement_counts = {"linked": 0, "copied": 0}
        for case in cases:
            disposition = _place_image(
                case.image_path,
                images_dir / f"{case.case_id}_0000.nii.gz",
                copy_images,
            )
            placement_counts[disposition] += 1

        component_counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    convert_mask,
                    case.mask_path,
                    labels_dir / f"{case.case_id}.nii.gz",
                    label_mode,
                ): case.case_id
                for case in cases
            }
            for future in as_completed(futures):
                component_counts[futures[future]] = future.result()

        dataset_json = {
            "channel_names": {"0": "T1"},
            "labels": {"background": 0, "VS": 1},
            "numTraining": len(cases),
            "file_ending": ".nii.gz",
            "dataset_name": dataset_name,
        }
        (stage / "dataset.json").write_text(
            json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "splits_final.json").write_text(
            json.dumps(make_splits(cases), indent=2) + "\n", encoding="utf-8"
        )
        with (stage / "component_counts.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=("case_id", "num_components"))
            writer.writeheader()
            for case_id, count in sorted(component_counts.items()):
                writer.writerow({"case_id": case_id, "num_components": count})

        stage.replace(destination)
        return placement_counts, component_counts
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.dataset_id <= 999:
        raise SystemExit("--dataset-id must be between 1 and 999")
    if not DATASET_NAME_PATTERN.fullmatch(args.dataset_name):
        raise SystemExit(
            "--dataset-name must start with an alphanumeric character and contain "
            "only letters, numbers, and underscores"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    data_csv = args.data_csv.expanduser().resolve()
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root
        else _default_source_root(data_csv)
    )
    configured_raw = args.nnunet_raw or os.environ.get("nnUNet_raw")
    nnunet_raw = (
        Path(configured_raw).expanduser().resolve()
        if configured_raw
        else REPOSITORY_ROOT / "nnUNet_data" / "nnUNet_raw"
    )
    folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    destination = nnunet_raw / folder_name

    cases = load_cases(data_csv, source_root)
    splits = make_splits(cases)
    print(f"Validated {len(cases)} cases from {data_csv}")
    print(f"Source root: {source_root}")
    print(f"Fold sizes: {[len(split['val']) for split in splits]}")
    print(f"Destination: {destination}")

    placements, component_counts = build_dataset(
        cases=cases,
        destination=destination,
        dataset_name=args.dataset_name,
        label_mode=args.label_mode,
        workers=args.workers,
        copy_images=args.copy_images,
    )
    multiple = sum(count > 1 for count in component_counts.values())
    print(
        f"Built {folder_name}: {placements['linked']} images linked, "
        f"{placements['copied']} copied, {multiple} masks originally had multiple components"
    )
    print(
        "After nnUNetv2_plan_and_preprocess, copy splits_final.json from this "
        "raw dataset into the matching nnUNet_preprocessed dataset directory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
