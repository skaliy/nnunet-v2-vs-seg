#!/usr/bin/env python3
"""Validate the DICOM contract of a completed Research-PACS container run."""

import argparse
from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.uid import UID

EXPECTED_OUTPUTS = ("fused", "fused_vote_map", "reports", "mask")


def _dicom_files(directory):
    files = sorted(path for path in Path(directory).rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"no DICOM files found in {directory}")
    return files


def _read_series(directory, *, validate_derived):
    datasets = []
    for path in _dicom_files(directory):
        try:
            dataset = dcmread(str(path))
        except Exception as exc:
            raise RuntimeError(f"cannot read DICOM output {path}: {exc}") from exc
        if validate_derived:
            series_uid = str(dataset.SeriesInstanceUID)
            sop_uid = str(dataset.SOPInstanceUID)
            if not UID(series_uid).is_valid or len(series_uid) > 64:
                raise RuntimeError(f"invalid SeriesInstanceUID in {path}: {series_uid}")
            if not UID(sop_uid).is_valid or len(sop_uid) > 64:
                raise RuntimeError(f"invalid SOPInstanceUID in {path}: {sop_uid}")
            if str(dataset.file_meta.MediaStorageSOPInstanceUID) != sop_uid:
                raise RuntimeError(f"dataset/file-meta SOP UID mismatch in {path}")
        datasets.append(dataset)
    return datasets


def validate(input_dir, output_dir):
    source = _read_series(input_dir, validate_derived=False)
    source_series = {str(dataset.SeriesInstanceUID) for dataset in source}
    source_sops = {str(dataset.SOPInstanceUID) for dataset in source}
    source_studies = {str(dataset.StudyInstanceUID) for dataset in source}

    output_series = set()
    output_sops = set()
    by_name = {}
    for name in EXPECTED_OUTPUTS:
        directory = Path(output_dir) / name
        if not directory.is_dir():
            raise RuntimeError(f"missing Research-PACS output directory: {directory}")
        datasets = _read_series(directory, validate_derived=True)
        by_name[name] = datasets
        series = {str(dataset.SeriesInstanceUID) for dataset in datasets}
        sops = [str(dataset.SOPInstanceUID) for dataset in datasets]
        studies = {str(dataset.StudyInstanceUID) for dataset in datasets}
        if len(series) != 1:
            raise RuntimeError(f"{name} contains {len(series)} DICOM series")
        if len(sops) != len(set(sops)):
            raise RuntimeError(f"{name} contains duplicate SOPInstanceUID values")
        if not studies.issubset(source_studies):
            raise RuntimeError(f"{name} does not preserve the source StudyInstanceUID")
        if output_series.intersection(series) or output_sops.intersection(sops):
            raise RuntimeError(f"DICOM UID collision involving output {name}")
        output_series.update(series)
        output_sops.update(sops)

    if output_series.intersection(source_series) or output_sops.intersection(source_sops):
        raise RuntimeError("derived DICOM UIDs collide with source UIDs")

    mask = by_name["mask"]
    if len(mask) != len(source):
        raise RuntimeError(f"mask slice count {len(mask)} != input {len(source)}")
    for dataset in mask:
        values = set(np.unique(dataset.pixel_array).tolist())
        if not values.issubset({0, 1}):
            raise RuntimeError(f"mask contains values outside {{0,1}}: {values}")
        image_type = list(dataset.get("ImageType") or [])
        if image_type[:2] != ["DERIVED", "SECONDARY"] or "MASK" not in image_type:
            raise RuntimeError(f"mask is not marked DERIVED/SECONDARY...MASK: {image_type}")
        if "nnU-Net" not in str(dataset.get("SeriesDescription", "")):
            raise RuntimeError("mask SeriesDescription lacks nnU-Net identity")
        if "model_version=" not in str(dataset.get("DerivationDescription", "")):
            raise RuntimeError("mask DerivationDescription lacks model identity")

    print(
        "Research-PACS DICOM validation: OK "
        f"({len(output_series)} series, {len(output_sops)} instances)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    validate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
