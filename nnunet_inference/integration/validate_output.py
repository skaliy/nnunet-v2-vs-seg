#!/usr/bin/env python3
"""Validate the DICOM contract of a completed Research-PACS container run."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from pydicom import dcmread
from pydicom.uid import UID

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nnunet_inference.redcap_output import FIELDS, _series, build_mask_payload, decode_mask, model_repeat_instance

EXPECTED_OUTPUTS = ("fused", "fused_vote_map", "reports", "mask", "redcap")


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
        if name == "redcap":
            continue
        if name == "reports" and not any(directory.rglob("*")):
            continue
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
    if "reports" not in by_name and any(np.any(d.pixel_array) for d in mask):
        raise RuntimeError("Nonempty mask did not produce a report")
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

    paths = list((Path(output_dir) / "redcap").glob("*/output.json"))
    if len(paths) != 1:
        raise RuntimeError("Expected exactly one REDCap output.json")
    rows = json.loads(paths[0].read_text())
    values = {row["field_name"]: row["value"] for row in rows}
    if len(rows) != len(FIELDS) or set(values) != set(FIELDS):
        raise RuntimeError("Unexpected REDCap fields")
    payload = json.loads(values["vs_mask_json"])
    deployment = dict(model_type=values["vs_model_type"], bundle_sha256=values["vs_bundle_sha256"],
                      members=[dict(member_id=m) for m in payload["model"]["member_ids"]])
    if deployment["model_type"] != "nnunet_medium":
        raise RuntimeError("Wrong REDCap model type")
    instance = model_repeat_instance(deployment["bundle_sha256"])
    if any(row["redcap_repeat_instance"] != instance or row["redcap_repeat_instrument"] != "pr2mask"
           for row in rows):
        raise RuntimeError("Incorrect REDCap model destination")
    expected, _ = build_mask_payload(Path(output_dir) / "mask", input_dir, deployment,
                                     version=values["vs_deployment_version"],
                                     use_tta=values["vs_tta"] == "1")
    np.testing.assert_array_equal(decode_mask(payload), decode_mask(expected))
    for key in ("geometry", "source", "model", "mask_series_uid", "prediction_id"):
        if payload[key] != expected[key]:
            raise RuntimeError(f"REDCap payload differs from written DICOM: {key}")
    if values["vs_prediction_id"] != expected["prediction_id"]:
        raise RuntimeError("REDCap prediction identity differs")
    if not isinstance(json.loads(values["vs_measurements_json"]), list):
        raise RuntimeError("Invalid REDCap measurements list")
    with zipfile.ZipFile(paths[0].with_name("output_data_dictionary.zip")) as archive:
        dictionary = list(csv.DictReader(io.StringIO(archive.read("instrument.csv").decode("utf-8-sig"))))
        if not set(FIELDS).issubset({row["Variable / Field Name"] for row in dictionary}):
            raise RuntimeError("Missing REDCap dictionary fields")

    print(
        "Research-PACS DICOM and lossless REDCap validation: OK "
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
