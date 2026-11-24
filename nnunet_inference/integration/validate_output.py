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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nnunet_inference.integration.dicom_checks import check_derived_identity
from nnunet_inference.redcap_output import (
    INSTRUMENT, MODEL_VERSION_FIELD, model_instance_base, pr2mask_dictionary_rows)

EXPECTED_OUTPUTS = ("fused", "fused_vote_map", "reports", "labels", "redcap")
# Outputs that must lie on the input's slices, and how far their geometry may differ (mm).
ON_INPUT_GRID = ("labels", "fused", "fused_vote_map")
GEOMETRY_TOLERANCE = 1e-3


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
            check_derived_identity(dataset, path)
        datasets.append(dataset)
    return datasets


def _geometry(dataset, name, keyword):
    value = dataset.get(keyword)
    if value is None:
        raise RuntimeError(f"{name} has a slice without {keyword}")
    return np.asarray(value, dtype=float)


def _check_geometry(name, datasets, source):
    """The output has the input's slices: the same number and, paired along the
    input's slice normal, the same pixel grid and frame of reference."""
    if len(datasets) != len(source):
        raise RuntimeError(f"{name} slice count {len(datasets)} != input {len(source)}")
    orientation = _geometry(source[0], "input", "ImageOrientationPatient")
    normal = np.cross(orientation[:3], orientation[3:])

    def along_normal(series, series_name):
        return sorted(series, key=lambda d: float(
            _geometry(d, series_name, "ImagePositionPatient") @ normal))

    pairs = zip(along_normal(datasets, name), along_normal(source, "input"))
    for number, (output, original) in enumerate(pairs, start=1):
        differences = [f"{keyword} {output.get(keyword)} != {original.get(keyword)}"
                       for keyword in ("Rows", "Columns")
                       if output.get(keyword) != original.get(keyword)]
        for keyword in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
            value, expected = _geometry(output, name, keyword), _geometry(original, "input", keyword)
            if value.shape != expected.shape or not np.allclose(
                    value, expected, rtol=0, atol=GEOMETRY_TOLERANCE):
                differences.append(f"{keyword} {value.tolist()} != {expected.tolist()}")
        frame, expected_frame = (str(d.get("FrameOfReferenceUID", "")) for d in (output, original))
        if frame != expected_frame:
            differences.append(f"FrameOfReferenceUID {frame} != {expected_frame}")
        if differences:
            raise RuntimeError(f"{name} slice {number} (along the slice normal) differs from "
                               f"the input slice: {'; '.join(differences)}")


def _redcap_destination(source):
    """record_id and redcap_event_name as the container derives them from the input."""
    first = source[0]
    event = str(first.get("ReferringPhysicianName", ""))
    return dict(record_id=str(first.get("PatientID", "")).rstrip(),
                redcap_event_name=event.removeprefix("EventName:"))


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
    for name in ON_INPUT_GRID:
        _check_geometry(name, by_name[name], source)

    mask = by_name["labels"]
    fused_study_ids = {str(d.get("StudyID", "")) for d in by_name["fused"]}
    if {str(d.get("StudyID", "")) for d in mask} != fused_study_ids:
        raise RuntimeError("mask StudyID differs from fused; the PACS would file them apart")
    if "reports" not in by_name and any(np.any(d.pixel_array) for d in mask):
        raise RuntimeError("Nonempty mask did not produce a report")
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
    if any(row["redcap_repeat_instrument"] != INSTRUMENT for row in rows):
        raise RuntimeError("Incorrect REDCap instrument")
    if len({(row["record_id"], row["redcap_event_name"]) for row in rows}) != 1:
        raise RuntimeError("Mixed REDCap destinations")
    # The values are patient identifiers, so the message does not show them.
    for field, expected in _redcap_destination(source).items():
        if rows[0][field] != expected:
            source_field = "PatientID" if field == "record_id" else "ReferringPhysicianName"
            raise RuntimeError(f"REDCap {field} differs from the input's {source_field}")
    instances = {}
    for row in rows:
        values = instances.setdefault(row["redcap_repeat_instance"], {})
        if row["field_name"] in values:
            raise RuntimeError("Duplicate REDCap field in one instance")
        values[row["field_name"]] = row["value"]
    resources = json.loads((Path(output_dir) / "runtime_resources.json").read_text())
    base = model_instance_base(resources["bundle_sha256"])
    if sorted(instances, key=int) != [str(base + k) for k in range(1, len(instances) + 1)]:
        raise RuntimeError(f"REDCap instances must be {base + 1}, {base + 2}, ...: {sorted(instances)}")
    with zipfile.ZipFile(paths[0].with_name("output_data_dictionary.zip")) as archive:
        dictionary = list(csv.DictReader(io.StringIO(archive.read("instrument.csv").decode("utf-8-sig"))))
    columns = [row["Variable / Field Name"] for row in dictionary]
    if any(list(values) != columns for values in instances.values()):
        raise RuntimeError("REDCap fields differ from the dictionary")
    if [c for c in columns if c != MODEL_VERSION_FIELD] != [
            row["Variable / Field Name"] for row in pr2mask_dictionary_rows()]:
        raise RuntimeError("pr2mask columns differ from pr2mask_dictionary.csv; regenerate it from this run")
    versions = {values[MODEL_VERSION_FIELD] for values in instances.values()}
    if len(versions) != 1 or not versions.pop():
        raise RuntimeError("Every REDCap instance needs the same model_version")
    # Each region of the published mask has its own instance.
    foreground = sum(int(np.count_nonzero(d.pixel_array)) for d in mask)
    measured = [values for values in instances.values()
                if any(v for k, v in values.items() if k != MODEL_VERSION_FIELD)]
    if foreground == 0:
        if measured or len(instances) != 1:
            raise RuntimeError("An empty mask must send one instance with blank pr2mask columns")
    elif (len(measured) != len(instances)
          or sum(int(values["number_of_pixel"]) for values in measured) != foreground):
        raise RuntimeError("REDCap regions do not add up to the published mask")

    print(
        "Research-PACS DICOM and REDCap validation: OK "
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
