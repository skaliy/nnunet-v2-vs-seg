#!/usr/bin/env python3
"""End-to-end validation for the nnU-Net DICOM inference tool.

Validates that, given a DICOM series in the research-PACS format, the tool
produces a valid, PACS-ready round-trip: two derived DICOM series (mask +
vote_map) whose geometry matches the input and whose UIDs are correctly
derived so PACS accepts them as new derived objects.

The HARD gates are all DICOM-tool concerns (geometry, value ranges, UID
derivation) — they must pass. Segmentation *content* (how many tumor voxels
the model finds) is a property of the trained model, not the tool, so it is
reported as INFORMATION, not asserted: real research-PACS inputs include scans
the model legitimately segments as empty (e.g. small/atypical tumours), and
that must not fail the DICOM round-trip.

    NNUNET_DICOM_FIXTURE=/path/to/dicom-series \
    NNUNET_MODEL_DIR=/path/to/trained-model \
        python -m nnunet_inference.tests.validate_dicom

Orientation correctness is independently proven, model-free, by the marker
round-trip in test_dicom_io.TestOutputBridge and by byte-identity with the
nnUNetv2_predict CLI (see nnunet_inference/README.md).
"""
import os
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import UID
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nnunet_inference import dicom_io
from nnunet_inference import pipeline

HERE = Path(__file__).resolve().parent
INPUT_DIR = Path(os.environ.get(
    "NNUNET_DICOM_FIXTURE",
    HERE / "dicom_input",
))
OUTPUT_DIR = HERE / "output"
OVERLAY_PNG = HERE / "validation_overlay.png"


def _dicom_identity(series_dir, *, derived=False):
    """Collect DICOM identity and enforce the derived-file UID contract."""
    sops, series, studies = [], set(), set()
    for path in sorted(Path(series_dir).iterdir()):
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        sop_uid = str(ds.SOPInstanceUID)
        series_uid = str(ds.SeriesInstanceUID)
        sops.append(sop_uid)
        series.add(series_uid)
        studies.add(str(ds.StudyInstanceUID))
        if derived:
            assert UID(series_uid).is_valid and len(series_uid) <= 64, \
                f"invalid derived SeriesInstanceUID in {path}: {series_uid}"
            assert UID(sop_uid).is_valid and len(sop_uid) <= 64, \
                f"invalid derived SOPInstanceUID in {path}: {sop_uid}"
            assert str(ds.file_meta.MediaStorageSOPInstanceUID) == sop_uid, \
                f"dataset/file-meta SOP UID mismatch in {path}"
    return sops, series, studies


def _model_dir_from_env():
    value = os.environ.get("NNUNET_MODEL_DIR")
    if not value:
        raise SystemExit("Set NNUNET_MODEL_DIR to a trained nnU-Net model folder.")
    return Path(value).expanduser().resolve()


def main():
    if not INPUT_DIR.is_dir() or not any(INPUT_DIR.iterdir()):
        raise SystemExit(
            "No DICOM input found. Set NNUNET_DICOM_FIXTURE to a local "
            "DICOM series directory."
        )
    print(f"Running inference on research-PACS series {INPUT_DIR} ...")
    summary = pipeline.run_inference(
        INPUT_DIR, OUTPUT_DIR, model_dir=_model_dir_from_env()
    )

    in_series = dicom_io.read_series(INPUT_DIR)
    mask_series = dicom_io.read_series(summary["mask_dir"])
    prob_series = dicom_io.read_series(summary["vote_map_dir"])

    img = np.asarray(in_series)
    mask = np.asarray(mask_series)
    prob = np.asarray(prob_series)

    # ---- HARD gate 1: geometry round-trip matches the input grid ----
    assert mask.shape == img.shape, f"mask {mask.shape} != input {img.shape}"
    assert prob.shape == img.shape, f"prob {prob.shape} != input {img.shape}"
    assert mask_series.slices == in_series.slices, "slice count mismatch"
    for got, exp in zip(mask_series.spacing, in_series.spacing):
        assert abs(float(got) - float(exp)) < 1e-3, "spacing mismatch"

    # ---- HARD gate 2: voxel value contracts ----
    assert set(np.unique(mask)).issubset({0, 1}), "mask not binary {0,1}"
    assert prob.dtype == np.uint16, "vote_map not uint16"
    assert prob.max() <= 65535 and prob.min() >= 0, "prob out of uint16 range"

    # ---- HARD gate 3: PACS-ready derived UIDs ----
    in_sops, in_series_uid, in_study_uid = _dicom_identity(INPUT_DIR)
    mask_sops, mask_series_uid, mask_study_uid = _dicom_identity(
        summary["mask_dir"], derived=True
    )
    prob_sops, prob_series_uid, prob_study_uid = _dicom_identity(
        summary["vote_map_dir"], derived=True
    )
    n = in_series.slices
    assert len(mask_sops) == n and len(set(mask_sops)) == n, "mask SOP UIDs not unique-per-slice"
    assert len(prob_sops) == n and len(set(prob_sops)) == n, "vote_map SOP UIDs not unique-per-slice"
    (mu,) = mask_series_uid
    (pu,) = prob_series_uid
    assert mu.startswith("2.25.") and pu.startswith("2.25."), \
        "derived series UIDs must use the numeric 2.25 root"
    assert mu != pu and mu not in in_series_uid and pu not in in_series_uid, \
        "derived series UIDs must differ from each other and from the input"
    assert mask_study_uid == in_study_uid and prob_study_uid == in_study_uid, \
        "derived outputs must preserve the source StudyInstanceUID"
    # outputs must not reuse any input SOP UID (PACS would reject/overwrite)
    assert not (set(mask_sops) | set(prob_sops)) & set(in_sops), \
        "derived SOP UIDs collide with the input series"

    # model provenance overwrites the mask's SoftwareVersions tag, and the mask is
    # flagged as a derived segmentation via ImageType
    mask_ds = pydicom.dcmread(
        str(sorted(Path(summary["mask_dir"]).iterdir())[0]), stop_before_pixels=True
    )
    mask_sv = mask_ds.get("SoftwareVersions")
    mask_sv_str = mask_sv if isinstance(mask_sv, str) else " ".join(mask_sv or [])
    assert "nnUNetv2" in mask_sv_str, \
        f"mask SoftwareVersions missing nnU-Net provenance: {mask_sv!r}"
    assert summary["uid_context"]["model_name"] in mask_sv_str, \
        f"mask SoftwareVersions missing model identity: {mask_sv!r}"
    mask_it = list(mask_ds.get("ImageType") or [])
    assert mask_it[:2] == ["DERIVED", "SECONDARY"] and "MASK" in mask_it, \
        f"mask ImageType not flagged DERIVED/SECONDARY...MASK: {mask_it!r}"

    print("\nDICOM ROUND-TRIP / PACS-READY GATES PASSED")
    print(f"  geometry: mask & vote_map == input {img.shape}, spacing matched")
    print(f"  values:   mask in {{0,1}}, vote_map uint16 [0,65535]")
    print(f"  UIDs:     {n} unique numeric mask SOPs, "
          f"{n} unique numeric vote_map SOPs; file meta synchronized; "
          f"no collision with input")
    print(f"  provenance: mask SoftwareVersions = {mask_sv_str!r}")
    print(f"  imagetype:  mask ImageType = {mask_it!r}")

    # ---- HARD gate 4: the tool actually segments this scan ----
    # With correct input reorientation (native -> LPS) the model segments this
    # sagittal MPRAGE. An empty mask here means the orientation handling has
    # regressed: nnU-Net ignores the direction matrix, so a non-axial scan fed
    # in its native orientation is silently segmented as empty.
    tv, pct = summary["tumor_voxels"], summary["tumor_pct"]
    assert tv > 0, (
        "empty segmentation — input orientation handling has regressed "
        "(non-axial scan must be reoriented to LPS for nnU-Net)"
    )
    print(f"\nMODEL SEGMENTATION: tumor voxels {tv} ({pct:.4f}%)")

    # ---- overlay PNG at the slice with the most tumor ----
    z = int(np.argmax(mask.sum(axis=(1, 2))))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[z], cmap="gray")
    ax[0].imshow(np.ma.masked_where(mask[z] == 0, mask[z]), cmap="autumn", alpha=0.5)
    ax[0].set_title(f"mask overlay (z={z}, {tv} vox)")
    ax[1].imshow(img[z], cmap="gray")
    ax[1].imshow(np.ma.masked_where(prob[z] == 0, prob[z]), cmap="jet", alpha=0.5)
    ax[1].set_title("vote_map overlay")
    for a in ax:
        a.axis("off")
    fig.savefig(OVERLAY_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  overlay written: {OVERLAY_PNG}")


if __name__ == "__main__":
    main()
