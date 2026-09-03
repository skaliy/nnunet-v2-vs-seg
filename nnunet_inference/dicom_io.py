"""DICOM <-> NIfTI I/O for nnU-Net VS segmentation inference.

Read a DICOM MR series, bridge to NIfTI for nnU-Net, and write nnU-Net's NIfTI
outputs back to DICOM using the input series as a geometry template.

Geometry contract (empirically verified): nnU-Net's prediction NIfTI is read
back with imagedata (NOT SimpleITK — imagedata and SimpleITK disagree on voxel
axis order, so a SimpleITK read would z-flip the result) and assigned into a
fresh imagedata template built from the input series, which preserves physical
orientation with no transpose. See _write_dicom_from_nifti for the rationale.
"""
import json
import uuid
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from imagedata.series import Series
from pydicom import dcmread
from pydicom.uid import UID

SLICE_TOLERANCE = 1e-2

# nnU-Net ignores the NIfTI direction matrix and assumes every image shares the
# training data's voxel storage orientation. The VS models were trained from
# images stored in LPS, so any inference input must be
# reoriented to LPS before prediction (and the result reoriented back to the
# input's native orientation for the DICOM write). Without this, a non-axial
# acquisition (e.g. a sagittal MPRAGE, stored PSR) is fed to the model in an
# orientation it never saw and is silently segmented as empty.
NNUNET_ORIENTATION = "LPS"

# Persistent Research-PACS UID registry for this nnU-Net deployment. Never
# reuse these codes or this namespace for a different meaning. Readable model
# names belong in SeriesDescription/SoftwareVersions, not in a DICOM UID.
DICOM_UID_FORMAT_VERSION = 1
DICOM_UID_ROOT = "2.25"
DICOM_UID_NAMESPACE = "bb88c59d-5a75-5f47-bb52-3bc9f6db7808"
DICOM_MODEL_CODE = 1                 # nnU-Net VS ResEnc-L
DICOM_DEPLOYMENT_CODE = 1            # nnU-Net fold ensemble/single-fold runtime
DICOM_OUTPUT_CODES = {
    "segmentation": 1,
    "probability": 2,
}


def read_series(input_dir):
    """Read a single DICOM series as an imagedata Series (z, y, x)."""
    return Series(str(input_dir), opts={"slice_tolerance": SLICE_TOLERANCE})


def get_orientation(nifti_path):
    """Return the 3-letter anatomical orientation code (e.g. 'LPS', 'PSR') of a
    NIfTI's voxel storage order, derived from its direction cosines."""
    img = sitk.ReadImage(str(nifti_path))
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection()
    )


def reorient_nifti(in_path, out_path, target_orientation):
    """Reorient a NIfTI to `target_orientation` (lossless voxel permutation/flip
    that preserves physical position) and write it to `out_path`."""
    img = sitk.ReadImage(str(in_path))
    reoriented = sitk.DICOMOrient(img, target_orientation)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(reoriented, str(out_path))


def resample_to_reference(in_path, reference_path, out_path, is_mask):
    """Resample `in_path` onto `reference_path`'s grid (size/spacing/origin/
    direction) and write to `out_path`. Used to map nnU-Net's LPS prediction back
    onto the input series' native grid before the DICOM write. Because that is a
    lossless reorientation (the grids cover the same physical voxels), nearest-
    neighbour is exact for the mask; linear is used for the probability map.
    """
    img = sitk.ReadImage(str(in_path))
    ref = sitk.ReadImage(str(reference_path))
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    out = sitk.Resample(img, ref, sitk.Transform(), interpolator, 0.0,
                        img.GetPixelID())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path))


def series_to_nifti(series, nifti_path):
    """Write `series` to a single NIfTI file at exactly `nifti_path`.

    `nifti_path` must end in nnU-Net's channel-0 suffix `_0000.nii.gz`;
    imagedata writes the file named exactly as the given path.
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.name.endswith("_0000.nii.gz"):
        raise ValueError(
            f"nnU-Net needs a *_0000.nii.gz channel file, got {nifti_path.name!r}"
        )
    nifti_path.parent.mkdir(parents=True, exist_ok=True)
    series.write(str(nifti_path), formats=["nifti"])


def make_uid_context(model_name, model_version, checkpoint_name, folds, tta):
    """Return the immutable inference identity used to derive DICOM UIDs.

    Any option that can change output pixels is included. ``model_version`` is
    the content hash written by the deployment build, or a locally computed
    model hash when running outside the container.
    """
    model_version = str(model_version).strip()
    if not model_version or model_version == "unknown":
        raise ValueError("model_version must identify the exact model content")
    return {
        "model_name": str(model_name),
        "model_version": model_version,
        "checkpoint_name": str(checkpoint_name),
        "folds": [str(fold) for fold in folds],
        "tta": bool(tta),
    }


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def make_derived_dicom_uid(uid_context, source_series_uid, output_kind, *,
                           source_sop_uid=None, slice_index=None):
    """Create a deterministic, standards-valid ``2.25.<UUID integer>`` UID."""
    try:
        output_code = DICOM_OUTPUT_CODES[output_kind]
    except KeyError as exc:
        raise ValueError(f"unknown DICOM output kind: {output_kind!r}") from exc
    if (source_sop_uid is None) != (slice_index is None):
        raise ValueError("source_sop_uid and slice_index must be provided together")

    identity = {
        "format_version": DICOM_UID_FORMAT_VERSION,
        "model_code": DICOM_MODEL_CODE,
        "deployment_code": DICOM_DEPLOYMENT_CODE,
        "output_code": output_code,
        "inference": uid_context,
        "source_series_uid": str(source_series_uid),
        "scope": "instance" if source_sop_uid is not None else "series",
    }
    if source_sop_uid is not None:
        identity.update({
            "source_sop_uid": str(source_sop_uid),
            "slice_index": int(slice_index),
        })
    generated = uuid.uuid5(
        uuid.UUID(DICOM_UID_NAMESPACE), _canonical_json(identity)
    )
    uid = f"{DICOM_UID_ROOT}.{generated.int}"
    if len(uid) > 64 or not UID(uid).is_valid:
        raise RuntimeError(f"generated invalid DICOM UID: {uid}")
    return uid


def _source_sop_uids(series_obj):
    """Return source SOP identity in imagedata slice order."""
    values = getattr(series_obj, "SOPInstanceUIDs", None)
    if isinstance(values, dict):
        ordered = []
        for slice_idx in range(series_obj.slices):
            value = values.get((0, slice_idx), values.get(slice_idx))
            ordered.append(
                str(value) if value is not None else f"missing-sop-{slice_idx}"
            )
        return ordered
    base = series_obj.getDicomAttribute("SOPInstanceUID")
    return [f"{base}:slice-{slice_idx}" for slice_idx in range(series_obj.slices)]


def _finalize_written_dicom(save_dir, remove_rescale=False):
    """Validate written UIDs and synchronize dataset/file-meta SOP identity."""
    paths = sorted(path for path in Path(save_dir).iterdir() if path.is_file())
    if not paths:
        raise RuntimeError(f"DICOM writer produced no files in {save_dir}")
    for path in paths:
        dataset = dcmread(str(path))
        series_uid = str(dataset.SeriesInstanceUID)
        sop_uid = str(dataset.SOPInstanceUID)
        if not UID(series_uid).is_valid or not UID(sop_uid).is_valid:
            raise RuntimeError(f"writer produced an invalid DICOM UID in {path}")
        dataset.file_meta.MediaStorageSOPInstanceUID = sop_uid
        if remove_rescale:
            for keyword in ("RescaleSlope", "RescaleIntercept", "RescaleType"):
                if keyword in dataset:
                    del dataset[keyword]
        dataset.save_as(str(path), write_like_original=False)


def save_series_pred(series_obj, save_dir, uid_context, output_kind,
                     remove_rescale=False):
    """Write a derived series with deterministic numeric Series/SOP UIDs."""
    source_series_uid = str(series_obj.seriesInstanceUID)
    source_sop_uids = _source_sop_uids(series_obj)
    series_uid = make_derived_dicom_uid(
        uid_context, source_series_uid, output_kind
    )
    series_obj.seriesInstanceUID = series_uid
    series_obj.setDicomAttribute("SeriesInstanceUID", series_uid)
    if getattr(series_obj, "patientID", None):
        series_obj.studyID = (
            series_obj.patientID[3:]
            if len(series_obj.patientID) > 3
            else series_obj.patientID
        )
    for slice_idx in range(series_obj.slices):
        sop_uid = make_derived_dicom_uid(
            uid_context,
            source_series_uid,
            output_kind,
            source_sop_uid=source_sop_uids[slice_idx],
            slice_index=slice_idx,
        )
        series_obj.setDicomAttribute("SOPInstanceUID", sop_uid, slice=slice_idx)
    series_obj.write(save_dir, opts={"keep_uid": True}, formats=["dicom"])
    # imagedata can retain the source SOP UID in file meta even after replacing
    # the dataset SOP UID, so repair and validate the files after writing.
    _finalize_written_dicom(save_dir, remove_rescale=remove_rescale)


def _write_dicom_from_nifti(nifti_path, template_dir, out_dir, output_kind,
                            uid_context, to_voxels, software_versions=None):
    """Read a NIfTI, map its voxels, and write a derived DICOM series.

    The NIfTI is read with imagedata, not SimpleITK: the libraries disagree on
    voxel axis order. The input must already be on the source DICOM grid. Both
    outputs are marked derived and carry readable provenance; UIDs remain
    numeric and opaque.
    """
    arr = np.asarray(Series(str(nifti_path)))
    voxels = to_voxels(arr)
    template = read_series(template_dir)
    if software_versions:
        template.setDicomAttribute("SoftwareVersions", list(software_versions))
    marker = "MASK" if output_kind == "segmentation" else "PROBABILITY"
    image_type = template.getDicomAttribute("ImageType")
    image_type = [] if image_type is None else (
        [image_type] if isinstance(image_type, str) else list(image_type))
    template.setDicomAttribute(
        "ImageType", ["DERIVED", "SECONDARY"] + image_type[2:] + [marker]
    )
    model_name = uid_context["model_name"]
    output_label = ("segmentation mask" if output_kind == "segmentation"
                    else "foreground probability")
    template.setDicomAttribute(
        "SeriesDescription", f"nnU-Net {model_name} {output_label}"[:64]
    )
    derivation = (
        f"nnU-Net {output_label}; model_version={uid_context['model_version']}; "
        f"checkpoint={uid_context['checkpoint_name']}; "
        f"folds={','.join(uid_context['folds'])}; "
        f"tta={'on' if uid_context['tta'] else 'off'}"
    )
    if output_kind == "probability":
        derivation += "; foreground probability = stored uint16 value / 65535"
    template.setDicomAttribute("DerivationDescription", derivation)
    template[:] = voxels
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_series_pred(
        template, str(out_dir), uid_context, output_kind,
        remove_rescale=output_kind == "probability",
    )
    return out_dir


def write_mask_dicom(mask_nifti, template_dir, out_dir, uid_context,
                     software_versions=None):
    """Binary label NIfTI -> derived DICOM mask series in ``{0, 1}``."""
    return _write_dicom_from_nifti(
        mask_nifti, template_dir, out_dir, "segmentation", uid_context,
        lambda a: (a > 0).astype(np.uint16),
        software_versions=software_versions,
    )


def write_prob_dicom(prob_nifti, template_dir, out_dir, uid_context,
                     software_versions=None):
    """Foreground probability -> derived uint16 DICOM vote-map series."""
    return _write_dicom_from_nifti(
        prob_nifti, template_dir, out_dir, "probability", uid_context,
        lambda a: np.rint(np.clip(a, 0.0, 1.0) * 65535).astype(np.uint16),
        software_versions=software_versions,
    )

def prob_npz_to_nifti(npz_path, seg_nifti_path, out_nifti_path, fg_channel=1):
    """Extract the foreground softmax from nnU-Net's .npz and save it as a NIfTI
    that carries the segmentation NIfTI's geometry.

    nnU-Net saves probabilities under key 'probabilities' as (C, z, y, x), the
    same spatial axis order as the seg array. Geometry is copied from the seg
    NIfTI so the probability NIfTI is grid-identical to the segmentation.
    """
    prob = np.load(str(npz_path))["probabilities"]
    fg = np.ascontiguousarray(prob[fg_channel].astype(np.float32))  # (z, y, x)
    seg_img = sitk.ReadImage(str(seg_nifti_path))
    prob_img = sitk.GetImageFromArray(fg)
    prob_img.CopyInformation(seg_img)
    Path(out_nifti_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(prob_img, str(out_nifti_path))
