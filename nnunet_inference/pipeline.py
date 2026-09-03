#!/usr/bin/env python3
"""DICOM-in / DICOM-out nnU-Net v2 inference for vestibular schwannoma.

    python -m nnunet_inference <input_dir> [output_dir] --model-dir PATH
        [--device {auto,cpu,cuda}]
        [--checkpoint {final,best}] [--folds auto|all|0,1,2,3,4]
        [--tta | --no-tta]

Reads a DICOM T1 series, runs the trained nnU-Net, and writes two derived
DICOM series with deterministic numeric Series and SOP Instance UIDs.
"""
import argparse
import hashlib
import importlib.metadata
import tempfile
from pathlib import Path

import SimpleITK as sitk

from . import dicom_io
from .predictor import (
    build_predictor,
    predict,
    resolve_folds,
)

CHECKPOINTS = {"final": "checkpoint_final.pth", "best": "checkpoint_best.pth"}
CASE_ID = "VS"


def _resolve_model_version(model_dir, checkpoint_name, folds):
    """Resolve a content identity for UID/provenance generation.

    Deployment builds provide ``MODEL_VERSION.txt`` after hashing plans,
    dataset metadata, and all selected checkpoints. Local runs without that
    file compute the same kind of identity directly from the selected files.
    """
    model_dir = Path(model_dir)
    version_path = model_dir / "MODEL_VERSION.txt"
    try:
        version = version_path.read_text().strip()
    except OSError:
        version = ""
    if version:
        return version

    fold_dirs = ["fold_all" if fold == "all" else f"fold_{fold}"
                 for fold in folds]
    paths = [model_dir / "plans.json", model_dir / "dataset.json"] + [
        model_dir / fold_dir / checkpoint_name for fold_dir in fold_dirs
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"cannot identify model; missing {path}")
        digest.update(str(path.relative_to(model_dir)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_inference(input_dir, output_dir=None, *, model_dir, device="auto",
                  checkpoint="final", disable_tta=False, folds="auto"):
    """Full DICOM -> nnU-Net -> DICOM pipeline. Returns a summary dict.

    `output_dir` defaults to the input series' parent folder, so `mask/` and
    `vote_map/` are written alongside the input series (matching the PACS layout
    where input/mask/vote_map are siblings).
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir does not exist: {input_dir}")
    output_dir = Path(output_dir) if output_dir is not None else input_dir.parent
    model_dir = Path(model_dir)
    checkpoint_name = CHECKPOINTS[checkpoint]
    resolved_folds = resolve_folds(model_dir, checkpoint_name, folds)

    # The exact model/inference identity is both readable provenance and input
    # to deterministic DICOM UID generation.
    nnunet_ver = importlib.metadata.version("nnunetv2")
    model_version = _resolve_model_version(
        model_dir, checkpoint_name, resolved_folds
    )
    try:
        model_name = (Path(model_dir) / "MODEL_NAME.txt").read_text().strip()
    except OSError:
        model_name = ""
    if not model_name:
        dataset_dir = next(
            (part for part in reversed(Path(model_dir).parts)
             if part.startswith("Dataset")),
            "nnU-Net model",
        )
        dataset_id = dataset_dir.split("_", 1)[0]
        model_name = f"ResEnc-L {dataset_id}"
        if resolved_folds != ("all",):
            model_name += " folds " + ",".join(
                str(fold) for fold in resolved_folds
            )
    software_versions = [
        model_name, model_version[:8], f"nnUNetv2 {nnunet_ver}"
    ]
    uid_context = dicom_io.make_uid_context(
        model_name=model_name,
        model_version=model_version,
        checkpoint_name=checkpoint_name,
        folds=resolved_folds,
        tta=not disable_tta,
    )

    fold_label = "fold_all" if resolved_folds == ("all",) else ",".join(
        str(fold) for fold in resolved_folds
    )
    print(f"  model folds: {fold_label}; TTA: "
          f"{'off' if disable_tta else 'on'}")

    predictor = build_predictor(
        model_dir, device=device, checkpoint_name=checkpoint_name,
        disable_tta=disable_tta, folds=resolved_folds,
    )
    series = dicom_io.read_series(input_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        native = tmp / "native" / f"{CASE_ID}_0000.nii.gz"
        src = tmp / "src"
        pred = tmp / "pred"
        nnunet_in = src / f"{CASE_ID}_0000.nii.gz"

        # DICOM -> NIfTI in the scan's native orientation, then reorient to the
        # training orientation (LPS) for nnU-Net, which ignores the direction
        # matrix (a non-axial scan fed natively is silently segmented as empty).
        dicom_io.series_to_nifti(series, native)
        print(f"  input orientation {dicom_io.get_orientation(native)} "
              f"-> {dicom_io.NNUNET_ORIENTATION} for nnU-Net")
        dicom_io.reorient_nifti(native, nnunet_in, dicom_io.NNUNET_ORIENTATION)

        seg_lps, prob_npz = predict(predictor, src, pred)   # outputs in LPS
        prob_lps = tmp / f"{CASE_ID}_prob_lps.nii.gz"
        dicom_io.prob_npz_to_nifti(prob_npz, seg_lps, prob_lps)

        # map predictions back onto the input series' native grid for the DICOM
        # write (lossless reorientation; write_*_dicom reads them with imagedata)
        seg_nifti = tmp / f"{CASE_ID}_seg.nii.gz"
        prob_nifti = tmp / f"{CASE_ID}_prob.nii.gz"
        dicom_io.resample_to_reference(seg_lps, native, seg_nifti, is_mask=True)
        dicom_io.resample_to_reference(prob_lps, native, prob_nifti, is_mask=False)

        mask_dir = dicom_io.write_mask_dicom(
            seg_nifti, input_dir, output_dir / "mask", uid_context,
            software_versions=software_versions,
        )
        vote_map_dir = dicom_io.write_prob_dicom(
            prob_nifti, input_dir, output_dir / "vote_map", uid_context,
            software_versions=software_versions,
        )

        seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_nifti)))

    tumor_voxels = int((seg_arr > 0).sum())
    total_voxels = int(seg_arr.size)
    tumor_pct = 100.0 * tumor_voxels / total_voxels
    summary = {
        "mask_dir": mask_dir,
        "vote_map_dir": vote_map_dir,
        "tumor_voxels": tumor_voxels,
        "total_voxels": total_voxels,
        "tumor_pct": tumor_pct,
        "model_version": model_version,
        "uid_context": uid_context,
    }
    print("=" * 60)
    print("nnU-Net DICOM inference complete")
    print(f"  mask     -> {mask_dir}")
    print(f"  vote_map -> {vote_map_dir}")
    print(f"  tumor voxels: {tumor_voxels} / {total_voxels} "
          f"({tumor_pct:.4f}%)")
    print("=" * 60)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="nnunet-dicom-inference",
        description="DICOM-in / DICOM-out nnU-Net v2 VS segmentation.",
    )
    parser.add_argument("input_dir", help="Directory of the input DICOM series")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Directory for mask/ + vote_map/ output "
                             "(default: alongside the input series)")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=(
            "Trained nnU-Net model folder containing plans.json and fold directories"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", choices=["final", "best"], default="final")
    parser.add_argument(
        "--folds", default="auto", metavar="FOLDS",
        help="Model folds: auto, all, or comma-separated integers such as "
             "0,1,2,3,4 (default: auto)",
    )
    tta = parser.add_mutually_exclusive_group()
    tta.add_argument(
        "--tta", dest="disable_tta", action="store_false",
        help="Enable mirror test-time augmentation (default)",
    )
    tta.add_argument(
        "--no-tta", "--disable-tta", dest="disable_tta", action="store_true",
        help="Disable mirror TTA for faster inference",
    )
    parser.set_defaults(disable_tta=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_inference(
        args.input_dir, args.output_dir, model_dir=args.model_dir,
        device=args.device, checkpoint=args.checkpoint,
        disable_tta=args.disable_tta, folds=args.folds,
    )


if __name__ == "__main__":
    main()
