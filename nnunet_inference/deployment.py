"""Stage and verify the curated ResEnc-M inference-only five-fold bundle."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

import torch

from . import dicom_io
from .predictor import FIVE_FOLDS, build_predictor

MODEL_TYPE = "nnunet_medium"
MODEL_NAME = "ResEnc-M Dataset003 folds 0,1,2,3,4"
DATASET = "Dataset003_VSSegmentationCurated"
PLANS = "nnUNetResEncUNetMPlans"
RUNTIME = {"nnunetv2": "2.6.2", "torch": "2.8.0+cpu",
           "dynamic-network-architectures": "0.4.2"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_sha256(manifest):
    payload = {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_plan(model_dir):
    plans = json.loads((Path(model_dir) / "plans.json").read_text())
    dataset = json.loads((Path(model_dir) / "dataset.json").read_text())
    if plans.get("dataset_name") != DATASET or plans.get("plans_name") != PLANS:
        raise ValueError("Expected curated Dataset003 ResEnc-M plans")
    config = plans["configurations"]["3d_fullres"]
    if (dataset.get("channel_names") != {"0": "T1"}
            or dataset.get("labels") != {"background": 0, "VS": 1}
            or dataset.get("numTraining") != 344
            or dataset.get("file_ending") != ".nii.gz"
            or config["patch_size"] != [48, 192, 192]
            or config["architecture"]["arch_kwargs"]["features_per_stage"] != [32, 64, 128, 256, 320, 320]
            or config["architecture"]["network_class_name"] != "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"):
        raise ValueError("Dataset or medium architecture does not match the deployment contract")
    return plans


def validate_deployment(model_dir, *, verify_files=True, verify_runtime=False):
    root = Path(model_dir)
    manifest = json.loads((root / "deployment_manifest.json").read_text())
    expected = dict(schema_version=2, model_type=MODEL_TYPE, model_name=MODEL_NAME,
                    dataset=DATASET, plans=PLANS, trainer="nnUNetTrainer",
                    configuration="3d_fullres", folds=list(FIVE_FOLDS),
                    runtime=RUNTIME, cpu=dict(max_threads=16, threads_per_fold=3,
                                           memory_limit_gib=64),
                    dicom_uid=dict(format_version=dicom_io.DICOM_UID_FORMAT_VERSION,
                                   root=dicom_io.DICOM_UID_ROOT,
                                   generation="uuid5", namespace_uuid=dicom_io.DICOM_UID_NAMESPACE,
                                   model_code=dicom_io.DICOM_MODEL_CODE,
                                   deployment_code=dicom_io.DICOM_DEPLOYMENT_CODE,
                                   output_codes=dicom_io.DICOM_OUTPUT_CODES))
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Deployment manifest does not match the medium model contract")
    checkpoint = manifest.get("checkpoint")
    if checkpoint not in ("checkpoint_best.pth", "checkpoint_final.pth"):
        raise ValueError("Unsupported deployment checkpoint")
    expected_files = {"plans.json", "dataset.json"} | {
        f"fold_{fold}/{checkpoint}" for fold in FIVE_FOLDS}
    if (set(manifest.get("files_sha256", {})) != expected_files
            or [m.get("member_id") for m in manifest.get("members", [])] != [f"fold_{f}" for f in FIVE_FOLDS]
            or {p.name for p in root.glob("fold_*")} != {f"fold_{f}" for f in FIVE_FOLDS}):
        raise ValueError("Expected exactly five declared numeric folds and model files")
    if manifest.get("bundle_sha256") != bundle_sha256(manifest):
        raise ValueError("Deployment manifest checksum mismatch")
    if ((root / "MODEL_VERSION.txt").read_text().strip() != manifest["bundle_sha256"]
            or (root / "MODEL_NAME.txt").read_text().strip() != MODEL_NAME
            or (root / "MODEL_FOLDS.txt").read_text().strip() != "0,1,2,3,4"):
        raise ValueError("Model identity files differ from manifest")
    validate_plan(root)
    for member in manifest["members"]:
        if member.get("sha256") != manifest["files_sha256"][f"{member['member_id']}/{checkpoint}"]:
            raise ValueError("Member hash differs from declared file hash")
    if verify_files:
        for name, digest in manifest["files_sha256"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"Packaged model file checksum mismatch: {name}")
    if verify_runtime:
        for package, version in RUNTIME.items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Expected {package}=={version}")
    return manifest


def prepare_bundle(source, output, checkpoint="best"):
    source, output = Path(source), Path(output)
    checkpoint_name = f"checkpoint_{checkpoint}.pth"
    plans = validate_plan(source)
    if checkpoint not in ("best", "final"):
        raise ValueError("Checkpoint must be best or final")
    for fold in FIVE_FOLDS:
        if not (source / f"fold_{fold}" / checkpoint_name).is_file():
            raise FileNotFoundError(f"Missing trained fold_{fold}/{checkpoint_name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Bundle already exists: {output}")
    torch.set_num_threads(3)
    with tempfile.TemporaryDirectory(prefix=".medium-build-", dir=output.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        for name in ("plans.json", "dataset.json"):
            shutil.copy2(source / name, stage / name)
        members = []
        for fold in FIVE_FOLDS:
            source_path = source / f"fold_{fold}" / checkpoint_name
            trained = torch.load(source_path, map_location="cpu", weights_only=False)
            if (trained.get("trainer_name") != "nnUNetTrainer"
                    or trained["init_args"].get("configuration") != "3d_fullres"
                    or trained["init_args"].get("fold") != fold):
                raise ValueError(f"Checkpoint identity differs for fold_{fold}")
            weights = trained["network_weights"]
            if any(not torch.isfinite(t).all() for t in weights.values() if t.is_floating_point()):
                raise ValueError(f"Nonfinite weights in fold_{fold}")
            # Native nnU-Net needs these four entries only. Keep original weights
            # and mirror axes; omit optimizer/training state and source paths.
            artifact = dict(network_weights=weights, trainer_name=trained["trainer_name"],
                            init_args={"configuration": "3d_fullres"},
                            inference_allowed_mirroring_axes=trained.get("inference_allowed_mirroring_axes"))
            path = stage / f"fold_{fold}" / checkpoint_name
            path.parent.mkdir()
            torch.save(artifact, path)
            members.append(dict(member_id=f"fold_{fold}", sha256=file_sha256(path),
                                source_checkpoint_sha256=file_sha256(source_path),
                                epoch=int(trained["current_epoch"])))
            del artifact, weights, trained
            predictor = build_predictor(stage, device="cpu", checkpoint_name=checkpoint_name, folds=(fold,))
            print(f"fold_{fold}: strict load OK", flush=True)
            del predictor
        files = {name: file_sha256(stage / name) for name in
                 ["plans.json", "dataset.json"] + [f"fold_{f}/{checkpoint_name}" for f in FIVE_FOLDS]}
        manifest = dict(schema_version=2, model_type=MODEL_TYPE, model_name=MODEL_NAME,
                        dataset=DATASET, plans=PLANS, trainer="nnUNetTrainer",
                        configuration="3d_fullres", checkpoint=checkpoint_name,
                        folds=list(FIVE_FOLDS), runtime=RUNTIME,
                        cpu=dict(max_threads=16, threads_per_fold=3, memory_limit_gib=64),
                        dicom_uid=dict(format_version=dicom_io.DICOM_UID_FORMAT_VERSION,
                                       root=dicom_io.DICOM_UID_ROOT, generation="uuid5",
                                       namespace_uuid=dicom_io.DICOM_UID_NAMESPACE,
                                       model_code=dicom_io.DICOM_MODEL_CODE,
                                       deployment_code=dicom_io.DICOM_DEPLOYMENT_CODE,
                                       output_codes=dicom_io.DICOM_OUTPUT_CODES),
                        members=members, files_sha256=files)
        manifest["bundle_sha256"] = bundle_sha256(manifest)
        (stage / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "MODEL_VERSION.txt").write_text(manifest["bundle_sha256"] + "\n")
        (stage / "MODEL_NAME.txt").write_text(MODEL_NAME + "\n")
        (stage / "MODEL_FOLDS.txt").write_text("0,1,2,3,4\n")
        (stage / "MODEL_SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in files.items()))
        validate_deployment(stage)
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = validate_deployment(args.output, verify_runtime=args.verify_runtime)
    else:
        if args.source is None:
            parser.error("--source is required to prepare a bundle")
        manifest = prepare_bundle(args.source, args.output, args.checkpoint)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
