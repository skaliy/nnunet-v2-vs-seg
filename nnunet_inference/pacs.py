"""Research PACS inference and pr2mask/REDCap publication for ResEnc-L."""
import argparse
from datetime import datetime
import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .cpu import MAX_THREADS, THREADS_PER_FOLD, set_thread_environment
from .deployment import MODELS, validate_deployment
from .dicom_io import tag_pr2mask_mask
from .pipeline import run_inference
from .predictor import FIVE_FOLDS
from .redcap_output import _series, model_instance_base, write_redcap_output

FINAL_OUTPUTS = ("labels", "fused", "fused_vote_map", "reports", "redcap")
LOG_NAME = "pacs_command.log"
# pr2mask fits about 40 title characters on a fused image, so long model types
# get a shorter visible label.
TITLE_LABELS = {"nnunet_large": "nnunet_l"}


def configure_cpu():
    required = len(FIVE_FOLDS) * THREADS_PER_FOLD
    cpus = sorted(os.sched_getaffinity(0))[:MAX_THREADS]
    if len(cpus) < required:
        raise RuntimeError(f"Five-fold inference requires at least {required} available CPU threads; got {len(cpus)}")
    os.sched_setaffinity(0, cpus)
    set_thread_environment()
    import torch
    import SimpleITK as sitk
    torch.set_num_threads(THREADS_PER_FOLD)
    torch.set_num_interop_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(THREADS_PER_FOLD)
    print(f"CPU affinity: {cpus}; {len(FIVE_FOLDS)} workers x {THREADS_PER_FOLD} threads", flush=True)


def prepare_output(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collisions = [n for n in FINAL_OUTPUTS if os.path.lexists(output_dir / n)]
    if collisions:
        raise RuntimeError(f"Owned output directories already exist: {collisions}")


def validate_input(input_dir):
    source = _series(input_dir)
    if any(str(d.get("Modality", "")) != "MR" for d in source):
        raise ValueError("Expected one MR input series")
    sops = [str(d.get("SOPInstanceUID", "")) for d in source]
    if not all(sops) or len(set(sops)) != len(sops):
        raise ValueError("Missing or duplicate input SOP identities")
    if not source[0].get("StudyInstanceUID") or not source[0].get("SeriesInstanceUID"):
        raise ValueError("Missing input Study/Series identity")


def publish(work_dir, output_dir, log_path):
    """Move the finished products into output_dir: all of them or none.

    They are copied to a hidden folder inside output_dir first. That folder is
    on the same file system, so each product then appears complete, by a
    rename. A failure removes the copies and anything already renamed.
    """
    stage = Path(tempfile.mkdtemp(prefix=".nnunet-publishing-", dir=output_dir))
    published = []
    try:
        for name in FINAL_OUTPUTS:
            shutil.copytree(work_dir / name, stage / name)
        shutil.copy2(log_path, stage / LOG_NAME)
        for path in stage.rglob("*"):
            path.chmod(path.stat().st_mode | (0o666 if path.is_file() else 0o777))
        for name in (*FINAL_OUTPUTS, LOG_NAME):
            os.rename(stage / name, output_dir / name)
            published.append(output_dir / name)
        stage.rmdir()
    except BaseException:
        for path in published:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        raise


def model_version(deployment, container):
    """REDCap model_version, e.g. 'nnunet_large 5-fold, nnunet-vsseg:20261111T090000Z'."""
    count = len(deployment["members"])
    members = f"{count}-fold" if count > 1 else "single model"
    return f"{deployment['model_type']} {members}, {container}"


def postprocess(input_dir, work_dir, output_dir, deployment, *, version, container,
                use_tta, pr2mask_dir):
    # The image version pins the bundle, so version, model and TTA keep pr2mask's
    # stable series UIDs distinct.
    tta = "tta" if use_tta else "no-tta"
    identity = f"{version}_{deployment['model_type']}_{tta}"
    report_identity = f"{version}_{tta}"  # the report title already names the model
    label = TITLE_LABELS.get(deployment["model_type"], deployment["model_type"])
    info = f"{label} {len(deployment['members'])}-model ensemble, {datetime.now():%b%d%Y}"
    common = [str(input_dir), str(work_dir / "mask"), str(work_dir)]
    commands = [
        [str(pr2mask_dir / "imageAndMask2Report"), *common, "-u", identity + "_report",
         "-i", report_identity, "--reporttype", "mosaic", "-t", info + " "],
        [str(pr2mask_dir / "imageAndMask2Fused"), *common,
         "-u", identity + "_fused", "-t", info + " "],
        [str(pr2mask_dir / "imageAndMask2Fused"), str(input_dir), str(work_dir / "vote_map"),
         str(work_dir), "--votemapmax", "65535", "--votemapagree", "0.5",
         "-u", identity + "_votemap", "-s", "peak agreement {peak_agreement}", "-t", info + " "],
    ]
    log_path = work_dir / LOG_NAME
    with log_path.open("w") as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    # pr2mask's labels/ series is the published mask; it only gets our provenance.
    labels = sorted(path for path in (work_dir / "labels").glob("*") if path.is_dir())
    if len(labels) != 1:
        raise RuntimeError(
            f"pr2mask must create exactly one mask series in {work_dir / 'labels'}; "
            f"found {len(labels)}")
    tag_pr2mask_mask(labels[0], work_dir / "mask")
    write_redcap_output(work_dir, input_dir, deployment,
                        model_version=model_version(deployment, container))
    missing = [n for n in FINAL_OUTPUTS if not (work_dir / n).is_dir()]
    if missing:
        raise RuntimeError(f"Missing required PACS products: {missing}")
    publish(work_dir, output_dir, log_path)


def run_pacs(input_dir, output_dir, *, model_dir="/app/model", version, container,
             use_tta=True, pr2mask_dir="/pr2mask", model_type=None):
    input_dir, output_dir, pr2mask_dir = map(Path, (input_dir, output_dir, pr2mask_dir))
    if not version or not container:
        raise RuntimeError("VERSION and CONTAINER_VERSION are required for PACS provenance")
    for name in ("imageAndMask2Report", "imageAndMask2Fused"):
        if not os.access(pr2mask_dir / name, os.X_OK):
            raise RuntimeError(f"Required pr2mask tool is unavailable: {name}")
    prepare_output(output_dir)
    validate_input(input_dir)
    deployment = validate_deployment(model_dir, verify_runtime=True)
    if model_type is not None and model_type != deployment["model_type"]:
        raise RuntimeError(f"Requested {model_type}, but this image packages {deployment['model_type']}")
    model_instance_base(deployment["bundle_sha256"])
    configure_cpu()
    with tempfile.TemporaryDirectory(prefix="nnunet-pacs-") as directory:
        work_dir = Path(directory)
        run_inference(input_dir, work_dir, model_dir=model_dir, device="cpu",
                      checkpoint=deployment["checkpoint"].removeprefix("checkpoint_").removesuffix(".pth"),
                      folds=tuple(deployment["folds"]), disable_tta=not use_tta,
                      parallel=True, deployment_version=version)
        postprocess(input_dir, work_dir, output_dir, deployment, version=version,
                    container=container, use_tta=use_tta, pr2mask_dir=pr2mask_dir)
    peak_path = Path("/sys/fs/cgroup/memory.peak")
    peak = int(peak_path.read_text()) if peak_path.exists() else None
    resources = output_dir / ".runtime_resources.json.writing"
    resources.write_text(json.dumps(dict(
        available_cpu_ids=sorted(os.sched_getaffinity(0)), threads_per_fold=THREADS_PER_FOLD,
        cgroup_memory_peak_bytes=peak, deployment_version=version,
        bundle_sha256=deployment["bundle_sha256"], tta=use_tta), indent=2) + "\n")
    os.replace(resources, output_dir / "runtime_resources.json")
    return deployment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--model-dir", default="/app/model")
    parser.add_argument("--model-type", choices=tuple(MODELS),
                        help="must match the packaged model; defaults to it")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    run_pacs(args.input_dir, args.output_dir, model_dir=args.model_dir,
             version=os.environ.get("VERSION"), container=os.environ.get("CONTAINER_VERSION"),
             use_tta=args.tta, model_type=args.model_type)


if __name__ == "__main__":
    main()
