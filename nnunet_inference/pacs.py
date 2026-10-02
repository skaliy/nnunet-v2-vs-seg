"""Research PACS inference and pr2mask/REDCap publication for ResEnc-M."""
import argparse
from datetime import datetime
import os
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .deployment import MODEL_TYPE, validate_deployment
from .pipeline import run_inference
from .redcap_output import _series, model_repeat_instance, write_redcap_mask

FINAL_OUTPUTS = ("mask", "fused", "fused_vote_map", "reports", "redcap")
LOG_NAME = "pacs_command.log"


def configure_cpu():
    cpus = sorted(os.sched_getaffinity(0))[:16]
    if len(cpus) < 15:
        raise RuntimeError(f"Five-fold inference requires at least 15 available CPU threads; got {len(cpus)}")
    os.sched_setaffinity(0, cpus)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[name] = "3"
    os.environ["nnUNet_def_n_proc"] = "3"
    os.environ["nnUNet_compile"] = "false"
    import torch
    import SimpleITK as sitk
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(3)
    print(f"CPU affinity: {cpus}; five workers x three threads", flush=True)


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


def postprocess(input_dir, work_dir, output_dir, deployment, *, version,
                use_tta, pr2mask_dir):
    identity = f"{version}_nn_m1_b{deployment['bundle_sha256'][:32]}_t{int(use_tta)}"
    info = f"{deployment['model_name']}, Predicted {datetime.now():%b%d%Y}"
    common = [str(input_dir), str(work_dir / "mask"), str(work_dir)]
    commands = [
        [str(pr2mask_dir / "imageAndMask2Report"), *common, "-u", identity + "_report",
         "-i", identity, "--reporttype", "mosaic", "-t", info + " "],
        [str(pr2mask_dir / "imageAndMask2Fused"), *common,
         "-u", identity + "_fused", "-i", identity],
        [str(pr2mask_dir / "imageAndMask2Fused"), str(input_dir), str(work_dir / "vote_map"),
         str(work_dir), "--votemapmax", "65535", "--votemapagree", "0.5",
         "-u", identity + "_votemap", "-s", "peak agreement {peak_agreement}", "-i", identity],
    ]
    log_path = work_dir / LOG_NAME
    with log_path.open("w") as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_redcap_mask(work_dir, input_dir, deployment, version=version, use_tta=use_tta)
    missing = [n for n in FINAL_OUTPUTS if not (work_dir / n).is_dir()]
    if missing:
        raise RuntimeError(f"Missing required PACS products: {missing}")
    for name in FINAL_OUTPUTS:
        shutil.copytree(work_dir / name, output_dir / name)
    shutil.copy2(log_path, output_dir / LOG_NAME)
    for path in output_dir.rglob("*"):
        path.chmod(path.stat().st_mode | (0o666 if path.is_file() else 0o777))


def run_pacs(input_dir, output_dir, *, model_dir="/app/model", version,
             use_tta=True, pr2mask_dir="/pr2mask"):
    input_dir, output_dir, pr2mask_dir = map(Path, (input_dir, output_dir, pr2mask_dir))
    if not version:
        raise RuntimeError("VERSION is required for PACS provenance")
    for name in ("imageAndMask2Report", "imageAndMask2Fused"):
        if not os.access(pr2mask_dir / name, os.X_OK):
            raise RuntimeError(f"Required pr2mask tool is unavailable: {name}")
    prepare_output(output_dir)
    validate_input(input_dir)
    deployment = validate_deployment(model_dir, verify_runtime=True)
    model_repeat_instance(deployment["bundle_sha256"])
    configure_cpu()
    with tempfile.TemporaryDirectory(prefix="nnunet-pacs-") as directory:
        work_dir = Path(directory)
        run_inference(input_dir, work_dir, model_dir=model_dir, device="cpu",
                      checkpoint=deployment["checkpoint"].removeprefix("checkpoint_").removesuffix(".pth"),
                      folds=tuple(deployment["folds"]), disable_tta=not use_tta,
                      parallel=True, deployment_version=version)
        postprocess(input_dir, work_dir, output_dir, deployment, version=version,
                    use_tta=use_tta, pr2mask_dir=pr2mask_dir)
    peak_path = Path("/sys/fs/cgroup/memory.peak")
    peak = int(peak_path.read_text()) if peak_path.exists() else None
    (output_dir / "runtime_resources.json").write_text(json.dumps(dict(
        available_cpu_ids=sorted(os.sched_getaffinity(0)), threads_per_fold=3,
        cgroup_memory_peak_bytes=peak, deployment_version=version,
        bundle_sha256=deployment["bundle_sha256"], tta=use_tta), indent=2) + "\n")
    return deployment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--model-dir", default="/app/model")
    parser.add_argument("--model-type", choices=(MODEL_TYPE,), default=MODEL_TYPE)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    run_pacs(args.input_dir, args.output_dir, model_dir=args.model_dir,
             version=os.environ.get("VERSION"), use_tta=args.tta)


if __name__ == "__main__":
    main()
