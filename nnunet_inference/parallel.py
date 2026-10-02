"""CPU-parallel nnU-Net folds with upstream logit aggregation and disk buffers."""
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

THREADS_PER_FOLD = 3
CPU_BUDGET = 16


def assign_worker_cpus(count, available=None):
    available = sorted(os.sched_getaffinity(0) if available is None else available)
    available = available[:CPU_BUDGET]
    required = count * THREADS_PER_FOLD
    if len(available) < required:
        raise RuntimeError(f"{count} folds require {required} available CPU threads; "
                           f"affinity exposes {len(available)}")
    return [available[i * THREADS_PER_FOLD:(i + 1) * THREADS_PER_FOLD]
            for i in range(count)]


def _fold_worker(model_dir, checkpoint_name, fold, use_tta, cpus, input_path,
                 output_path, error_path):
    try:
        os.sched_setaffinity(0, cpus)
        torch.set_num_threads(THREADS_PER_FOLD)
        torch.set_num_interop_threads(1)
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(THREADS_PER_FOLD)
        from .predictor import build_predictor
        predictor = build_predictor(model_dir, device="cpu", folds=(fold,),
                                    checkpoint_name=checkpoint_name,
                                    disable_tta=not use_tta)
        predictor.allow_tqdm = False
        data = np.load(input_path, mmap_mode="c", allow_pickle=False)
        print(f"fold_{fold}: CPUs {cpus}, {THREADS_PER_FOLD} threads, "
              f"TTA={'on' if use_tta else 'off'}, shape={tuple(data.shape)}", flush=True)
        started = time.monotonic()
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))
        np.save(output_path, logits.cpu().numpy(), allow_pickle=False)
        print(f"fold_{fold}: completed in {time.monotonic() - started:.1f}s", flush=True)
    except BaseException:
        Path(error_path).write_text(traceback.format_exc())
        raise


def average_fold_logits(paths):
    """Add in fold order and original dtype, exactly as nnU-Net 2.6.2 does."""
    prediction = None
    for path in paths:
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        logits = torch.from_numpy(array)
        if prediction is None:
            prediction = logits.clone()
        else:
            if prediction.shape != logits.shape or prediction.dtype != logits.dtype:
                raise RuntimeError("Fold logit shapes/dtypes differ")
            prediction += logits
    if prediction is None:
        raise RuntimeError("No fold predictions")
    prediction /= len(paths)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Nonfinite ensemble logits")
    return prediction


class ParallelFoldMixin:
    """Replace only fold evaluation; keep native preprocessing, export and TTA."""
    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data):
        allocations = assign_worker_cpus(len(self.parallel_folds))
        context = mp.get_context("spawn")
        # Children import torch before entering the worker. Bound those imports
        # as well; large tensors never travel through a Queue or /dev/shm.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
            os.environ[name] = str(THREADS_PER_FOLD)
        os.environ["nnUNet_def_n_proc"] = str(THREADS_PER_FOLD)
        os.environ["nnUNet_compile"] = "false"
        with tempfile.TemporaryDirectory(prefix="nnunet-folds-") as directory:
            root = Path(directory)
            input_path = root / "input.npy"
            np.save(input_path, data.detach().cpu().numpy(), allow_pickle=False)
            workers, paths = [], []
            try:
                for fold, cpus in zip(self.parallel_folds, allocations):
                    output = root / f"fold_{fold}.npy"
                    error = root / f"fold_{fold}.error"
                    process = context.Process(target=_fold_worker, args=(
                        self.parallel_model_dir, self.parallel_checkpoint_name, fold,
                        self.use_mirroring, cpus, input_path, output, error))
                    process.start()
                    workers.append((process, fold, error))
                    paths.append(output)
                while any(p.is_alive() for p, _, _ in workers):
                    for process, fold, error in workers:
                        if error.exists() or process.exitcode not in (None, 0):
                            detail = error.read_text() if error.exists() else f"exit code {process.exitcode}"
                            raise RuntimeError(f"fold_{fold} failed: {detail}")
                    time.sleep(0.2)
                for process, fold, error in workers:
                    process.join()
                    if process.exitcode != 0:
                        detail = error.read_text() if error.exists() else f"exit code {process.exitcode}"
                        raise RuntimeError(f"fold_{fold} failed: {detail}")
                return average_fold_logits(paths)
            finally:
                for process, _, _ in workers:
                    if process.is_alive():
                        process.terminate()
                for process, _, _ in workers:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
