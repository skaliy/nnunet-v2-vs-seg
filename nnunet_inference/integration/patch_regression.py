"""Compare native sequential/parallel fold logits on a real full-sized patch."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch

from nnunet_inference.deployment import validate_deployment
from nnunet_inference.pacs import configure_cpu
from nnunet_inference.predictor import build_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/app/model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--tta", choices=("both", "true", "false"), default="both")
    args = parser.parse_args()
    manifest = validate_deployment(args.model_dir, verify_runtime=True)
    configure_cpu()
    folds = tuple(int(f) for f in args.folds.split(","))
    conditions = (False, True) if args.tta == "both" else (args.tta == "true",)
    results = []
    for tta in conditions:
        native = build_predictor(args.model_dir, device="cpu", checkpoint_name=manifest["checkpoint"],
                                 folds=folds, disable_tta=not tta)
        native.allow_tqdm = False
        shape = (1, *native.configuration_manager.patch_size)
        patch = torch.randn(shape, generator=torch.Generator().manual_seed(71))
        started = time.monotonic()
        expected = native.predict_logits_from_preprocessed_data(patch)
        sequential_seconds = time.monotonic() - started
        del native
        gc.collect()
        parallel = build_predictor(args.model_dir, device="cpu", checkpoint_name=manifest["checkpoint"],
                                   folds=folds, disable_tta=not tta, parallel=True)
        started = time.monotonic()
        actual = parallel.predict_logits_from_preprocessed_data(patch)
        parallel_seconds = time.monotonic() - started
        difference = float((actual.float() - expected.float()).abs().max())
        equal = torch.equal(actual, expected)
        masks_equal = torch.equal(actual.argmax(0), expected.argmax(0))
        row = dict(tta=tta, folds=list(folds), patch_shape=list(shape),
                   max_logit_difference=difference, bitwise_equal=equal,
                   masks_equal=masks_equal, sequential_seconds=sequential_seconds,
                   parallel_seconds=parallel_seconds)
        print(json.dumps(row), flush=True)
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        if not equal or not masks_equal:
            raise RuntimeError("Parallel fold aggregation differs from native nnUNet")
        del parallel, patch, expected, actual
        gc.collect()


if __name__ == "__main__":
    main()
