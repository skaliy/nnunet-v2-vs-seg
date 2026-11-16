# Checks that VS still works as released

Run these on the workstation that holds the trained model, Dataset003, the source
images and Docker. They are skipped everywhere else. A change to shared code is done
only when all of them pass.

## Model, dataset and metrics

From the repository root, about 12 minutes:

```bash
VS_CHECKS=1 python -m unittest discover -s projects/vs/checks -p 'test_*.py' -v
```

| Test | Passes when |
| --- | --- |
| `test_staging_the_model_gives_the_registered_bundle` | Staging the ResEnc-L folds gives bundle `e91f17a3…0e79cc`, the one registered for REDCap block 100 |
| `test_prepare_reproduces_dataset003` | A rebuild has the same `dataset.json`, `splits_final.json` and `component_counts.csv` byte for byte, the same image files and the same label voxels |
| `test_evaluate_reproduces_the_stored_metrics` | Re-scoring the stored fold predictions gives the stored ResEnc-L `per_case.csv` within 1e-6 |

`NNUNET_WORKSPACE` (default `nnUNet_data/`) and `VS_SOURCE_ROOT` (default
`../fastMONAI/research/vestibular_schwannoma`) say where the data are.

## Container

The release to match is `nnunet-vsseg:20261112T095746Z`, built on Fiona base
`sha256:9d36332e725d57c8d5907fba944fb246b5da0df30313de3ec7808fbbd37ec7a4`.
From `nnunet_inference/integration/`:

```bash
RELEASE=20261112T095746Z
VERSION=$RELEASE IMAGE_NAME=nnunet-vsseg-check \
FIONA_BASE=haukebartsch/fiona-component-python@sha256:9d36332e725d57c8d5907fba944fb246b5da0df30313de3ec7808fbbd37ec7a4 \
    ./build.sh --build

./release-regression.sh nnunet-vsseg:$RELEASE nnunet-vsseg-check:$RELEASE \
    ../tests/dicom_input "${NNUNET_WORKSPACE:-../../nnUNet_data}/regression/$(date +%F)"
```

The second command prints `Release regression: OK` when the candidate matches the
release with TTA on and off. If it reports differences, first compare the Python
packages of the two images (`docker run --rm --entrypoint /bin/bash IMAGE -lc 'pip freeze'`):
a rebuilt environment layer can differ without any change to this repository.

Then the fold regression, which must report `"bitwise_equal": true` twice:

```bash
mkdir -p /tmp/nnunet-patch-regression
docker run --rm --network none --cpus 16 --memory 64g --memory-swap 64g \
    --entrypoint /bin/bash \
    -v "$PWD/patch_regression.py:/validation/patch_regression.py:ro" \
    -v /tmp/nnunet-patch-regression:/validation/output \
    nnunet-vsseg-check:$RELEASE -lc \
    'python /validation/patch_regression.py --output /validation/output/patch_results.json'
```

Remove the candidate afterwards: `docker image rm nnunet-vsseg-check:$RELEASE nnunet-vsseg-check:latest`.
