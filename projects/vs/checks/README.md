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
| `test_evaluate_reproduces_the_stored_metrics` | Re-scoring the stored fold predictions gives the stored ResEnc-L `per_case.csv` and `summary.csv` within 1e-6, and the stored `per_case.csv` has exactly the three columns that `presentation_summary.py` added besides |

`NNUNET_WORKSPACE` (default `nnUNet_data/`) and `VS_SOURCE_ROOT` (default
`../fastMONAI/research/vestibular_schwannoma`) say where the data are. When set,
`NNUNET_WORKSPACE` must be an absolute path: these checks run from the repository root and
the container commands below from `nnunet_inference/integration/`.

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

`build.sh` refuses to build when `nnunet-vsseg-check:$RELEASE` already exists, so remove
the candidate of an earlier check first (the command at the end of this section).

Relative paths are resolved from the current directory. The work directory must lie
outside the repository or in a folder that Git ignores, as `nnUNet_data/` is, because its
snapshots and REDCap files hold patient identifiers; the script refuses any other.

The second command prints `Release regression: OK` when the candidate matches the
release with TTA on and off. A line after it names, for each TTA setting, the pixels of
an output on which the two release runs disagreed and which were therefore not compared.
If it reports differences, first compare the Python packages of the two images
(`docker run --rm --entrypoint /bin/bash IMAGE -lc 'pip freeze'`): a rebuilt environment
layer can differ without any change to this repository.

### Shorter check: masks, UIDs and REDCap values only

`release-regression.sh` runs the release four times on the day of the check, because the
fused images and the report have the date drawn into them. When the masks are what
matters, the candidate can instead be compared with release runs kept from earlier days;
it then runs once, with TTA on, in about an hour. A baseline made from two release runs
of different days holds every entry that does not depend on the day: all mask pixels,
the tags and UIDs of every series, and all REDCap values. The pixels of `fused`,
`fused_vote_map` and `reports` are left out, and `stable` says so.

```bash
RUNS="${NNUNET_WORKSPACE:-../../nnUNet_data}/regression"
tta=true
# Once per release: the baseline from release runs of two different days.
python compare_outputs.py stable \
    "$RUNS/2026-11-16/reference-52ef0f316a90-tta-$tta-1.json" \
    "$RUNS/2026-11-18/reference-52ef0f316a90-tta-$tta-1.json" \
    "$RUNS/release-$RELEASE/baseline-across-days-tta-$tta.json"
# For each check: run the candidate under the release's name and compare.
out="$RUNS/$(date +%F)/candidate-tta-$tta"
IMAGE_TAG=nnunet-vsseg-check:$RELEASE TTA=$tta \
CONTAINER_VERSION_OVERRIDE=nnunet-vsseg:$RELEASE \
    ./smoke-test.sh ../tests/dicom_input "$out"
python compare_outputs.py snapshot "$out" "$out.json"
python compare_outputs.py check \
    "$RUNS/release-$RELEASE/baseline-across-days-tta-$tta.json" "$out.json"
```

The `check` must print `candidate matches the baseline`. A run with TTA off (`tta=false`)
works the same way; the fold regression below compares TTA off bit for bit, so it is not
needed for an ordinary change. Run the candidate on the same
CPUs as the release runs (the default, the first sixteen): the CPU numbers are recorded
in `runtime_resources.json` and compared.

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
