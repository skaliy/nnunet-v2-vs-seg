# nnU-Net ResEnc-M Research PACS container

The active build packages the trained **Dataset003_VSSegmentationCurated medium
ResEnc-M** model: numeric folds 0–4 with `3d_fullres`. Each fold uses its best
checkpoint by default, matching fastMONAI's fold-best release convention.
`NNUNET_CHECKPOINT=final` selects final checkpoints instead. The previous
Dataset002 ResEnc-L images and release archives are retained for rollback.

The container follows fastMONAI's PACS output and REDCap contract. It keeps
nnU-Net's own preprocessing, Gaussian sliding windows, trained patch size
`48 x 192 x 192` in nnU-Net axis order, mirror axes and ordered fold **logit**
averaging. Probability export occurs after the logit ensemble. No largest
connected-component filter is applied during inference.

## Prepare, test, build and export

Use Python 3.11 with nnUNet 2.6.2, dynamic-network-architectures 0.4.2 and
PyTorch 2.8 to stage bundles. `requirements.yml` pins the CPU deployment stack.
The builder strict-loads each fold and stages inference-only native `.pth`
files with the original weights and mirror settings. Optimizer and training
state are omitted; original trained checkpoints remain unchanged.

From the repository root, run:

```bash
python -m unittest discover -s nnunet_inference/tests -p 'test_*.py' -v
bash -n nnunet_inference/integration/build.sh
bash -n nnunet_inference/integration/entrypoint.sh
bash -n nnunet_inference/integration/smoke-test.sh
cd nnunet_inference/integration
./build.sh --stage-only
./build.sh --build
./build.sh --save
```

Each invocation stages and verifies the bundle. Successful builds remove the
reproducible `model/` and `_pkg/` staging copies; `--stage-only` retains them
for inspection. Choose one build mode per release; `--save` builds, verifies
and exports in one invocation. `VERSION`
defaults to a UTC timestamp. Set `NNUNET_PYTHON` to a staging interpreter when
needed. `NNUNET_MODEL_SRC` may point to a copy of the same approved Dataset003
medium model, whose dataset/plan/trainer/fold identities are checked.

Model metadata, inference checkpoint hashes, source checkpoint hashes,
checkpoint epochs, fold order, CPU settings, runtime versions and the DICOM
UID contract are bound to `deployment_manifest.json` and the bundle SHA-256.
The image verifies files and pinned model runtime versions. Builds pull the
current Fiona base, resolve it to an immutable digest and record that digest.

Before preparing a release with changed weights or checkpoint selection,
allocate its bundle SHA-256 an unused permanent REDCap instance in
`../redcap_model_instances.json`. Unknown bundles fail closed. Never renumber
existing allocations: fastMONAI UNet is 1, DynUNet is 2, and the initial
curated medium nnUNet fold-best bundle is **3**. Preserve these allocations
across the projects when adding future releases.

## CPU and memory contract

Five folds execute concurrently in five CPU worker processes, each with three
disjoint logical CPUs and three BLAS/OpenMP/PyTorch/ITK threads. The coordinator
caps affinity to 16 available logical CPUs; at least 15 are required. The
preprocessed volume and fold results use disk-backed NumPy buffers rather than
multiprocessing queues, avoiding Docker's default 64 MiB shared-memory limit.

Enforce **16 CPU threads and 64 GiB**, with no additional swap allowance:

```bash
docker run --rm --cpus 16 --memory 64g --memory-swap 64g \
  -e ROR_CONT_OPTIONS='{"model-type":"nnunet_medium","tta":true}' \
  -v /absolute/path/dicom:/data/input:ro \
  -v /absolute/path/empty-output:/output nnunet-vsseg-5fold:<VERSION>
```

For fixed host CPU selection, also pass `--cpuset-cpus` with 16 allowed logical
CPU IDs. The PACS administrator must set equivalent limits on its container
launcher; memory limits cannot be encoded in a Dockerfile.

## Runtime options and outputs

`ROR_CONT_OPTIONS` supports `model-type` (`nnunet_medium` only) and `tta`
(a JSON Boolean). Defaults are medium nnUNet and **TTA enabled**, as in
fastMONAI. `{"tta":false}` disables mirror TTA. Unknown keys, other model
types, and string/numeric TTA values are rejected before inference.

The container returns `mask`, `fused`, `fused_vote_map`, `reports`, `redcap`,
`pacs_command.log` and `runtime_resources.json`. `vote_map` is intermediate.
Vote-map intensities are `round(probability * 65535)`. Empty masks may produce
an empty `reports` directory. Existing owned output directories are rejected.
Derived DICOM UIDs bind source geometry/identities, model content, checkpoint,
folds, TTA and release version. pr2mask identities also distinguish nnUNet
from fastMONAI bundles. Temporary inference products are removed after use.

`redcap/<report-series-UID>/output.json` contains the same fields as fastMONAI:
`vs_mask_json`, `vs_measurements_json`, model/bundle/prediction identity,
deployment version, TTA and creation time. The lossless mask uses schema 1,
bit-packing, gzip and Base64; geometry is DICOM LPS with lengths in millimetres
and ordered per-slice positions. Read it using
`nnunet_inference.redcap_output.decode_mask`. Measurements are one complete
JSON list, preserving their original region numbers inside that list. A
same-bundle rerun replaces its permanent model instance; different bundles use
different instances. `output_data_dictionary.zip` extends the pr2mask instrument.
Install the added fields in the receiving project and ensure its importer
honors numeric repeat instances. The container writes export files; live
PACS-to-REDCap transfer requires separate site qualification.

## Qualification and handoff

Use a controlled DICOM fixture for full inference and a generated nonuniform
full-sized patch for native-versus-parallel numerical regression:

```bash
IMAGE_TAG=nnunet-vsseg-5fold:<VERSION> TTA=false \
  ./smoke-test.sh /path/to/test-dicom /tmp/nnunet-medium-output
# Repeat with TTA=true and a fresh output directory to qualify full-case TTA.
```

`smoke-test.sh` fixes CPU affinity to 16 logical threads, applies a 64 GiB
memory/swap cap and validates all DICOM outputs, reconstructed REDCap voxels,
physical geometry, source IDs, repeat instance and dictionary fields.
`patch_regression.py` compares actual medium-network logits and masks for
sequential and parallel folds, with TTA both off and on. Run it using the
pinned image environment; it requires no medical images.

From this integration directory:

```bash
mkdir -p /tmp/nnunet-patch-regression
docker run --rm --network none --cpus 16 --memory 64g --memory-swap 64g \
  --entrypoint /bin/bash \
  -v "$PWD/patch_regression.py:/validation/patch_regression.py:ro" \
  -v /tmp/nnunet-patch-regression:/validation/output \
  nnunet-vsseg-5fold:<VERSION> -lc \
  'python /validation/patch_regression.py --output /validation/output/patch_results.json'
```

The helper caps affinity to 16 logical CPUs and compares all five folds by
default. Add `--folds 0,1` for a smaller diagnostic regression or `--tta false`
to run only the non-TTA condition. Completed production qualification should
record the full five-fold results.

Exports go to `release_packages/<VERSION>/` with the Docker archive, checksum,
README and `release.json` containing image/base/model/source fingerprints.
Record completed checks, resource peak and exact limitations in
`qualification.json`. A built image is not automatically inference-qualified.
Verify the archive checksum and load the dated tag before handoff. Use a
site-controlled selector for a single CE-T1 input series:

```json
{
  "MMIVVestSchNNUNetMedium": {
    "select": "<PACS-ADMIN-APPROVED VS CE-T1 SERIES SELECTOR>",
    "ROR_CONT_OPTIONS": "{\"model-type\":\"nnunet_medium\",\"tta\":true}",
    "docker_image": "nnunet-vsseg-5fold:<QUALIFIED_VERSION>"
  }
}
```
