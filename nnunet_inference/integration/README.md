# nnU-Net ResEnc-M / ResEnc-L Research PACS container

Each image packages one trained **Dataset003_VSSegmentationCurated** model,
numeric folds 0–4 with `3d_fullres`: large ResEnc-L (`NNUNET_MODEL=large`,
the default, image `nnunet-vsseg`) or medium ResEnc-M
(`NNUNET_MODEL=medium`, image `nnunet-vsseg-5fold`). Large is the PACS model:
under 16 CPUs it is faster than medium and had the fewest false positives. Each fold uses its
final checkpoint by default, as nnU-Net does and as in the cross-validation
results. `NNUNET_CHECKPOINT=best` selects fold-best checkpoints. Medium is not deployed and
has no REDCap instance, so a medium build stops until one is registered. The previous Dataset002 ResEnc-L
images and release archives are retained for rollback.

The container follows fastMONAI's PACS output and REDCap contract. It keeps
nnU-Net's own preprocessing, Gaussian sliding windows, trained patch size
`64 x 320 x 256` (large) or `48 x 192 x 192` (medium) in nnU-Net axis order and
ordered fold **logit** averaging. TTA averages the original and one copy flipped
inferior-superior and anterior-posterior together (nnU-Net axes 0 and 1). On the
held-out folds this had as few or fewer false positives than nnU-Net's
eight-flip mirroring, at a quarter of the cost. Probability export occurs after the logit ensemble. No largest
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
NNUNET_MODEL=medium ./build.sh --save
```

Each invocation stages and verifies the bundle. Successful builds remove the
reproducible `model/` and `_pkg/` staging copies; `--stage-only` retains them
for inspection. Choose one build mode per release; `--save` builds, verifies
and exports in one invocation. `VERSION`
defaults to a UTC timestamp. Set `NNUNET_PYTHON` to a staging interpreter when
needed. `NNUNET_MODEL_SRC` may point to a copy of the same approved Dataset003
model, whose dataset/plan/trainer/fold identities are checked.

Model metadata, inference checkpoint hashes, source checkpoint hashes,
checkpoint epochs, fold order, CPU settings, runtime versions and the DICOM
UID contract are bound to `deployment_manifest.json` and the bundle SHA-256.
The image verifies files and pinned model runtime versions. Builds pull the
current Fiona base, resolve it to an immutable digest and record that digest.

Before preparing a release with changed weights or checkpoint selection, give its bundle
SHA-256 an unused block of 100 REDCap instances in `../redcap_model_instances.json`. The
registry stores the block's base, a multiple of 100; region k goes to base + k. The large
final-checkpoint bundle has **100** (101-199) and the fastMONAI DynUNet bundle **200**.
Instances below 100 belong to the manual mask (1), pr2mask's own numbering and the old
single-instance results (large at 10, earlier 2), which are no longer updated.
Unknown bundles and invalid or duplicate bases fail closed. Don't renumber blocks after
REDCap imports, and keep them unique across both projects.

## CPU and memory contract

Five folds execute concurrently in five CPU worker processes, each with three
disjoint logical CPUs and three BLAS/OpenMP/PyTorch/ITK threads. The coordinator
caps affinity to 16 available logical CPUs; at least 15 are required. The
preprocessed volume and fold results use disk-backed NumPy buffers rather than
multiprocessing queues, avoiding Docker's default 64 MiB shared-memory limit.

Each worker keeps one copy of its weights. A lock shared by the workers lets one
fold at a time run the full-resolution decoder stage, where a forward pass peaks
in memory. Without it the folds run in step and their peaks add up. On the
120-slice qualification case this lowers the peak from 31.7 to 20.8 GiB at the
same speed, with bit-identical logits.

Enforce **16 CPU threads and 64 GiB**, with no additional swap allowance:

```bash
docker run --rm --cpus 16 --memory 64g --memory-swap 64g \
  -e ROR_CONT_OPTIONS='{"tta":true}' \
  -v /absolute/path/dicom:/data/input:ro \
  -v /absolute/path/empty-output:/output nnunet-vsseg:<VERSION>
```

For fixed host CPU selection, also pass `--cpuset-cpus` with 16 allowed logical
CPU IDs. The PACS administrator must set equivalent limits on its container
launcher; memory limits cannot be encoded in a Dockerfile.

## Runtime options and outputs

No options are needed. ROR passes the JSON given with `-envs` as `ROR_CONT_OPTIONS`:

```bash
ror trigger -cont nnunet-vsseg:<VERSION> -each
ror trigger -cont nnunet-vsseg:<VERSION> -each -envs '{"tta":false}'
```

`ROR_CONT_OPTIONS` supports `model-type` and `tta` (a JSON Boolean).
`model-type` is optional and must name the packaged model (`nnunet_medium` or
`nnunet_large`). **TTA is enabled** by default, as in fastMONAI. TTA averages the original and one copy flipped inferior-superior and anterior-posterior together.
`{"tta":false}` disables mirror TTA. Unknown keys, other model
types, string/numeric TTA values and non-objects such as `""` are rejected before
inference; the error prints the value it received.

The container returns `labels`, `fused`, `fused_vote_map`, `reports`, `redcap`,
`pacs_command.log` and `runtime_resources.json`. The nnU-Net `mask` and `vote_map`
series are intermediate pr2mask inputs. pr2mask writes its own mask series to
`labels/`, using its folder layout; nnU-Net provenance (SeriesDescription,
SoftwareVersions, DerivationDescription and a `MASK` ImageType value) is copied
onto it before publication. Its study identity, window and UIDs stay as pr2mask
wrote them, so the PACS files the mask with the other pr2mask outputs.
Vote-map intensities are `round(probability * 65535)`, with the probabilities
interpolated linearly between model slices. pr2mask draws values from 32768 up
in blue, so values inside the mask are raised to at least 32768 and values
outside it lowered to at most 32767: the blue region is exactly the mask, and
yellow marks voxels with some foreground probability outside it. Empty masks may produce
an empty `reports` directory. Existing owned output directories are rejected.
Derived DICOM UIDs bind source geometry/identities, model content, checkpoint,
folds, TTA and release version. pr2mask identities also distinguish nnUNet
from fastMONAI bundles. Temporary inference products are removed after use.

`redcap/<mask-series-UID>/output.json` holds pr2mask's rows for the `pr2mask` instrument,
the same as fastMONAI:

- **One instance per region.** pr2mask numbers the mask's 26-connected regions 1, 2, 3, ...
  Region k goes to repeat instance base + k (101, 102, ... for the large bundle). Only
  instances with a region are written. A rerun overwrites the bundle's instances; if it finds
  fewer regions, the earlier run's higher instances stay.
- **Empty mask:** instance base + 1 with blank columns, so REDCap shows the model ran. It is
  routed by the source PatientID and ReferringPhysicianName (`EventName:` removed), like
  pr2mask. pr2mask lists no columns then, so the exporter takes them from
  `pr2mask_dictionary.csv`.
- **`model_version`** is added to every instance, for example
  `nnunet_large 5-fold, nnunet-vsseg:20261111T090000Z` (from the image's `CONTAINER_VERSION`).
  The field is on the REDCap form, not in pr2mask; the exporter also adds it to
  `output_data_dictionary.zip`.
- **Mask text.** pr2mask's `mask_base64_gzip_compressed` covers the whole scan at 16 bits
  per voxel: about 84,000 characters at 512x512x120. REDCap may cut values over 65,535
  characters; this hasn't been checked in the receiving project. The value is sent as
  pr2mask wrote it, and the full mask is always the published `labels` series.

`pr2mask_dictionary.csv` holds the `instrument.csv` rows of a non-empty run of the packaged
pr2mask. When the Fiona base changes, regenerate it from the smoke test;
`validate_output.py` fails while it differs. Install the emitted
`output_data_dictionary.zip` in the receiving project; its importer must honour numeric
repeat instances. The container writes export files; live PACS-to-REDCap transfer requires
separate site qualification.

## Qualification and handoff

Use a controlled DICOM fixture for full inference and a generated nonuniform
full-sized patch for native-versus-parallel numerical regression:

```bash
IMAGE_TAG=nnunet-vsseg:<VERSION> TTA=false \
  ./smoke-test.sh /path/to/test-dicom /tmp/nnunet-large-output
# Repeat with TTA=true and a fresh output directory to qualify full-case TTA.
```

`smoke-test.sh` fixes CPU affinity to 16 logical threads, applies a 64 GiB
memory/swap cap and validates all DICOM outputs, physical geometry, source IDs and the
REDCap rows: instances base + 1, base + 2, ..., the dictionary fields, one
`model_version`, and region sizes that add up to the published mask.
`patch_regression.py` compares actual network logits and masks for
sequential and parallel folds, with TTA both off and on. Run it using the
pinned image environment; it requires no medical images.

From this integration directory:

```bash
mkdir -p /tmp/nnunet-patch-regression
docker run --rm --network none --cpus 16 --memory 64g --memory-swap 64g \
  --entrypoint /bin/bash \
  -v "$PWD/patch_regression.py:/validation/patch_regression.py:ro" \
  -v /tmp/nnunet-patch-regression:/validation/output \
  nnunet-vsseg:<VERSION> -lc \
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
    "docker_image": "nnunet-vsseg-5fold:latest"
  },
  "MMIVVestSchNNUNetLarge": {
    "select": "<PACS-ADMIN-APPROVED VS CE-T1 SERIES SELECTOR>",
    "ROR_CONT_OPTIONS": "{\"model-type\":\"nnunet_large\",\"tta\":true}",
    "docker_image": "nnunet-vsseg:latest"
  }
}
```

The entries use `latest`, so a new release needs no config change. The release archive
carries both tags, and loading it moves `latest` to the new image. The dated tag shows
which release `latest` is.
