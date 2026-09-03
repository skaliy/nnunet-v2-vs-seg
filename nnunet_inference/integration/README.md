# Research PACS container

Builds the MMIV Research PACS image for the approved Dataset002 ensemble:
`checkpoint_final.pth` from folds 0–4, with `fold_all` excluded. TTA is off by
default.

## Files

- `build.sh`: stages the runtime and model, verifies hashes, builds, and exports
- `entrypoint.sh`: validates the runtime contract and runs inference plus pr2mask
- `validate_output.py` and `smoke-test.sh`: validate a completed container run
- `.ror/virt/Dockerfile` and `requirements.yml`: container definition
- `release_packages/<version>/`: ignored archives, checksums, and handoff notes

## Build

Requires Docker, `jq`, `gzip`, and GNU coreutils. Model weights are distributed
separately and are not part of this repository.

Run from this directory:

```bash
./build.sh --stage-only  # verify staged files without Docker
./build.sh --build       # build dated and latest tags
./build.sh --save        # build, validate, and export with SHA-256
```

Saved deliverables are written to `release_packages/<VERSION>/`.

`VERSION` defaults to a UTC timestamp. Set it for a controlled release, and use
`NNUNET_MODEL_SRC` only to point to the same approved five-fold model layout:

```bash
VERSION=20260903T120000Z ./build.sh --save
```

Qualify and hand off the dated tag; `latest` is only a convenience alias.
Each build intentionally pulls the current Fiona `latest` base. Preserve the
exported image and checksum for an exact qualified deployment; a later rebuild
may resolve a newer Fiona image.

The build packages `plans.json`, `dataset.json`, and exactly five final
checkpoints. A manifest and checksums bind the model files, fold order, nnU-Net
version, and DICOM UID contract.

## Runtime and qualification

The container reads `/data/input` and returns exactly `fused`,
`fused_vote_map`, `reports`, and `mask` under `/output`. The only supported
`ROR_CONT_OPTIONS` key is `tta`:

```json
{"tta": false}
```

Run before handoff:

```bash
(cd ../.. && python -m unittest discover \
    -s nnunet_inference/tests -p "test_*.py" -v)
bash -n build.sh
bash -n entrypoint.sh
bash -n smoke-test.sh

IMAGE_TAG=nnunet-vsseg-5fold:<VERSION> \
    ./smoke-test.sh /path/to/approved-dicom /tmp/nnunet-vsseg-output
```

Qualification must use an approved Research PACS DICOM series.

## PACS handoff

Use an administrator-approved selector for the single T1 input series; never
register a broad any-MR selector:

```json
{
  "MMIVVestSchNNUNet5Fold": {
    "select": "<PACS-ADMIN-APPROVED VS T1 SERIES SELECTOR>",
    "ROR_CONT_OPTIONS": "{\"tta\":false}",
    "docker_image": "nnunet-vsseg-5fold:<QUALIFIED_VERSION>"
  }
}
```

From its versioned release-package directory, verify and load the image with:

```bash
sha256sum -c nnunet-vsseg-5fold-<VERSION>.tar.gz.sha256
gzip -dc nnunet-vsseg-5fold-<VERSION>.tar.gz | docker load
docker image inspect nnunet-vsseg-5fold:<VERSION>
```

Record the qualified tag, image/base digests, archive and model hashes, Git
commit, selector, supported options, UID contract, and real-DICOM validation.
Site paths, credentials, AE title, and trigger rules belong in controlled PACS
infrastructure documentation.
