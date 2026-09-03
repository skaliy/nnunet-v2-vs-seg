#!/bin/bash
# Build/export the fixed five-fold Research-PACS image.
set -euo pipefail
cd "$(dirname "$0")"

readonly INTEGRATION_DIR="$(pwd)"
readonly PACKAGE_SRC="$(cd .. && pwd)"
readonly PROJECT_ROOT="$(cd ../.. && pwd)"
readonly DEFAULT_MODEL_SRC="${PROJECT_ROOT}/nnUNet_data/nnUNet_results/Dataset002_VSSegmentationRaw/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
readonly MODEL_SRC="${NNUNET_MODEL_SRC:-${DEFAULT_MODEL_SRC}}"
readonly IMAGE_NAME="${IMAGE_NAME:-nnunet-vsseg-5fold}"
readonly VERSION="${VERSION:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly DATED_TAG="${IMAGE_NAME}:${VERSION}"
readonly LATEST_TAG="${IMAGE_NAME}:latest"
readonly CONDA_ENV=nnunet-vsseg

mode="${1:---build}"
case "${mode}" in
    --stage-only|--build|--save) ;;
    *) echo "Usage: $0 [--stage-only|--build|--save]" >&2; exit 2 ;;
esac

echo "Staging nnunet_inference runtime package ..."
rm -rf _pkg
mkdir -p _pkg/nnunet_inference
for module in __init__.py __main__.py dicom_io.py predictor.py pipeline.py; do
    cp "${PACKAGE_SRC}/${module}" "_pkg/nnunet_inference/${module}"
done

echo "Staging Dataset002 final checkpoints for folds 0-4 ..."
for metadata in plans.json dataset.json; do
    if [ ! -f "${MODEL_SRC}/${metadata}" ]; then
        echo "Error: missing ${MODEL_SRC}/${metadata}" >&2
        exit 2
    fi
done
for fold in 0 1 2 3 4; do
    if [ ! -f "${MODEL_SRC}/fold_${fold}/checkpoint_final.pth" ]; then
        echo "Error: missing final checkpoint for fold_${fold}" >&2
        exit 2
    fi
done

rm -rf model
mkdir -p model
cp "${MODEL_SRC}/plans.json" "${MODEL_SRC}/dataset.json" model/
for fold in 0 1 2 3 4; do
    mkdir -p "model/fold_${fold}"
    cp --reflink=auto "${MODEL_SRC}/fold_${fold}/checkpoint_final.pth" "model/fold_${fold}/"
done

printf '%s\n' "0,1,2,3,4" > model/MODEL_FOLDS.txt
printf '%s\n' "ResEnc-L Dataset002 folds 0,1,2,3,4" > model/MODEL_NAME.txt
(
    cd model
    sha256sum plans.json dataset.json \
        fold_0/checkpoint_final.pth fold_1/checkpoint_final.pth \
        fold_2/checkpoint_final.pth fold_3/checkpoint_final.pth \
        fold_4/checkpoint_final.pth > MODEL_SHA256SUMS.txt
)
model_version="$(sha256sum model/MODEL_SHA256SUMS.txt | cut -d' ' -f1)"
printf '%s\n' "${model_version}" > model/MODEL_VERSION.txt
jq -n \
    --arg bundle_sha256 "${model_version}" \
    --arg model_name "ResEnc-L Dataset002 folds 0,1,2,3,4" \
    '{
        schema_version: 1,
        model_name: $model_name,
        dataset: "Dataset002_VSSegmentationRaw",
        trainer: "nnUNetTrainer",
        plans: "nnUNetResEncUNetLPlans",
        configuration: "3d_fullres",
        checkpoint: "checkpoint_final.pth",
        folds: [0, 1, 2, 3, 4],
        nnunet_version: "2.6.2",
        bundle_sha256: $bundle_sha256,
        dicom_uid: {
            format_version: 1,
            root: "2.25",
            generation: "uuid5",
            namespace_uuid: "bb88c59d-5a75-5f47-bb52-3bc9f6db7808",
            model_code: 1,
            deployment_code: 1,
            output_codes: {segmentation: 1, probability: 2}
        }
    }' > model/deployment_manifest.json

if [ -d model/fold_all ]; then
    echo "Error: fold_all must not be packaged" >&2
    exit 2
fi
fold_count="$(find model -mindepth 2 -maxdepth 2 -name checkpoint_final.pth -type f | wc -l)"
if [ "${fold_count}" -ne 5 ]; then
    echo "Error: expected exactly five staged checkpoints, found ${fold_count}" >&2
    exit 2
fi
(
    cd model
    sha256sum -c MODEL_SHA256SUMS.txt
)
test "$(jq -r '.bundle_sha256' model/deployment_manifest.json)" = "${model_version}"

echo "Staged model: $(cat model/MODEL_NAME.txt)"
echo "Model version: ${model_version:0:12}"
du -ch model/fold_*/checkpoint_final.pth | tail -n 1

if [ "${mode}" = "--stage-only" ]; then
    echo "Stage-only validation complete."
    exit 0
fi

echo "Refreshing and recording the Research-PACS Fiona base image ..."
docker pull haukebartsch/fiona-component-python:latest
docker image inspect haukebartsch/fiona-component-python:latest --format \
    'base_id={{.Id}} repo_digests={{json .RepoDigests}}'

echo "Building ${DATED_TAG} and ${LATEST_TAG} ..."
docker build --pull \
    --build-arg "conda_env=${CONDA_ENV}" \
    --build-arg "VERSION=${VERSION}" \
    -f .ror/virt/Dockerfile \
    -t "${DATED_TAG}" -t "${LATEST_TAG}" "${INTEGRATION_DIR}"

echo "Validating image metadata, manifest, dependencies, and checkpoints ..."
docker image inspect "${DATED_TAG}" --format \
    'image_id={{.Id}} tags={{json .RepoTags}} version={{index .Config.Labels "org.opencontainers.image.version"}} folds={{index .Config.Labels "com.mmiv.model.folds"}} tta_default={{index .Config.Labels "com.mmiv.tta.default"}}'
docker run --rm --entrypoint /bin/bash "${DATED_TAG}" -lc \
    'set -e; cd /app/model; sha256sum -c MODEL_SHA256SUMS.txt; test "$(jq -r .bundle_sha256 deployment_manifest.json)" = "$(tr -d "[:space:]" < MODEL_VERSION.txt)"; test "$(jq -r ".folds | join(\",\")" deployment_manifest.json)" = "0,1,2,3,4"; test ! -d fold_all; test -x /pr2mask/imageAndMask2Report; test -x /pr2mask/imageAndMask2Fused; python -m nnunet_inference --help >/dev/null; echo "five-fold image validation: OK"'

if [ "${mode}" = "--save" ]; then
    release_dir="${INTEGRATION_DIR}/release_packages/${VERSION}"
    archive_name="${IMAGE_NAME}-${VERSION}.tar.gz"
    archive="${release_dir}/${archive_name}"
    if [ -e "${release_dir}" ]; then
        echo "Error: release package already exists: ${release_dir}" >&2
        exit 2
    fi
    mkdir -p "${release_dir}"

    echo "Exporting both image references -> ${archive} ..."
    docker save "${DATED_TAG}" "${LATEST_TAG}" | gzip -1 > "${archive}"
    (
        cd "${release_dir}"
        sha256sum "${archive_name}" > "${archive_name}.sha256"
        sha256sum -c "${archive_name}.sha256"
    )
    {
        printf 'nnU-Net vestibular schwannoma Research PACS release\n\n'
        printf 'Docker image: %s\n' "${DATED_TAG}"
        printf 'Archive: %s\n' "${archive_name}"
        printf 'Model: ResEnc-L Dataset002 folds 0,1,2,3,4\n'
        printf 'Model bundle SHA-256: %s\n' "${model_version}"
        printf 'TTA default: off\n'
        printf 'ROR_CONT_OPTIONS: {"tta":false}\n\n'
        printf 'Verify with:\n  sha256sum -c %s.sha256\n' "${archive_name}"
        printf 'Load with:\n  gzip -dc %s | docker load\n' "${archive_name}"
    } > "${release_dir}/README.md"

    echo "Qualified image tag: ${DATED_TAG}"
    echo "Release package: ${release_dir}"
fi
