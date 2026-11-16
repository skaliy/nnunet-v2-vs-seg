#!/bin/bash
# Stage, build, verify and export the curated ResEnc-L nnU-Net PACS image.
set -euo pipefail
cd "$(dirname "$0")"
readonly INTEGRATION_DIR="$(pwd)"
readonly PACKAGE_SRC="$(cd .. && pwd)"
readonly PROJECT_ROOT="$(cd ../.. && pwd)"
readonly MODEL_SRC="${NNUNET_MODEL_SRC:-${PROJECT_ROOT}/nnUNet_data/nnUNet_results/Dataset003_VSSegmentationCurated/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres}"
readonly CHECKPOINT="${NNUNET_CHECKPOINT:-final}"
readonly PYTHON_BIN="${NNUNET_PYTHON:-python}"
readonly IMAGE_NAME="${IMAGE_NAME:-nnunet-vsseg}"
readonly VERSION="${VERSION:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly DATED_TAG="${IMAGE_NAME}:${VERSION}"
# FIONA_BASE pins the base, for example to rebuild on the digest of an earlier release.
readonly BASE_IMAGE="${FIONA_BASE:-haukebartsch/fiona-component-python:latest}"
mode="${1:---build}"
case "${mode}" in
    --stage-only|--build|--save) ;;
    *) echo "Usage: $0 [--stage-only|--build|--save]" >&2; exit 2 ;;
esac
if [[ ! "${VERSION}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo 'Error: VERSION must be a valid Docker tag' >&2; exit 2
fi
if [ "${mode}" = --save ] && [ -e "release_packages/${VERSION}" ]; then
    echo 'Error: release package already exists; select a new VERSION' >&2; exit 2
fi

# Stage only inference files. Preserve all original trained checkpoints and releases.
stage_dir="$(mktemp -d "${INTEGRATION_DIR}/.bundle-stage-XXXXXX")"
trap 'rm -rf "${stage_dir}"' EXIT
PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -m nnunet_inference.deployment \
    --source "${MODEL_SRC}" --output "${stage_dir}/model" --checkpoint "${CHECKPOINT}"
PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -c \
    'import sys; from nnunet_inference.deployment import validate_deployment; from nnunet_inference.redcap_output import model_instance_base; base = model_instance_base(validate_deployment(sys.argv[1], verify_files=False)["bundle_sha256"]); print(f"REDCap instances: {base + 1}-{base + 99}")' \
    "${stage_dir}/model"
rm -rf model _pkg
mv "${stage_dir}/model" model
# The runtime package: its modules and data files, without tests, integration or documents.
mkdir -p _pkg/nnunet_inference
find "${PACKAGE_SRC}" -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' -o -name '*.csv' \) \
    -exec cp {} _pkg/nnunet_inference/ \;
if [ "${mode}" = --stage-only ]; then
    echo 'Stage-only model and registry validation complete.'
    exit 0
fi

docker pull "${BASE_IMAGE}"
BASE_REF="$(docker image inspect "${BASE_IMAGE}" --format '{{index .RepoDigests 0}}')"
test -n "${BASE_REF}"
docker build --pull --build-arg "VERSION=${VERSION}" --build-arg "IMAGE_NAME=${IMAGE_NAME}" \
    --build-arg "FIONA_BASE=${BASE_REF}" \
    -f .ror/virt/Dockerfile -t "${DATED_TAG}" -t "${IMAGE_NAME}:latest" .
docker run --rm --network none --entrypoint /bin/bash "${DATED_TAG}" -lc \
    'set -e
    python -m nnunet_inference.deployment --output /app/model --verify-only --verify-runtime --manifest-only
    python -m nnunet_inference.pacs --help >/dev/null'

if [ "${mode}" = --save ]; then
    release_dir="${INTEGRATION_DIR}/release_packages/${VERSION}"
    mkdir -p "${release_dir}"
    archive_name="${IMAGE_NAME}-${VERSION}.tar.gz"
    docker save "${DATED_TAG}" "${IMAGE_NAME}:latest" | gzip -1 > "${release_dir}/${archive_name}"
    gzip -t "${release_dir}/${archive_name}"
    "${PYTHON_BIN}" release.py "${release_dir}" "${DATED_TAG}" "${BASE_REF}"
    echo "Image and release archive verified: ${release_dir}"
    echo 'Use smoke-test.sh to qualify complete inference before PACS deployment.'
fi

# Successful images contain these reproducible copies; retain them only for --stage-only.
rm -rf model _pkg
