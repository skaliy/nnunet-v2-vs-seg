#!/bin/bash
# Stage, build, verify and export the curated medium nnU-Net PACS image.
set -euo pipefail
cd "$(dirname "$0")"
readonly INTEGRATION_DIR="$(pwd)"
readonly PACKAGE_SRC="$(cd .. && pwd)"
readonly PROJECT_ROOT="$(cd ../.. && pwd)"
readonly MODEL_SRC="${NNUNET_MODEL_SRC:-${PROJECT_ROOT}/nnUNet_data/nnUNet_results/Dataset003_VSSegmentationCurated/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres}"
readonly CHECKPOINT="${NNUNET_CHECKPOINT:-best}"
readonly PYTHON_BIN="${NNUNET_PYTHON:-python}"
readonly IMAGE_NAME="${IMAGE_NAME:-nnunet-vsseg-5fold}"
readonly VERSION="${VERSION:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly DATED_TAG="${IMAGE_NAME}:${VERSION}"
readonly BASE_IMAGE=haukebartsch/fiona-component-python:latest
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
    'import sys; from nnunet_inference.deployment import validate_deployment; from nnunet_inference.redcap_output import model_repeat_instance; print("REDCap instance:", model_repeat_instance(validate_deployment(sys.argv[1])["bundle_sha256"]))' \
    "${stage_dir}/model"
rm -rf model _pkg
mv "${stage_dir}/model" model
mkdir -p _pkg/nnunet_inference
for module in __init__.py __main__.py dicom_io.py predictor.py pipeline.py parallel.py deployment.py pacs.py redcap_output.py redcap_model_instances.json; do
    cp "${PACKAGE_SRC}/${module}" "_pkg/nnunet_inference/${module}"
done
if [ "${mode}" = --stage-only ]; then
    echo 'Stage-only model and registry validation complete.'
    exit 0
fi

docker pull "${BASE_IMAGE}"
BASE_REF="$(docker image inspect "${BASE_IMAGE}" --format '{{index .RepoDigests 0}}')"
test -n "${BASE_REF}"
docker build --pull --build-arg "VERSION=${VERSION}" --build-arg "FIONA_BASE=${BASE_REF}" \
    -f .ror/virt/Dockerfile -t "${DATED_TAG}" -t "${IMAGE_NAME}:latest" .
docker run --rm --network none --entrypoint /bin/bash "${DATED_TAG}" -lc \
    'set -e
    python -m nnunet_inference.deployment --output /app/model --verify-only --verify-runtime
    python -m nnunet_inference.pacs --help >/dev/null'

if [ "${mode}" = --save ]; then
    release_dir="${INTEGRATION_DIR}/release_packages/${VERSION}"
    mkdir -p "${release_dir}"
    archive_name="${IMAGE_NAME}-${VERSION}.tar.gz"
    docker save "${DATED_TAG}" "${IMAGE_NAME}:latest" | gzip -1 > "${release_dir}/${archive_name}"
    (cd "${release_dir}"; sha256sum "${archive_name}" > "${archive_name}.sha256"; sha256sum -c "${archive_name}.sha256"; gzip -t "${archive_name}")
    "${PYTHON_BIN}" release.py "${release_dir}" "${DATED_TAG}" "${BASE_REF}"
    echo "Image and release archive verified: ${release_dir}"
    echo 'Use smoke-test.sh to qualify complete inference before PACS deployment.'
fi

# Successful images contain these reproducible copies; retain them only for --stage-only.
rm -rf model _pkg
