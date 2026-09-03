#!/bin/bash
# Run the packaged container against a DICOM series and verify PACS outputs.
set -euo pipefail
cd "$(dirname "$0")"

readonly INPUT_DIR="${1:-}"
readonly OUTPUT_DIR="${2:-$(pwd)/smoke-test-output}"
readonly IMAGE_TAG="${IMAGE_TAG:-nnunet-vsseg-5fold:latest}"

if [ -z "${INPUT_DIR}" ] || [ ! -d "${INPUT_DIR}" ]; then
    echo "Usage: $0 DICOM_INPUT_DIR [OUTPUT_DIR]" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}"
if [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "Error: output directory must be empty: ${OUTPUT_DIR}" >&2
    exit 2
fi

docker run --rm \
    -e 'ROR_CONT_OPTIONS={"tta":false}' \
    -v "$(realpath "${INPUT_DIR}"):/data/input:ro" \
    -v "$(realpath "${OUTPUT_DIR}"):/output" \
    "${IMAGE_TAG}"

python validate_output.py "${INPUT_DIR}" "${OUTPUT_DIR}"
echo "Container DICOM smoke test: OK (${OUTPUT_DIR})"
