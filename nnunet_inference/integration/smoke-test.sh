#!/bin/bash
# Qualify the packaged image with 16 fixed CPUs, 64 GiB and no swap allowance.
set -euo pipefail
cd "$(dirname "$0")"
readonly INPUT_DIR="${1:-}"
readonly OUTPUT_DIR="${2:-$(pwd)/smoke-test-output}"
readonly IMAGE_TAG="${IMAGE_TAG:-nnunet-vsseg-5fold:latest}"
readonly PYTHON_BIN="${NNUNET_PYTHON:-python}"
readonly TTA="${TTA:-true}"
if [ -z "${INPUT_DIR}" ] || [ ! -d "${INPUT_DIR}" ]; then
    echo "Usage: $0 DICOM_INPUT_DIR [OUTPUT_DIR]" >&2; exit 2
fi
case "${TTA}" in true|false) ;; *) echo 'TTA must be true or false' >&2; exit 2 ;; esac
mkdir -p "${OUTPUT_DIR}"
if [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo 'Error: output directory must be empty' >&2; exit 2
fi
CPUSET="${CPUSET:-$("${PYTHON_BIN}" -c 'import os; ids=sorted(os.sched_getaffinity(0))[:16]; assert len(ids)==16, "16 available CPUs required"; print(",".join(map(str,ids)))')}"
docker run --rm --network none --cpus 16 --cpuset-cpus "${CPUSET}" \
    --memory 64g --memory-swap 64g \
    -e "ROR_CONT_OPTIONS={\"model-type\":\"nnunet_medium\",\"tta\":${TTA}}" \
    -v "$(realpath "${INPUT_DIR}"):/data/input:ro" \
    -v "$(realpath "${OUTPUT_DIR}"):/output" "${IMAGE_TAG}"
PYTHONPATH="$(cd ../.. && pwd)" "${PYTHON_BIN}" validate_output.py "${INPUT_DIR}" "${OUTPUT_DIR}"
echo "Container DICOM and REDCap smoke test: OK (${OUTPUT_DIR})"
