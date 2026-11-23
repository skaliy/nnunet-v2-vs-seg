#!/bin/bash
# Qualify the packaged image with 16 fixed CPUs, 64 GiB and no swap allowance.
# Relative paths are resolved from the current directory.
set -euo pipefail
if [ ! -d "${1:-}" ]; then
    echo "Usage: $0 DICOM_INPUT_DIR [OUTPUT_DIR]" >&2; exit 2
fi
INPUT_DIR="$(realpath -e -- "$1")"
OUTPUT_DIR="$(realpath -m -- "${2:-$(dirname "$0")/smoke-test-output}")"
readonly INPUT_DIR OUTPUT_DIR
readonly IMAGE_TAG="${IMAGE_TAG:-nnunet-vsseg:latest}"
readonly PYTHON_BIN="${NNUNET_PYTHON:-python}"
readonly TTA="${TTA:-true}"
cd "$(dirname "$0")"
case "${TTA}" in true|false) ;; *) echo 'TTA must be true or false' >&2; exit 2 ;; esac

# DICOM headers and REDCap rows hold patient identifiers: keep them out of Git.
# The trailing slash makes Git match a directory that does not exist yet.
repository="$(git rev-parse --show-toplevel 2>/dev/null)" || repository=
if [ -n "${repository}" ] && [[ "${OUTPUT_DIR}/" == "${repository}/"* ]] \
        && ! git -C "${repository}" check-ignore -q -- "${OUTPUT_DIR}/"; then
    echo "Error: the output directory ${OUTPUT_DIR} is inside the repository ${repository}" \
        "and not ignored by Git; its files hold patient identifiers" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}"
if [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo 'Error: output directory must be empty' >&2; exit 2
fi
# A regression run gives the candidate the reference's name, so that the name
# does not show up as a difference in SoftwareVersions and model_version.
identity=()
if [ -n "${CONTAINER_VERSION_OVERRIDE:-}" ]; then
    identity=(-e "CONTAINER_VERSION=${CONTAINER_VERSION_OVERRIDE}")
fi
CPUSET="${CPUSET:-$("${PYTHON_BIN}" -c 'import os; ids=sorted(os.sched_getaffinity(0))[:16]; assert len(ids)==16, "16 available CPUs required"; print(",".join(map(str,ids)))')}"
docker run --rm --network none --cpus 16 --cpuset-cpus "${CPUSET}" \
    --memory 64g --memory-swap 64g \
    -e "ROR_CONT_OPTIONS={\"tta\":${TTA}}" ${identity[@]+"${identity[@]}"} \
    -v "${INPUT_DIR}:/data/input:ro" \
    -v "${OUTPUT_DIR}:/output" "${IMAGE_TAG}"
PYTHONPATH="$(cd ../.. && pwd)" "${PYTHON_BIN}" validate_output.py "${INPUT_DIR}" "${OUTPUT_DIR}"
echo "Container DICOM and REDCap smoke test: OK (${OUTPUT_DIR})"
