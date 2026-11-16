#!/bin/bash
# Compare a candidate image with a released image on one DICOM series.
# The reference runs twice for each TTA setting. Whatever those two runs agree
# on (pixels, UIDs, REDCap rows) must be the same in the candidate.
# Finished runs in WORK_DIR are reused for the same image ID, so a rebuilt
# candidate runs again. Dates are drawn into the fused images, so still use a
# new WORK_DIR each day.
set -euo pipefail
cd "$(dirname "$0")"
readonly REFERENCE="${1:-}"
readonly CANDIDATE="${2:-}"
readonly INPUT_DIR="${3:-}"
readonly WORK_DIR="${4:-}"
readonly PYTHON_BIN="${NNUNET_PYTHON:-python}"
if [ -z "${WORK_DIR}" ] || [ ! -d "${INPUT_DIR}" ]; then
    echo "Usage: $0 REFERENCE_IMAGE CANDIDATE_IMAGE DICOM_INPUT_DIR WORK_DIR" >&2; exit 2
fi

# image_id IMAGE: the first 12 hex digits of the local image's ID.
image_id() {
    local id
    id="$(docker image inspect "$1" --format '{{.Id}}')" || id=
    if [[ ! "${id}" =~ ^sha256:([0-9a-f]{12}) ]]; then
        echo "Error: image $1 is not present locally" >&2; exit 2
    fi
    echo "${BASH_REMATCH[1]}"
}
REFERENCE_ID="$(image_id "${REFERENCE}")"
CANDIDATE_ID="$(image_id "${CANDIDATE}")"
readonly REFERENCE_ID CANDIDATE_ID
mkdir -p "${WORK_DIR}"

# run IMAGE NAME TTA [CONTAINER_VERSION]: run once and snapshot. A finished run
# (one with a snapshot) is kept; an unfinished one is removed and run again.
run() {
    local output="${WORK_DIR}/$2"
    if [ -f "${output}.json" ]; then
        return 0
    fi
    if [ -e "${output}" ]; then
        echo "Removing the unfinished run ${output}"
        rm -rf -- "${output}"
    fi
    IMAGE_TAG="$1" TTA="$3" CONTAINER_VERSION_OVERRIDE="${4:-}" \
        ./smoke-test.sh "${INPUT_DIR}" "${output}"
    "${PYTHON_BIN}" compare_outputs.py snapshot "${output}" "${output}.json"
}

status=0
for tta in true false; do
    reference="reference-${REFERENCE_ID}-tta-${tta}"
    candidate="candidate-${CANDIDATE_ID}-tta-${tta}"
    baseline="${WORK_DIR}/baseline-${REFERENCE_ID}-tta-${tta}.json"
    run "${REFERENCE}" "${reference}-1" "${tta}"
    run "${REFERENCE}" "${reference}-2" "${tta}"
    "${PYTHON_BIN}" compare_outputs.py stable "${WORK_DIR}/${reference}-1.json" \
        "${WORK_DIR}/${reference}-2.json" "${baseline}"
    run "${CANDIDATE}" "${candidate}" "${tta}" "${REFERENCE}"
    "${PYTHON_BIN}" compare_outputs.py check "${baseline}" "${WORK_DIR}/${candidate}.json" \
        || status=1
done
if [ "${status}" -eq 0 ]; then
    echo "Release regression: OK (${CANDIDATE} matches ${REFERENCE})"
fi
exit "${status}"
