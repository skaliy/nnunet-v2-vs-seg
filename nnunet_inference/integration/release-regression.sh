#!/bin/bash
# Compare a candidate image with a released image on one DICOM series.
# The reference runs twice for each TTA setting. Whatever those two runs agree
# on (pixels, UIDs, REDCap rows) must be the same in the candidate.
# Dates are drawn into the fused images, so use a new WORK_DIR each day.
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
mkdir -p "${WORK_DIR}"

# run IMAGE NAME TTA [CONTAINER_VERSION]: run once and snapshot; a finished run is kept.
run() {
    local output="${WORK_DIR}/$2"
    if [ ! -f "${output}.json" ]; then
        IMAGE_TAG="$1" TTA="$3" CONTAINER_VERSION_OVERRIDE="${4:-}" \
            ./smoke-test.sh "${INPUT_DIR}" "${output}"
        "${PYTHON_BIN}" compare_outputs.py snapshot "${output}" "${output}.json"
    fi
}

status=0
for tta in true false; do
    run "${REFERENCE}" "reference-tta-${tta}-1" "${tta}"
    run "${REFERENCE}" "reference-tta-${tta}-2" "${tta}"
    "${PYTHON_BIN}" compare_outputs.py stable "${WORK_DIR}/reference-tta-${tta}-1.json" \
        "${WORK_DIR}/reference-tta-${tta}-2.json" "${WORK_DIR}/baseline-tta-${tta}.json"
    run "${CANDIDATE}" "candidate-tta-${tta}" "${tta}" "${REFERENCE}"
    "${PYTHON_BIN}" compare_outputs.py check "${WORK_DIR}/baseline-tta-${tta}.json" \
        "${WORK_DIR}/candidate-tta-${tta}.json" || status=1
done
if [ "${status}" -eq 0 ]; then
    echo "Release regression: OK (${CANDIDATE} matches ${REFERENCE})"
fi
exit "${status}"
