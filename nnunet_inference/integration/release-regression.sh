#!/bin/bash
# Compare a candidate image with a released image on one DICOM series.
# The reference runs twice for each TTA setting. Whatever those two runs agree
# on (pixels, UIDs, REDCap rows) must be the same in the candidate.
# Relative paths are resolved from the current directory. WORK_DIR belongs to
# one input series, and finished runs in it are reused for the same image ID,
# so a rebuilt candidate runs again. Dates are drawn into the fused images, so
# still use a new WORK_DIR each day.
set -euo pipefail
if [ -z "${4:-}" ] || [ ! -d "${3:-}" ]; then
    echo "Usage: $0 REFERENCE_IMAGE CANDIDATE_IMAGE DICOM_INPUT_DIR WORK_DIR" >&2; exit 2
fi
readonly REFERENCE="$1"
readonly CANDIDATE="$2"
INPUT_DIR="$(realpath -e -- "$3")"
WORK_DIR="$(realpath -m -- "$4")"
readonly INPUT_DIR WORK_DIR
readonly PYTHON_BIN="${NNUNET_PYTHON:-python}"
cd "$(dirname "$0")"

# Snapshots and REDCap files hold patient identifiers: keep them out of Git.
# The trailing slash makes Git match a directory that does not exist yet.
repository="$(git rev-parse --show-toplevel 2>/dev/null)" || repository=
if [ -n "${repository}" ] && [[ "${WORK_DIR}/" == "${repository}/"* ]] \
        && ! git -C "${repository}" check-ignore -q -- "${WORK_DIR}/"; then
    echo "Error: the work directory ${WORK_DIR} is inside the repository ${repository}" \
        "and not ignored by Git; its files hold patient identifiers" >&2
    exit 2
fi

# image_id IMAGE: the full ID (sha256:...) of the local image.
image_id() {
    local id
    id="$(docker image inspect "$1" --format '{{.Id}}')" || id=
    if [[ ! "${id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "Error: image $1 is not present locally" >&2; exit 2
    fi
    echo "${id}"
}
# container_version IMAGE: the CONTAINER_VERSION of the image's configuration.
container_version() {
    local line
    while IFS= read -r line; do
        if [[ "${line}" == CONTAINER_VERSION=?* ]]; then
            echo "${line#CONTAINER_VERSION=}"; return 0
        fi
    done <<< "$(docker image inspect "$1" --format '{{range .Config.Env}}{{println .}}{{end}}')"
    echo "Error: image $1 has no CONTAINER_VERSION in its configuration" >&2; exit 2
}
# Runs start by ID, so that a tag moved meanwhile does not change the image.
REFERENCE_ID="$(image_id "${REFERENCE}")"
CANDIDATE_ID="$(image_id "${CANDIDATE}")"
# The candidate runs under the reference's name as the image records it, so
# that the name is not a difference however the reference was given.
REFERENCE_VERSION="$(container_version "${REFERENCE_ID}")"
readonly REFERENCE_ID CANDIDATE_ID REFERENCE_VERSION
mkdir -p -- "${WORK_DIR}"

# The work directory belongs to one input series, identified by a SHA-256 over
# the sorted sha256sum lines of its files. A work directory of the previous
# version of this script has no record yet and gets one.
digest="$(cd -- "${INPUT_DIR}" && find -L . -type f -print0 | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum | sha256sum)"
digest="${digest%% *}"
record="${WORK_DIR}/input-series.txt"
if [ -f "${record}" ]; then
    read -r recorded_digest recorded_path < "${record}"
    if [ "${recorded_digest}" != "${digest}" ]; then
        echo "Error: ${WORK_DIR} holds runs of ${recorded_path} (sha256 ${recorded_digest})," \
            "not of ${INPUT_DIR} (sha256 ${digest}); use another work directory" >&2
        exit 2
    fi
else
    printf '%s %s\n' "${digest}" "${INPUT_DIR}" > "${record}.tmp"
    mv -- "${record}.tmp" "${record}"
fi

# run IMAGE_ID NAME TTA [CONTAINER_VERSION]: run once and snapshot. A finished
# run (one with a snapshot) is kept; an unfinished one is removed and run again.
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
uncompared=()
for tta in true false; do
    reference="reference-${REFERENCE_ID:7:12}-tta-${tta}"
    candidate="candidate-${CANDIDATE_ID:7:12}-tta-${tta}"
    baseline="${WORK_DIR}/baseline-${REFERENCE_ID:7:12}-tta-${tta}.json"
    run "${REFERENCE_ID}" "${reference}-1" "${tta}"
    run "${REFERENCE_ID}" "${reference}-2" "${tta}"
    "${PYTHON_BIN}" compare_outputs.py stable "${WORK_DIR}/${reference}-1.json" \
        "${WORK_DIR}/${reference}-2.json" "${baseline}"
    pixels="$("${PYTHON_BIN}" compare_outputs.py uncompared "${baseline}")"
    if [ -n "${pixels}" ]; then
        uncompared+=("Not compared with TTA ${tta}, because the two release runs disagree: ${pixels}")
    fi
    run "${CANDIDATE_ID}" "${candidate}" "${tta}" "${REFERENCE_VERSION}"
    "${PYTHON_BIN}" compare_outputs.py check "${baseline}" "${WORK_DIR}/${candidate}.json" \
        || status=1
done
if [ "${status}" -eq 0 ]; then
    echo "Release regression: OK (${CANDIDATE} matches ${REFERENCE})"
fi
if [ "${#uncompared[@]}" -gt 0 ]; then
    printf '%s\n' "${uncompared[@]}"
fi
exit "${status}"
