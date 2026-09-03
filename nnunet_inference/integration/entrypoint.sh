#!/bin/bash --login
# Research-PACS entrypoint: /data/input DICOM -> nnU-Net -> pr2mask -> /output.
set -euo pipefail

readonly INPUT_DIR=/data/input
readonly OUTPUT_DIR=/output
readonly WORK_OUTPUT=/output_tmp
readonly MODEL_DIR=/app/model
readonly MANIFEST="${MODEL_DIR}/deployment_manifest.json"
readonly FOLDS=0,1,2,3,4
readonly CONDA_ENV_NAME="${NNUNET_CONDA_ENV:-nnunet-vsseg}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: expected an input DICOM directory at ${INPUT_DIR}" >&2
    exit 2
fi
if [ -z "${VERSION:-}" ]; then
    echo "Error: image VERSION is not set" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}" "${WORK_OUTPUT}"

# Five-fold ensembling is fixed. TTA is the only supported ROR option and is
# off by default for PACS latency. Reject typos and unsafe/unknown options.
ror_options="${ROR_CONT_OPTIONS:-}"
if [ -z "${ror_options}" ]; then
    ror_options='{}'
fi
if ! jq -e '
    def valid_bool:
      if type == "boolean" then true
      elif type == "number" then (. == 0 or . == 1)
      elif type == "string" then
        ascii_downcase as $value
        | ["true", "false", "1", "0", "yes", "no", "on", "off"]
        | index($value) != null
      else false
      end;
    type == "object"
    and ((keys - ["tta"]) | length == 0)
    and (if has("tta") then (.tta | valid_bool) else true end)
  ' >/dev/null <<< "${ror_options}"; then
    echo "Error: invalid ROR_CONT_OPTIONS; supported option: tta" >&2
    exit 2
fi

tta_value="$(jq -r 'if has("tta") then .tta else false end' <<< "${ror_options}" \
    | tr '[:upper:]' '[:lower:]')"
case "${tta_value}" in
    true|1|yes|on) USE_TTA=1; TTA_ARG=--tta ;;
    false|0|no|off) USE_TTA=0; TTA_ARG=--no-tta ;;
    *) echo "Error: ROR_CONT_OPTIONS.tta must be a Boolean" >&2; exit 2 ;;
esac

echo "Model folds: ${FOLDS}; TTA: $([ "${USE_TTA}" -eq 1 ] && echo on || echo off)"

# Validate the declared model bundle, not merely the presence of five files.
for required in "${MANIFEST}" "${MODEL_DIR}/MODEL_NAME.txt" \
                "${MODEL_DIR}/MODEL_VERSION.txt" \
                "${MODEL_DIR}/MODEL_SHA256SUMS.txt"; do
    if [ ! -s "${required}" ]; then
        echo "Error: required model declaration is missing: ${required}" >&2
        exit 2
    fi
done
if [ -d "${MODEL_DIR}/fold_all" ]; then
    echo "Error: this image must not contain a fold_all model" >&2
    exit 2
fi
if ! jq -e '
    .schema_version == 1
    and .dataset == "Dataset002_VSSegmentationRaw"
    and .trainer == "nnUNetTrainer"
    and .plans == "nnUNetResEncUNetLPlans"
    and .configuration == "3d_fullres"
    and .checkpoint == "checkpoint_final.pth"
    and .folds == [0, 1, 2, 3, 4]
    and .nnunet_version == "2.6.2"
    and .dicom_uid == {
      "format_version": 1,
      "root": "2.25",
      "generation": "uuid5",
      "namespace_uuid": "bb88c59d-5a75-5f47-bb52-3bc9f6db7808",
      "model_code": 1,
      "deployment_code": 1,
      "output_codes": {"segmentation": 1, "probability": 2}
    }
  ' "${MANIFEST}" >/dev/null; then
    echo "Error: deployment manifest does not match the approved contract" >&2
    exit 2
fi
(
    cd "${MODEL_DIR}"
    sha256sum -c MODEL_SHA256SUMS.txt
)
MODEL_VERSION="$(tr -d '[:space:]' < "${MODEL_DIR}/MODEL_VERSION.txt")"
if [ "$(jq -r '.bundle_sha256' "${MANIFEST}")" != "${MODEL_VERSION}" ]; then
    echo "Error: model version does not match deployment manifest" >&2
    exit 2
fi
MODEL_NAME="$(tr -d '\n\r' < "${MODEL_DIR}/MODEL_NAME.txt")"
if [ "$(jq -r '.model_name' "${MANIFEST}")" != "${MODEL_NAME}" ]; then
    echo "Error: model name does not match deployment manifest" >&2
    exit 2
fi

export PATH="/pr2mask:${PATH}"
for tool in /pr2mask/imageAndMask2Report /pr2mask/imageAndMask2Fused; do
    if [ ! -x "${tool}" ]; then
        echo "Error: required Fiona/pr2mask tool is unavailable: ${tool}" >&2
        exit 2
    fi
done
readonly LOG_FILE="${OUTPUT_DIR}/stub_command.log"
INFO="${MODEL_NAME} ${MODEL_VERSION:0:8}, Predicted $(date '+%b%d%Y')"

# The Fiona image supplies conda. Its activation hook is not strict-mode safe.
set +e
set +u
set +o pipefail
conda activate "${CONDA_ENV_NAME}"
activate_status=$?
set -euo pipefail
if [ "${activate_status}" -ne 0 ]; then
    echo "Error: activating conda environment ${CONDA_ENV_NAME} failed" >&2
    exit 2
fi

cmd=(
    python -m nnunet_inference
    "${INPUT_DIR}" "${WORK_OUTPUT}"
    --device cpu
    --model-dir "${MODEL_DIR}"
    --checkpoint final
    --folds "${FOLDS}"
    "${TTA_ARG}"
)
printf 'run now:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "imageAndMask2Report:"
/pr2mask/imageAndMask2Report "${INPUT_DIR}" "${WORK_OUTPUT}/mask" "${WORK_OUTPUT}" \
    -u "${VERSION}" -i "${VERSION}" --reporttype mosaic -t "${INFO} " \
    >> "${LOG_FILE}" 2>&1
echo "imageAndMask2Fused:"
/pr2mask/imageAndMask2Fused "${INPUT_DIR}" "${WORK_OUTPUT}/mask" "${WORK_OUTPUT}" \
    -u "${VERSION}_fused" -i "${VERSION}" >> "${LOG_FILE}" 2>&1
echo "imageAndMask2Fused (vote map):"
/pr2mask/imageAndMask2Fused "${INPUT_DIR}" "${WORK_OUTPUT}/vote_map" "${WORK_OUTPUT}" \
    --votemapmax 65535 --votemapagree 0.5 -u "${VERSION}_votemap" \
    -s "peak agreement {peak_agreement}" -i "${VERSION}" \
    >> "${LOG_FILE}" 2>&1

# Return exactly the four series in the approved Research-PACS contract.
for result in fused fused_vote_map reports mask; do
    if [ ! -d "${WORK_OUTPUT}/${result}" ]; then
        echo "Error: pr2mask did not create ${WORK_OUTPUT}/${result}" >&2
        exit 3
    fi
    cp -R "${WORK_OUTPUT}/${result}" "${OUTPUT_DIR}/"
done
chmod -R a+rwX "${OUTPUT_DIR}"
echo "$(date): processing done" >> "${LOG_FILE}"
