#!/bin/bash --login
# /data/input -> nnU-Net ResEnc-M or ResEnc-L -> pr2mask + REDCap -> /output
set -euo pipefail

model_type="$(jq -r .model_type "${NNUNET_MODEL_DIR:-/app/model}/deployment_manifest.json")"
ror_options="${ROR_CONT_OPTIONS:-}"
if [ -z "${ror_options}" ]; then ror_options='{}'; fi
if ! jq -e --arg model_type "${model_type}" 'type == "object"
    and ((keys - ["model-type", "tta"]) | length == 0)
    and (if has("model-type") then .["model-type"] == $model_type else true end)
    and (if has("tta") then (.tta | type == "boolean") else true end)' \
    >/dev/null <<< "${ror_options}"; then
    echo "Error: ROR_CONT_OPTIONS supports model-type=\"${model_type}\" and Boolean tta" >&2
    exit 2
fi
tta_arg=--tta
if [ "$(jq -r 'if has("tta") then .tta else true end' <<< "${ror_options}")" = false ]; then
    tta_arg=--no-tta
fi

export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=3
export nnUNet_def_n_proc=3 nnUNet_compile=false
export PATH="/pr2mask:${PATH}"

set +e
set +u
set +o pipefail
conda activate "${NNUNET_CONDA_ENV:-nnunet-vsseg}"
activate_status=$?
set -euo pipefail
if [ "${activate_status}" -ne 0 ]; then
    echo 'Error: activating the nnUNet deployment environment failed' >&2
    exit 2
fi
exec python -m nnunet_inference.pacs /data/input /output "${tta_arg}" --model-type "${model_type}"
