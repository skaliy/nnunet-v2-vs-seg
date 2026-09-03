FROM haukebartsch/fiona-component-python:latest

WORKDIR /app

ARG conda_env=nnunet-vsseg
ENV NNUNET_CONDA_ENV=${conda_env} \
    PYTHONPATH=/app/_pkg \
    REPORT_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
    DCMDICTPATH=/dcmtk/dcmdata/data/dicom.dic:/dcmtk/dcmdata/data/private.dic

LABEL com.ror.vendor="MMIV.no" \
      com.ror.conda.env.name="${conda_env}" \
      com.mmiv.model.folds="0,1,2,3,4" \
      com.mmiv.tta.default="false" \
      description="Five-fold nnU-Net v2 vestibular schwannoma DICOM inference for MMIV Research PACS (CPU, TTA off by default)."

COPY requirements.yml /app/requirements.yml
RUN conda env create -f /app/requirements.yml && \
    conda clean --all --yes

# Required immutable release identity. Declared after the dependency layer so a
# new release version does not rebuild the environment.
ARG VERSION
RUN test -n "${VERSION}" || \
    (echo "VERSION build argument is required" >&2; exit 1)
LABEL org.opencontainers.image.version="${VERSION}"
ENV VERSION=${VERSION}

# Keep the environment active for interactive debugging as well as ENTRYPOINT.
RUN echo "conda activate ${conda_env}" >> /root/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

COPY _pkg /app/_pkg
COPY model /app/model
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod 0755 /app/entrypoint.sh && \
    test -f /app/model/fold_0/checkpoint_final.pth && \
    test -f /app/model/fold_1/checkpoint_final.pth && \
    test -f /app/model/fold_2/checkpoint_final.pth && \
    test -f /app/model/fold_3/checkpoint_final.pth && \
    test -f /app/model/fold_4/checkpoint_final.pth && \
    test ! -d /app/model/fold_all && \
    test "$(tr -d '[:space:]' < /app/model/MODEL_FOLDS.txt)" = "0,1,2,3,4" && \
    cd /app/model && sha256sum -c MODEL_SHA256SUMS.txt && \
    test "$(jq -r .bundle_sha256 deployment_manifest.json)" = "$(tr -d '[:space:]' < MODEL_VERSION.txt)" && \
    test -x /pr2mask/imageAndMask2Report && \
    test -x /pr2mask/imageAndMask2Fused && \
    python -c "import nnunetv2, imagedata, SimpleITK, pydicom; import nnunet_inference.pipeline"

ENTRYPOINT ["/app/entrypoint.sh"]
