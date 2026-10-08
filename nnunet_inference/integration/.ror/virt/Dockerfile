ARG FIONA_BASE=haukebartsch/fiona-component-python:latest
FROM ${FIONA_BASE}

WORKDIR /app

ARG conda_env=nnunet-vsseg
ENV NNUNET_CONDA_ENV=${conda_env} \
    PYTHONPATH=/app/_pkg \
    REPORT_FONT_PATH=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
    DCMDICTPATH=/dcmtk/dcmdata/data/dicom.dic:/dcmtk/dcmdata/data/private.dic

LABEL com.ror.vendor="MMIV.no" \
      com.ror.conda.env.name="${conda_env}" \
      com.mmiv.model.folds="0,1,2,3,4" \
      com.mmiv.tta.default="true" \
      com.mmiv.model.type="nnunet_medium" \
      com.mmiv.cpu.threads="16" \
      com.mmiv.memory.gib="64" \
      description="Curated ResEnc-M five-fold nnU-Net v2 VS PACS inference and REDCap export (CPU, TTA on)."

COPY requirements.yml /app/requirements.yml
RUN conda env create -f /app/requirements.yml && \
    conda clean --all --yes

# Required immutable release identity. Declared after the dependency layer so a
# new release version does not rebuild the environment.
ARG VERSION
ARG IMAGE_NAME=nnunet-vsseg-5fold
RUN test -n "${VERSION}" || \
    (echo "VERSION build argument is required" >&2; exit 1)
LABEL org.opencontainers.image.version="${VERSION}"
ENV VERSION=${VERSION}
# pr2mask records this as the container version in its DICOM SoftwareVersions.
ENV CONTAINER_VERSION=${IMAGE_NAME}:${VERSION}

# Keep the environment active for interactive debugging as well as ENTRYPOINT.
RUN echo "conda activate ${conda_env}" >> /root/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

COPY _pkg /app/_pkg
COPY model /app/model
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod 0755 /app/entrypoint.sh && \
    test -x /pr2mask/imageAndMask2Report && \
    test -x /pr2mask/imageAndMask2Fused && \
    python -m nnunet_inference.deployment --output /app/model --verify-only --verify-runtime && \
    python -c "from nnunet_inference.deployment import validate_deployment; from nnunet_inference.redcap_output import model_repeat_instance; model_repeat_instance(validate_deployment('/app/model')['bundle_sha256'])"

ENTRYPOINT ["/app/entrypoint.sh"]
