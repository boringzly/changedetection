ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="GeoChangeMLLM trainer" \
      org.opencontainers.image.description="Remote-sensing change detection and language-model joint trainer"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /opt/geochangemllm

# PyTorch and CUDA are supplied by the NGC base image. Install only project-level
# dependencies so the image keeps the tested CUDA/PyTorch pair from NGC.
RUN python -m pip install --no-cache-dir \
      "accelerate==1.14.0" \
      "peft==0.20.0" \
      "Pillow==12.3.0" \
      "rasterio==1.5.0" \
      "transformers==5.14.1"

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin trainer \
    && mkdir -p /workspace /data /cache \
    && chown -R 10001:10001 /workspace /data /cache /home/trainer

USER 10001:10001
WORKDIR /workspace

CMD ["python", "-m", "geochangemllm.doctor"]
