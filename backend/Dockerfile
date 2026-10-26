# syntax=docker/dockerfile:1.7
# HybridIR backend image: used by both the `seed` (one-shot) and `api` services.
#
# Model weights are NOT baked into the image. The seed service downloads them into the
# `hf_cache` named volume (HF_HOME=/hf_cache) on first run; the api service mounts the same
# volume and runs with HF_HUB_OFFLINE=1, so after the first run nothing is downloaded again.

FROM python:3.11.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    DATA_DIR=/data \
    RESULTS_DIR=/results

WORKDIR /app

# CPU-only torch first (avoids ~2 GB of CUDA wheels), then the pinned requirements.
COPY requirements-base.txt requirements-ml.txt requirements-dev.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r requirements-base.txt -r requirements-ml.txt \
 && pip install pytest==8.3.4

# Non-root user. The volume mount points are created (and owned) here so that Docker copies
# this ownership into the named volumes the first time they are mounted.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser \
 && mkdir -p /data /results /hf_cache \
 && chown -R appuser:appuser /data /results /hf_cache \
 && chmod 0777 /results

COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser tests ./tests

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=30 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4); sys.exit(0)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "15", "--no-access-log"]
