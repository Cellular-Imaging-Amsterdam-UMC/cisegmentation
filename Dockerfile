FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CISEGMENTATION_MODELS=/opt/cisegmentation/models \
    CELLPOSE3_LEGACY_LOCAL_MODELS_PATH=/opt/cisegmentation/models/cellpose3 \
    CELLPOSE_LOCAL_MODELS_PATH=/opt/cisegmentation/models/cellpose-sam \
    SPOTIFLOW_CACHE_DIR=/opt/cisegmentation/models/spotiflow \
    SPOTIFLOW_LOCAL_MODELS_PATH=/opt/cisegmentation/models/spotiflow \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt
COPY cisegmentation/ /app/cisegmentation/
COPY wrapper.py bilayers_cli.py config.yaml /app/
COPY tools/download_models.py tools/cuda_smoke.py /app/tools/
RUN python /app/tools/download_models.py \
    && rm -rf /root/.cache /tmp/*
RUN mkdir -p /data/in /data/out
ENTRYPOINT ["python", "/app/wrapper.py"]
