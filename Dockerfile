ARG RUNTIME_CACHE_IMAGE=w_cisegmentation-runtime-cache:latest
FROM ${RUNTIME_CACHE_IMAGE}

WORKDIR /app
COPY cisegmentation/ /app/cisegmentation/
COPY wrapper.py bilayers_cli.py config.yaml /app/
COPY tools/cuda_smoke.py /app/tools/cuda_smoke.py
RUN python -m compileall -q -j 0 /app

ENTRYPOINT ["python", "/app/wrapper.py"]
