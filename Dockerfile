ARG RUNTIME_CACHE_IMAGE=w_cisegmentation-runtime-cache:latest
FROM ${RUNTIME_CACHE_IMAGE}

WORKDIR /app
COPY requirements_cellpose_transformer.txt /app/
# Keep checkpoint architecture dependencies available when reusing a previously
# published inference runtime as RUNTIME_CACHE_IMAGE.
ARG INSTALL_CHECKPOINT_DEPENDENCIES=true
RUN if [ "$INSTALL_CHECKPOINT_DEPENDENCIES" = true ]; then \
      python -m pip install -r /app/requirements_cellpose_transformer.txt \
      && python -m pip uninstall -y triton; \
    fi
COPY cisegmentation/ /app/cisegmentation/
COPY wrapper.py bilayers_cli.py config.yaml /app/
COPY tools/cuda_smoke.py /app/tools/cuda_smoke.py
RUN python -m compileall -q -j 0 /app

ENTRYPOINT ["python", "/app/wrapper.py"]
