ARG RUNTIME_CACHE_IMAGE=w_cisegmentation-runtime-cache:latest
FROM ${RUNTIME_CACHE_IMAGE}

WORKDIR /app
COPY requirements_cellpose_transformer.txt /app/
# Install the small CPU scheduling dependency even when reusing an older
# inference runtime cache, so spawned workers can limit nested BLAS threads.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install 'threadpoolctl>=3.5,<4'
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
