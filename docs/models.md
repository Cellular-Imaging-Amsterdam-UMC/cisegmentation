# Models and Docker build caches

`tools/download_models.py` prepares an idempotent cache in the repository's
Git-ignored `models/` folder containing all
registered Cellpose 3 models, Cellpose-SAM v2, the original Cellpose-SAM,
`SD_Nuclei_Versatile`,
`SD_Foci_Aggregates`, `SD_Foci_Finn`, all three InstanSeg models, and all six
Spotiflow models. Local Spotiflow mask refinement is deterministic CPU
post-processing and requires no additional model checkpoint.
The three StarDist source folders are bundled from the pinned
`cistardist_pytorch` models and converted to PyTorch checkpoints only when the
corresponding `.pt` file is missing.

After successful CPU smoke-loading, the downloader removes Spotiflow download
ZIPs and training-only `last.pt` checkpoints; runtime loads the extracted
`best.pt` folders directly and therefore remains offline. StarDist Keras H5
conversion sources are likewise removed after the converted `.pt` checkpoint
has loaded successfully. The completion inventory is written only after this
cleanup, so interrupted or invalid caches are repaired on the next run.

Cellpose-SAM v2 is the first Cellpose-SAM choice in the launcher. Cellpose
documents it as reducing spurious masks in low-contrast regions. The original
checkpoint remains selectable for reproducibility. Both checkpoints are about
1.23 GB each, so v2 adds approximately 1.23 GB to the prepared model cache.

The headless image removes Triton after installing the pinned PyTorch stack.
Triton is used by `torch.compile`/Inductor, while this inference-only workflow
uses eager PyTorch execution. GPU smoke tests cover Cellpose 3, both
Cellpose-SAM versions, StarDist, InstanSeg, and Spotiflow without it. The
NVIDIA CUDA runtime packages remain installed because PyTorch 2.11 links
against them directly, including
cuDNN, cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, NCCL, NVSHMEM,
CUPTI, NVJitLink, and cuFile.

`builddocker.cmd` updates this host cache before building: valid files are
reused and only absent or invalid artifacts are downloaded. It creates a local,
fingerprinted `w_cisegmentation-model-cache` image only when the cache manifest
changes. Normal code rebuilds exclude the 4+ GB host cache from their context
and reuse that image. Local `wrapper.py` runs automatically fall back to the
repository cache; set `CISEGMENTATION_MODELS` only when using a different cache
location. Runtime jobs therefore do not need network access.

## Docker images

```text
builddocker.cmd             headless Bilayers/BIOMERO workflow image
```

Default builds and CI build only the workflow image, tagged
`w_cisegmentation:<version>` and `w_cisegmentation:latest`. Model and runtime
cache images are internal dependencies of this build.

Optional Gradio and Jupyter Dockerfiles, requirements, and manual build scripts
are isolated in [bilayers_extra](../bilayers_extra/README.md). They reuse an already
built workflow image and are excluded from default builds, CI builds, and Docker
Hub publication.
