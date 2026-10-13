# CI Segmentation

GPU-enabled instance segmentation for OME-Zarr images and HCS plates, designed
for Bilayers and BIOMERO. One CUDA 12.6 environment supports Cellpose 3,
Cellpose-SAM v2 and the original Cellpose-SAM, PyTorch StarDist, InstanSeg and
Spotiflow, with optional local spot-mask refinement.

## What it produces

- Input: top-level `.ome.zarr` stores, including complete HCS plates.
- Output: `<source>__cisegmentation.ome.zarr`, preserving source pixels and
  adding native label groups for enabled cells, nuclei, cytoplasm, spots,
  foci or bacteria. The source remains untouched.
- DuckDB measurements by default, with SQLite or skip options; spatial measurements,
  colocalization, tracking and companion geometry databases are opt-in.
- Benchmark mode creates a model-comparison gallery instead of normal labels
  and measurements.

Time and Z are preserved. User-facing channels are one-based; physical
parameters use OME-Zarr calibration. Enable at least one segmentation step.

## Get started

In BIOMERO, configure the workflow repository at `tree/v0.6.4`, initialize
the image, then select the installed version and input image or plate.
The published headless image is:

```sh
docker pull cellularimagingcf/w_cisegmentation:v0.6.4
```

For the Windows local frontend, run `create_env.cmd`, then `launch.cmd`.
The launcher supports local Python execution and the release-pinned local
Docker image. `wrapper.py` is the container entrypoint; `launcher.py` is the
supported local frontend. See [local execution](docs/local-execution.md) for
environment setup, direct examples and the OMERO round trip.

Large images stream bounded tiles. Plate fields use automatically sized
worker pools constrained by visible GPU/MIG memory and allocated CPU/RAM.
Version 0.6.4 includes MIG-aware sizing and structured resource diagnostics.
The original sizing failure was resolved on a 2,844-field H100 MIG run;
forced-OOM recovery and scientific comparisons remain separate validation
checks. See [resource evidence](docs/slurm-resource-checks.md).

## Documentation

| Topic | Guide |
| --- | --- |
| Workflow steps, parameters and output behavior | [Workflow](docs/workflow.md) |
| Launcher, direct examples, benchmarks and round trip | [Local execution](docs/local-execution.md) |
| Streaming, concurrency and timing interpretation | [Large inputs](docs/large-inputs.md) |
| Allocation logs, MIG evidence and recovery checks | [Slurm resource checks](docs/slurm-resource-checks.md) |
| Measurement schema, units and query examples | [Measurements](docs/measurements.md) |
| Spatial measurements, colocalization, tracking and geometry | [Measurement extensions](docs/measurement-extensions.md) |
| Retained validation inputs | [Public datasets](docs/publicdata.md) |
| Checkpoints, offline runtime and Docker caches | [Models](docs/models.md) |
| Repository structure and test layers | [Development](docs/development.md) |
| Versions, builds, tags and publication | [Releasing](docs/releasing.md) |

The [agent plugin](plugin.json) bundles focused skills under `skills/` for
workflow operation, measurement analysis and geometry analysis. Their
references are self-contained for consumers that load skills without this
repository's documentation tree.

Optional Gradio and Jupyter builds are maintained separately in
[bilayers_extra](bilayers_extra/README.md) and are excluded from default builds
and Docker Hub publication.

Licensed under [MIT](LICENSE).
