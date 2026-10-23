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

In BIOMERO, configure the workflow repository at `tree/v0.6.5`, initialize
the image, then select the installed version and input image or plate.
The published headless image is:

```sh
docker pull cellularimagingcf/w_cisegmentation:v0.6.5
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


### Channel numbering and Cellpose calibration preflight

Channel selections are **one-based** everywhere: `1` selects the first image
channel, `2` the second, and so on. The optional **extra nucleus channel** in
Step 1 alone uses `0` to mean **unused**, never the first channel. Disable an
entire segmentation step by selecting `Skip`. Every active channel is checked
against every input field before inference.

Cellpose inputs require valid physical X/Y calibration, resolved from NGFF
metadata or matching embedded OME-XML and converted to micrometers. The default
plausibility range is **0.001 to 2.0 um/pixel**, inclusive, independently for X
and Y. Native 3D also requires positive calibrated Z spacing; the lateral
2 um limit is not applied to the distance between Z planes. Use
`--cellpose-min-pixel-size-um` and `--cellpose-max-pixel-size-um` only when the
source metadata has been independently verified.

The metadata-only preflight checks the entire run before staging output,
model initialization, inference probes, or measurements. Invalid calibration
in a later field aborts the complete run with a source/field-specific log
message; no calibration is invented or clamped. `diameter=-1` selects the
supported model-default scale but does not waive the calibration check.
Excessive enlargement (default maximum `--cellpose-max-rescale=8`), projected
working dimensions, and a conservative estimate of working-array RAM are also
checked without allocating the projected arrays. This is a safety estimate,
not a measurement of the eventual model's peak memory.

Normal segmentation defaults to DuckDB measurements. The Bilayers descriptor
also declares DuckDB/SQLite as optional measurement outputs, allowing BIOMERO
to suggest returning those files as attachments. Skip and benchmark mode do
not produce a measurement database.
