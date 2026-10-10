# CI Segmentation

CI Segmentation is a GPU-enabled Bilayers/BIOMERO workflow for instance
segmentation from OME-Zarr to labeled OME-Zarr. It supports Cellpose 3,
Cellpose-SAM v2 and the original Cellpose-SAM, PyTorch StarDist, InstanSeg,
and Spotiflow with optional local mask refinement from one CUDA 12.6 environment.

## Workflow contract

- Input: one or more top-level `.ome.zarr` stores in `/data/in`, including HCS plates.
- Normal output: `<source>__cisegmentation.ome.zarr` in `/data/out`, containing
  native OME-Zarr label groups. By default the source store is copied with its
  original pixels and remains untouched. A configured BIOMERO shallower can remove
  verified duplicate arrays before transfer. The UI no longer offers Include
  Original Data; legacy CLI flags retain their labels-only overlay behavior.
- Benchmark output: **only** `benchmark_gallery_<image>.ome.zarr`.
- Axes are normalized to `TCZYX`; time and Z are preserved in normal runs.
- Output OME-Zarrs and schema-v5 measurement databases share an
  `output_store_uuid`; `object_navigation` supplies portable field, label, and
  ROI coordinates without embedding deployment-specific OMERO IDs.
- Spotiflow points become uniquely numbered single pixels or voxels by default.
  Advanced local refinement can replace 2D or slice-wise points with bounded,
  locally thresholded instance masks.

The internal `wrapper.py` is the container/Bilayers entrypoint. It is not
packaged as an end-user CLI. `launcher.py` is the supported local frontend and
constructs the Docker invocation from `config.yaml`.

## Local environment

Run `create_env.cmd` to create or update the `cisegmentation` Conda environment with
Python 3.11, PyTorch 2.11.0, torchvision 0.26.0, and CUDA 12.6 wheels. The
script also installs the PyQt launcher dependencies and finishes with a CUDA
smoke test.

Open this repository as the VS Code workspace to select and activate the
`cisegmentation` environment automatically. Start the supported local frontend
with `launch.cmd`; it uses that environment explicitly and stops with an error
if the environment has not been created. Run tests with `test.cmd`. For other
commands outside VS Code, use `conda run -n cisegmentation` so they cannot
accidentally run in Conda's base environment.

The launcher defaults to `inputfolder` and `outputfolder` in the repository
root. Tests clean `tests/inputfolder` and `tests/outputfolder`, then copy fresh
OME-Zarr fixtures from `tests/data` into the test input folder when required.
The launcher provides separate **Run Docker** and **Run Locally** buttons; local
mode uses the active Python environment and executes `wrapper.py` directly.
**Run Docker** uses the locally built, release-pinned `w_cisegmentation:v0.6.4-beta`
image; the
organization-qualified image in `config.yaml` is reserved for BIOMERO registry
metadata.

During execution, the job log reports the selected workflow and tuning values,
input T/C/Z/Y/X dimensions, datatype, physical scales, and channel names. Every
model call reports its resolved CPU/GPU device, dimensional mode, runtime,
timing breakdown, and model-cache status. Enable **Labels Log Info** to also
calculate label counts, foreground fractions, and size distributions for model
and final post-processed outputs. It is disabled by default because those full
label-array scans can be costly.

HCS plates are processed model by model across all fields. Worker counts are
sized from a model memory probe and the available GPU or CPU allocation. Native
labels are finalized in a CPU pool after inference, then measurements run in
spawned CPU workers and stream bounded field databases to one parent writer.
The run-specific temporary working directory is a hidden sibling in the output
folder, never a child of the input OME-Zarr. For full-data output, a verified
source copy runs alongside label finalization and generated labels are committed
only to that copy. The final store and database become visible only after every
phase succeeds.

The `v0.6.4-beta` workflow logs allocation and worker decisions with the prefix
`[CISEGMENTATION_RESOURCE_CHECK]`, including Slurm CPU/RAM limits and the
CUDA-visible GPU/MIG memory budget. See the
[resource log and MIG acceptance guide](docs/slurm-resource-checks.md).
Beta publication updates its explicit version tag and leaves stable Docker Hub
`latest` unchanged.

## Streaming large fields

Large fields automatically use streaming when XY exceeds 4096 pixels or the
input exceeds a conservative live RAM budget. Resources are discovered from
Slurm CPU/memory allocations, CPU affinity, container/cgroup limits, available
host RAM, and the currently visible CUDA device. These checks run before every
tile. They do not assume a fixed 4 CPU / 16 GB / 12 GB cluster profile.

Streaming profiles a representative tile including its halo, then sizes a pool
from its measured peak RAM/GPU use and the available CPU allocation. The same
safety margins used for plate fields apply to all model families. Workers infer
bounded tiles; one parent matches instance IDs against completed overlaps and
writes tile cores in their original order. The probe prediction is reused.
When plate fields already run concurrently, each field keeps its tiles serial
to avoid multiplying worker pools. **Maximum Inference Workers** also caps tile
workers; `1` forces serial tiles and `0` selects the count automatically.

Live RAM or GPU pressure reduces pending tile sizes. An allocation error first
halves tile concurrency and retries only uncommitted cores; at one worker it
splits the failing tile. Known operation-size limits also split the affected
tile. Pool allocation failures reduce the number of field workers. Final
cell/nucleus matching, label pyramids, and measurement
database writing remain bounded. Measurement crops share a bounded decoded
chunk cache, and database shards merge in batches of 1024 rows.

Advanced controls are **Streaming Mode** (`auto` or `on`), **Tile Size** (core XY,
default 1024), **Tile Overlap** (halo on each side, default 96), **Tile Depth**
(native 3D core Z, default 32), **Tile Overlap Z** (default 8), and **Tile Match
Threshold** (default 0.5). Slice-wise models process one Z plane at a time.
Native 3D Spotiflow retains at least 64 slices of Z context on each side,
matching the bundled checkpoints' two 32-voxel overlap blocks, and at least
192 pixels of XY context, validated against a positive 3D boundary test.
The effective halo is recorded in provenance and included in memory estimates.
Choose overlap large enough to include the expected objects and model context.
Smaller tiles can change model predictions through local normalization/context;
streaming does not promise pixel-identical results to whole-image inference.
An allocation that cannot fit the model and a minimum tile plus overlap fails
with a resource message. Exception recovery cannot guarantee recovery from an
operating-system kill. An individual object whose measurement crop exceeds the
bounded read budget also requires more resources or a smaller object.

The log and OME-Zarr provenance record probe memory, selected/final tile workers,
concurrency reductions, worker restarts, worker memory peaks, tile counts,
allocation retries, pressure splits, smallest cores, overlap, and initial/minimum
available resources. Per-worker temporary input/mask files keep large arrays out
of multiprocessing messages; only a bounded batch is in flight and files are
removed after committing each core.
Scratch disk must hold raw labels, finalized labels, their pyramids, and database
shards until the existing atomic publication finishes. Input stores are preserved.

`tools/tilescan_resource_smoke.py` runs a direct StarDist workflow or overlapping
native model-family tests and records resource samples/output pyramid checks.
For example, on an allocated GPU node:

```bash
python tools/tilescan_resource_smoke.py --input /data/large.ome.zarr \
  --output /data/test-output --device cuda --measurements duckdb
```

For testing a smaller GPU on a larger physical card, its optional
`--gpu-memory-mb 12288` sets `CISEGMENTATION_GPU_MEMORY_LIMIT_MB` and a real
PyTorch per-process allocator cap inherited by inference workers. Production
runs normally omit this override and discover their actual device/allocation.

`tools/test_tile_parallelism.py` compares serial and automatic tile workers in
separate processes on a large region, checks label pyramids, and records label
differences and resource use for all five model families (including both
Cellpose transformer checkpoints). Run it in the usual GPU environment:

```bash
python tools/test_tile_parallelism.py --input /data/large.ome.zarr \
  --output /data/tile-worker-test --size 5120 --gpu-memory-mb 12288
```

For a direct local run:

```powershell
conda run -n cisegmentation python wrapper.py --infolder inputfolder --outfolder outputfolder `
  --cell-model cellpose3:cyto3 `
  --cell-channel 1 --cell-nuclei-channel 0 --device cuda
```

Benchmark example:

```powershell
conda run -n cisegmentation python wrapper.py --infolder inputfolder --outfolder outputfolder `
  --cell-model skip --nucleus-model cellpose3:nuclei --nucleus-channel 1 `
  --benchmark true --device cuda
```

Benchmark mode selects the first input/field and first timepoint, center-crops
XY to at most 1024×1024, and writes one rendered 2D XY RGB OME-Zarr montage.
Like the QuPath extension gallery, it places input projections above colored
segmentation results and includes model names, object counts, runtimes, skips,
and failures in the image. It benchmarks every model offered by each enabled
workflow step, using that step's configured input channel. Separate enabled
Step 3a–3d slots are benchmarked independently, including when they use the
same channel.

## Parameters

All channel numbers shown to users are one-based. Physical parameters are
converted internally using the OME-Zarr XY scale metadata.

| Parameter | Use |
| --- | --- |
| Step 1: Cell Detection (`--cell-model`) / Cyto Channel / Nucleus Channel | Selects `Skip`, a direct cell model, or `Cell expansion using …`. Direct models receive the Cyto Channel and, when the optional nucleus channel is greater than zero and different, that channel as a second input. For expansion, the seed channel may be entered in either Step 1 channel field. A channel above 1 is preferred over a default-like 0 or 1, and the Nucleus Channel wins when both are above 1. This input does not independently create matched nucleus labels; select Step 2 for that. |
| Step 1 Expansion Distance | Sets the maximum XY expansion distance in µm. Physical X/Y scales are read from OME-Zarr metadata. Expansion produces matched cell, nucleus, and cytoplasm channels directly. |
| Step 2: Nuclei Detection (`--nucleus-model`) / Channel | Selects `Skip` or an independent nucleus model. When cells and nuclei are both available, they are matched by overlap; only the largest nucleus per cell is retained, cells without nuclei are removed, and cytoplasm is written with shared IDs. Step 2 may repeat the nucleus model used for Step 1 expansion. |
| Step 3a–3d: Foci Detection (`--foci-model-1` … `--foci-model-4`) / Channel | Step 3a is a beginner selector; Steps 3b–3d appear first in the advanced options. Each offers `Skip`, Spotiflow, `SD_Foci_*` StarDist, and Cellpose 3 `bact` models. Repeating models or channels is allowed. StarDist outputs are named `foci`; Cellpose bacterial outputs are named `bacteria`. |
| Original data (legacy CLI only: `--include-original-data`) | Removed from the UI. Normal output copies source pixels and final labels, retaining the input. The CLI still accepts false for a labels-only overlay; `--include-original-channels` retains its legacy mapping. |
| Existing Labels (`--existing-labels`) | `overwrite` (default) replaces generated-name collisions while preserving unrelated labels; `remove` replaces the complete labels tree; `append` preserves all groups and assigns collision-safe suffixes consistently across the plate. |
| Maximum Inference / Measurement Workers | Advanced caps for automatically sized inference and measurement pools. Zero means automatic based on GPU memory, CPU allocation/affinity, and RAM. GPU sizing uses the greater of PyTorch peak allocation and NVIDIA's complete worker-process memory, reserves at least 2 GiB or 20% of VRAM, and applies a 50% per-worker safety margin. |
| Labels Log Info (`--labels-log-info`) | Advanced option, disabled by default. Calculate and log per-step and final label counts, foreground fraction, and label size statistics. Leave disabled for faster processing of large fields. |
| Smooth Rescaled StarDist Labels (`--smooth-stardist-labels`) | Advanced option, enabled by default. When StarDist inference downsamples a high-magnification image, its polygons are scaled and rasterized directly on the source grid for smooth boundaries. Disable it to reproduce nearest-neighbor label-map restoration. |

## Maintainer publication tools

The repository includes two Windows maintainer scripts. They contain no
credentials: Docker authentication is read from Docker's credential store and
GitHub authentication is handled by `gh auth`.

Inspect Docker tags and push commands without changing the registry:

```bat
pushdocker.cmd --skip-build --dry-run
```

Build and publish the version from `version.txt` only after reviewing the dry
run. Publication includes only the headless workflow image (version and
`latest` tags); optional Gradio and Jupyter images are no longer published:

```bat
pushdocker.cmd --yes
```

Validate a GitHub release without creating a tag or release:

```bat
release_github.cmd --dry-run
```

Create the release after reviewing the dry run:

```bat
release_github.cmd --yes
```

Publication requires a semantic version, optionally prefixed with `v` (for
example, `v0.3.0`). Docker publishing is locked to
`cellularimagingcf/w_cisegmentation`; GitHub publishing is locked to the
`Cellular-Imaging-Amsterdam-UMC/cisegmentation` origin. The GitHub release tool
also requires a clean branch that is synchronized with its upstream. If the
Docker tag in `config.yaml` differs from `version.txt`, the confirmed release
updates it, commits and pushes that synchronization, and then tags the resulting
commit. A dry run only prints those actions.
| Remove Border Cells (`--remove-border-cells`) | Advanced option, enabled by default. Removes cells touching an XY image edge and propagates removal to matched nuclei and derived cytoplasm. Z-stack endpoints are not treated as image borders. |
| Compute Device (`--device`) | `auto` selects CUDA when available; `cuda` requires a GPU; `cpu` forces CPU inference. |
| Dimension Mode (`--dimension-mode`) | `auto` uses native 3D where supported; `slice-2d` independently segments and relabels every Z plane. |
| Cellpose Diameter (`--diameter`) | Object diameter in µm, converted using mean XY pixel size. `0` resolves to 12 µm for nuclei or 25 µm for cells; a negative value uses the model default. |
| Cellpose Probability Threshold (`--cellprob-threshold`) | Cellpose cell-probability acceptance threshold. Higher values generally produce fewer masks. |
| Cellpose Flow Threshold (`--flow-threshold`) | Cellpose flow-consistency error threshold. |
| StarDist Probability Threshold (`--stardist-prob-threshold`) | Minimum object probability. `-1` loads `prob` from the selected model's `thresholds.json`. |
| StarDist NMS Threshold (`--stardist-nms-threshold`) | Allowed overlap during non-maximum suppression. `-1` loads `nms` from `thresholds.json`. |
| Spotiflow Probability Threshold (`--spotiflow-prob-threshold`) | Spot acceptance threshold. `-1` uses the checkpoint default. |
| Spotiflow Minimum Distance (`--spotiflow-min-distance`) | Minimum separation in µm, converted to pixels from the mean XY pixel size. |
| Spotiflow Local Mask Refinement (`--spotiflow-local-refinement`) | Advanced checkbox, disabled by default. Lightly smooths the selected channel, estimates background and noise around every Spotiflow point, and grows only the seed-connected signal. Growth is bounded to a 1.0 µm radius, overlaps are assigned by local signal-to-noise score, and weak points remain single pixels. Native-3D Spotiflow requires Force slice-wise 2D. The former `--spotiflow-microsam-refinement` flag remains a hidden compatibility alias. |
| Create Measurements Database (`--measurements-database`) | Beginner selector: `duckdb` (default), `sqlite`, or `skip`. Writes one database per top-level image or HCS screen containing final-object shape/location features, per-original-channel intensity statistics, and pairwise mask relationships. See the [measurements database reference](docs/measurements.md). |
| Benchmark Gallery (`--benchmark`) | Advanced option. Processes the first deterministic image/field and first timepoint, runs all selectable models for every enabled step, then writes only a 2D XY OME-Zarr gallery. |

Optional spatial measurements (`--spatial-measurements`), channel colocalization
(`--colocalization`), time-series tracking (`--tracking`) and a separate geometry
database (`--export-geometry`) are disabled by default. They require DuckDB or
SQLite measurements and add extension schema 1 while preserving base schema 5
and raster label values. Cells/nuclei may divide; independent spots never split
or merge. See [measurement extensions](docs/measurement-extensions.md) for units,
controls, scientific limits and repeatable validation. Spot-only runs can skip
both cell and nucleus detection. Retained public inputs and their pyramidal
OME-Zarr conversion are documented in [public datasets](docs/publicdata.md).

`SD_Nuclei_Versatile` is automatically downsampled to 0.5 µm/px per XY axis
when the source resolution is finer. By default, its predicted polygons are
rasterized directly on the source grid for smooth boundaries; the advanced
toggle can restore the previous nearest-neighbor behavior. The model is based
on DSB2018 nuclei data described by the
[official StarDist project](https://github.com/stardist/stardist),
and 0.5 µm/px matches the detection resolution in the
[official QuPath StarDist example](https://qupath.readthedocs.io/en/latest/docs/deep/stardist.html).
Other StarDist checkpoints retain their native input scale.
InstanSeg always reads pixel size directly from OME-Zarr metadata.

Normal label outputs store one channel per result type and use a rendering
window from zero through that channel's maximum label gray value. The metadata
requests OMERO's `glasbey_inverted.lut`; its value zero is black. Because NGFF
0.4 does not standardize LUT selection, non-OMERO readers also receive
non-black semantic fallback colors. The launcher OMERO roundtrip explicitly
applies and saves the Glasbey LUT after BIOMERO imports each result.

Normal inference always uses this optional-step workflow; there is no separate
single-model mode. If Cell Detection, Nuclei Detection, and all four Foci
Detection slots are disabled, the workflow stops with a clear validation
error. Original data channels do not receive the Glasbey label LUT and retain
their source names, colors, and display windows when those are available.

## Performance and timing provenance

Loaded models are cached for the lifetime of one `wrapper.py` process using
the stable model ID and resolved device as the key. Repeated timepoints, plate
fields, and input images therefore reuse the same Cellpose, StarDist,
InstanSeg, or Spotiflow model without deserializing it or transferring it to
the device again. Separate workflow invocations remain isolated.

Each regular output records `model_cache_hits`, `model_cache_misses`, and a
`timings` object in its root `cisegmentation` metadata. Within one image
timepoint, repeated segmentation requests with identical inference inputs reuse
an independent copy of the first result. These are logged separately and
recorded as `result_cache_hits`; they are not counted as model-cache hits.
Plate outputs aggregate these values at the plate root while retaining
per-field provenance. Timing fields are `startup_seconds`,
`zarr_read_seconds`, `import_seconds`,
`device_setup_seconds`, `model_load_seconds`, `inference_seconds`,
`zarr_write_seconds`, `measurement_seconds`, and `total_seconds`. Regular
outputs also record `runtime_seconds` as the total model segmentation runtime
and `segmentation_count`, which are used to report average time per
segmentation. Benchmark run records contain their per-model import, load,
inference, and cache information as well.
Refined Spotiflow runs additionally record `spot_detection_seconds`,
`local_refinement_seconds`, the physical/pixel radius, threshold policy, and
the numbers of detected points, grown masks, single-pixel fallbacks, duplicate
seeds, and removed overlap pixels.

## Models

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
are isolated in [bilayers_extra](bilayers_extra/README.md). They reuse an already
built workflow image and are excluded from default builds, CI builds, and Docker
Hub publication.

## OMERO round trip

**Run OMERO Roundtrip** processes the first top-level OME-Zarr through local
OMERO, BIOMERO, and Slurm, then re-exports the OMERO-imported result into the
selected output folder. `builddocker.cmd` records the source fingerprint and
immutable image ID in a Git-ignored local state file. The roundtrip skips its
Docker build and SIF conversion when those identities still match.
Roundtrip registration is isolated as `rt_cisegmentation`, with Slurm images
under `workflows/RT_cisegmentation`. It never writes into the standard
`workflows/cisegmentation` cache. Roundtrip cache directories and artifacts are
created for the Slurm account (`990:990`) with group-writable, setgid directory
permissions so later BIOMERO downloads cannot be blocked by root-owned files.

The progress dialog reports BIOMERO and Slurm identifiers when available and
prints a heartbeat while analysis or result import is still running. Cancel
terminates the local process tree and cancels the active cisegmentation Slurm
job; temporary OMERO objects and Slurm evidence are retained. A fully
successful roundtrip exports results, collects correlated logs under
`outputfolder/logs/<roundtrip-id>/`, and deletes its temporary OMERO objects.
`biomero_workflow.log` is compacted for normal inspection, while
`biomero_workflow.raw.log` retains the complete unfiltered BIOMERO output.
