# Large inputs, concurrency and timing

## Plate processing

HCS plates are processed model by model across all fields. Worker counts are
sized from a model memory probe and the available GPU or CPU allocation. Native
labels are finalized in a CPU pool after inference, then measurements run in
spawned CPU workers and stream bounded field databases to one parent writer.
The run-specific temporary working directory is a hidden sibling in the output
folder, never a child of the input OME-Zarr. For full-data output, a verified
source copy runs alongside label finalization and generated labels are committed
only to that copy. The final store and database become visible only after every
phase succeeds.

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

Timing counters can sum work across workers and overlap phases. Use the logged
`CI segmentation completed in ... seconds` for process elapsed time, and use the
workflow tracker for end-to-end elapsed time. `import_seconds` measures Python
library imports, not result import into an external image server. Do not infer
transfer/import stage durations from folder names or file modification times.
See [resource checks](slurm-resource-checks.md) for structured allocation evidence.
