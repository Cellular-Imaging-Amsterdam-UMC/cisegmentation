# CI Segmentation

CI Segmentation is a GPU-enabled Bilayers/BIOMERO workflow for instance
segmentation from OME-Zarr to labeled OME-Zarr. It supports Cellpose 3,
Cellpose-SAM, PyTorch StarDist, InstanSeg, and Spotiflow from one CUDA 12.6
environment.

## Workflow contract

- Input: one or more top-level `.ome.zarr` stores in `/data/in`, including HCS plates.
- Normal output: standalone nonnegative `int32` label OME-Zarrs in `/data/out`.
  Signed 32-bit storage preserves instance IDs while remaining readable by
  QuPath 0.7.0, whose Bio-Formats image server rejects `uint32` tiles.
- Benchmark output: **only** `benchmark_gallery_<image>.ome.zarr`.
- Axes are normalized to `TCZYX`; time and Z are preserved in normal runs.
- Spotiflow points become uniquely numbered single pixels or voxels.

The internal `wrapper.py` is the container/Bilayers entrypoint. It is not
packaged as an end-user CLI. `launcher.py` is the supported local frontend and
constructs the Docker invocation from `config.yaml`.

## Local environment

Run `create_env.cmd` to create or update the `cisegmentation` Conda environment with
Python 3.11, PyTorch 2.11.0, torchvision 0.26.0, and CUDA 12.6 wheels. The
script also installs the PyQt launcher dependencies and finishes with a CUDA
smoke test.

The launcher defaults to `inputfolder` and `outputfolder` in the repository
root. Tests clean `tests/inputfolder` and `tests/outputfolder`, then copy fresh
OME-Zarr fixtures from `tests/data` into the test input folder when required.
The launcher provides separate **Run Docker** and **Run Locally** buttons; local
mode uses the active Python environment and executes `wrapper.py` directly.
**Run Docker** uses the locally built `w_cisegmentation:latest` image; the
organization-qualified image in `config.yaml` is reserved for BIOMERO registry
metadata.

During execution, the job log reports the selected workflow and tuning values,
input T/C/Z/Y/X dimensions, datatype, physical scales, and channel names. Every
model call reports its resolved CPU/GPU device, dimensional mode, runtime and
timing breakdown, model-cache status, label count, foreground fraction, and
label-size distribution. Post-processing statistics show the final output of
cell/nucleus matching and border removal. Structured per-step and final
statistics are also stored in the output provenance.

For a direct local run after activating the environment:

```powershell
python wrapper.py --infolder inputfolder --outfolder outputfolder `
  --cell-model cellpose3:cyto3 `
  --cell-channel 1 --cell-nuclei-channel 0 --device cuda
```

Benchmark example:

```powershell
python wrapper.py --infolder inputfolder --outfolder outputfolder `
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
| Step 1: Cell Detection (`--cell-model`) / Cyto Channel / Nucleus Channel | Selects `Skip`, a direct cell model, or `Cell expansion using …`. Direct models receive the Cyto Channel and, when the optional nucleus channel is greater than zero and different, that channel as a second input. Expansion uses the Step 1 Nucleus Channel as its seed signal when greater than zero, otherwise it falls back to the Cyto Channel. This input does not independently create matched nucleus labels; select Step 2 for that. |
| Step 1 Expansion Distance | Sets the maximum XY expansion distance in µm. Physical X/Y scales are read from OME-Zarr metadata. Expansion produces matched cell, nucleus, and cytoplasm channels directly. |
| Step 2: Nuclei Detection (`--nucleus-model`) / Channel | Selects `Skip` or an independent nucleus model. When cells and nuclei are both available, they are matched by overlap; only the largest nucleus per cell is retained, cells without nuclei are removed, and cytoplasm is written with shared IDs. Step 2 may repeat the nucleus model used for Step 1 expansion. |
| Step 3a–3d: Foci Detection (`--foci-model-1` … `--foci-model-4`) / Channel | Each beginner selector offers `Skip`, Spotiflow, `SD_Foci_*` StarDist, and Cellpose 3 `bact` models. Repeating models or channels is allowed. StarDist outputs are named `foci`; Cellpose bacterial outputs are named `bacteria`. |
| Include Original Data Channels (`--include-original-channels`) | Final beginner option. Prepends all source channels before the label channels. Generated label names use the collision-safe `labels_<name>` form with underscores only. The combined image remains `int32`: compatible integer intensities are preserved, while finite in-range floating-point intensities are rounded to the nearest integer. The original datatype and conversion are recorded in provenance. |
| Write Native OME-Zarr Labels (`--write-ome-zarr-labels`) | Advanced option. Keeps the original image and datatype at the output root and writes every segmentation separately under `labels/` using the OME-Zarr 0.4 `labels` and `image-label` metadata. Duplicate output names receive a numeric suffix. This replaces the extra-label-channel layout; the Include Original Data Channels setting is therefore unnecessary in this mode. |
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
run:

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

Publication requires a strict semantic version. Docker publishing is locked to
`cellularimagingcf/w_cisegmentation`; GitHub publishing is locked to the
`Cellular-Imaging-Amsterdam-UMC/cisegmentation` origin. The GitHub release tool
also requires a clean branch that is synchronized with its upstream.
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
| Benchmark Gallery (`--benchmark`) | Advanced option. Processes the first deterministic image/field and first timepoint, runs all selectable models for every enabled step, then writes only a 2D XY OME-Zarr gallery. |

`SD_Nuclei_Versatile` is automatically downsampled to 0.5 µm/px per XY axis
when the source resolution is finer, and its labels are restored to the source
grid with nearest-neighbor interpolation. The model is based on DSB2018 nuclei
data described by the [official StarDist project](https://github.com/stardist/stardist),
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
`timings` object in its root `cisegmentation` metadata. Plate outputs aggregate
these values at the plate root while retaining per-field provenance. Timing
fields are `startup_seconds`, `zarr_read_seconds`, `import_seconds`,
`device_setup_seconds`, `model_load_seconds`, `inference_seconds`,
`zarr_write_seconds`, and `total_seconds`. Benchmark run records contain their
per-model import, load, inference, and cache information as well.

## Models

`tools/download_models.py` prepares an idempotent cache in the repository's
Git-ignored `models/` folder containing all
registered Cellpose 3 models, Cellpose-SAM, `SD_Nuclei_Versatile`,
`SD_Foci_Aggregates`, `SD_Foci_Finn`, all three InstanSeg models, and all six
Spotiflow models. The three StarDist source folders are bundled from the pinned
`cistardist_pytorch` models and converted to PyTorch checkpoints only when the
corresponding `.pt` file is missing.

After successful CPU smoke-loading, the downloader removes Spotiflow download
ZIPs and training-only `last.pt` checkpoints; runtime loads the extracted
`best.pt` folders directly and therefore remains offline. StarDist Keras H5
conversion sources are likewise removed after the converted `.pt` checkpoint
has loaded successfully. The completion inventory is written only after this
cleanup, so interrupted or invalid caches are repaired on the next run.

The headless image removes Triton after installing the pinned PyTorch stack.
Triton is used by `torch.compile`/Inductor, while this inference-only workflow
uses eager PyTorch execution. GPU smoke tests cover Cellpose 3, Cellpose-SAM,
StarDist, InstanSeg, and Spotiflow without it. The NVIDIA CUDA runtime packages
remain installed because PyTorch 2.11 links against them directly, including
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
builddocker.cmd             headless Bilayers/BIOMERO image
builddocker_gradio.cmd      headless + generated Gradio image
builddocker_jupyter.cmd     headless + generated JupyterLab image
```

The images are tagged `w_cisegmentation:<version>`,
`w_cisegmentation:<version>-gradio`, and
`w_cisegmentation:<version>-jupyter`.

## OMERO round trip

**Run OMERO Roundtrip** processes the first top-level OME-Zarr through local
OMERO, BIOMERO, and Slurm, then re-exports the OMERO-imported result into the
selected output folder. `builddocker.cmd` records the source fingerprint and
immutable image ID in a Git-ignored local state file. The roundtrip skips its
Docker build and SIF conversion when those identities still match.

The progress dialog reports BIOMERO and Slurm identifiers when available and
prints a heartbeat while analysis or result import is still running. Cancel
terminates the local process tree and cancels the active cisegmentation Slurm
job; temporary OMERO objects and Slurm evidence are retained. A fully
successful roundtrip exports results, collects correlated logs under
`outputfolder/logs/<roundtrip-id>/`, and deletes its temporary OMERO objects.
`biomero_workflow.log` is compacted for normal inspection, while
`biomero_workflow.raw.log` retains the complete unfiltered BIOMERO output.
