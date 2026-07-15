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

For a direct local run after activating the environment:

```powershell
python wrapper.py --infolder inputfolder --outfolder outputfolder `
  --model cellpose3:nuclei --target nuclei --device cuda
```

Benchmark example:

```powershell
python wrapper.py --infolder inputfolder --outfolder outputfolder `
  --benchmark true --benchmark-models all --target nuclei --device cuda
```

Benchmark mode selects the first input/field and first timepoint, center-crops
XY to at most 1024×1024, and writes one rendered 2D XY RGB OME-Zarr montage.
Like the QuPath extension gallery, it places input projections above colored
segmentation results and includes model names, object counts, runtimes, skips,
and failures in the image. Presets cover every model, Cellpose-SAM only,
legacy Cellpose 3 only, StarDist, InstanSeg, or Spotiflow. Each preset runs
every model in that family using a target supported by that model.

## Parameters

All channel numbers shown to users are one-based. Physical parameters are
converted internally using the OME-Zarr XY scale metadata.

| Parameter | Use |
| --- | --- |
| Model (`--model`) | Stable model/checkpoint identifier to run. |
| Segmentation Target (`--target`) | Required biological output: nuclei, cells, foci, or spots. The selected model must support it. |
| Primary Channel (`--primary-channel`) | Main signal channel. |
| Nuclei Channel (`--nuclei-channel`) | Optional nuclei channel for cell models; `0` means none. The InstanSeg brightfield model automatically consumes the first three channels. |
| Multi-step Segmentation (`--multi-step`) | Enables the optional cell, nucleus, and repeated spot-channel pipeline. The original Model/Target/Primary Channel fields are used only when this is off. |
| Segment Cells / Cell Model / Cell Signal Channel | Runs a cell model using its one-based primary signal channel and optional Cell-step Nuclei Channel (`0` means none). |
| Segment Nuclei / Nucleus Model / Nucleus Signal Channel | Independently segments nuclei. When the cell step is also enabled, nuclei are matched to cells by overlap; the largest nucleus is retained and cells without nuclei are removed. |
| Segment Spots/Foci / Spot/Foci Model / Spot Channels | Runs a Spotiflow model, either `SD_Foci_*` StarDist model, or a Cellpose 3 checkpoint containing `bact` once per comma-separated one-based channel. Entries are not deduplicated, so `2,2` creates two label channels. StarDist channels are named `foci`, Cellpose bacterial channels are named `bacteria`, and a Cellpose diameter of `0` uses the bacterial model default. |
| Derive Cytoplasm (`--derive-cytoplasm`) | Adds a cytoplasm channel containing matched cell masks minus their matched nucleus masks. Cell, nucleus, and cytoplasm gray-value IDs correspond. |
| Remove Border Cells (`--remove-border-cells`) | Removes cells touching an XY image edge and propagates removal to matched nuclei and derived cytoplasm. Z-stack endpoints are not treated as image borders. |
| Compute Device (`--device`) | `auto` selects CUDA when available; `cuda` requires a GPU; `cpu` forces CPU inference. |
| Dimension Mode (`--dimension-mode`) | `auto` uses native 3D where supported; `slice-2d` independently segments and relabels every Z plane. |
| Cellpose Diameter (`--diameter`) | Object diameter in µm, converted using mean XY pixel size. `0` resolves to 12 µm for nuclei or 25 µm for cells; a negative value uses the model default. |
| Cellpose Probability Threshold (`--cellprob-threshold`) | Cellpose cell-probability acceptance threshold. Higher values generally produce fewer masks. |
| Cellpose Flow Threshold (`--flow-threshold`) | Cellpose flow-consistency error threshold. |
| StarDist Probability Threshold (`--stardist-prob-threshold`) | Minimum object probability. `-1` loads `prob` from the selected model's `thresholds.json`. |
| StarDist NMS Threshold (`--stardist-nms-threshold`) | Allowed overlap during non-maximum suppression. `-1` loads `nms` from `thresholds.json`. |
| Spotiflow Probability Threshold (`--spotiflow-prob-threshold`) | Spot acceptance threshold. `-1` uses the checkpoint default. |
| Spotiflow Minimum Distance (`--spotiflow-min-distance`) | Minimum separation in µm, converted to pixels from the mean XY pixel size. |
| Benchmark Gallery (`--benchmark`) | Processes the first deterministic image/field and first timepoint, then writes only a 2D XY OME-Zarr gallery. |
| Benchmark Models (`--benchmark-models`) | Chooses all models or every model in the selected algorithm family. |

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
