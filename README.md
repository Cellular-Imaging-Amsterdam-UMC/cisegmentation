# CI Segmentation

CI Segmentation is a GPU-enabled Bilayers/BIOMERO workflow for instance
segmentation from OME-Zarr to labeled OME-Zarr. It supports Cellpose 3,
Cellpose-SAM, PyTorch StarDist, InstanSeg, and Spotiflow from one CUDA 12.6
environment.

## Workflow contract

- Input: one or more top-level `.ome.zarr` stores in `/data/in`, including HCS plates.
- Normal output: standalone `uint32` label OME-Zarrs in `/data/out`.
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

For a direct local run after activating the environment:

```powershell
python wrapper.py --infolder tests/data --outfolder outputs `
  --model cellpose3:nuclei --target nuclei --device cuda
```

Benchmark example:

```powershell
python wrapper.py --infolder tests/data --outfolder outputs `
  --benchmark true --benchmark-models all --target nuclei --device cuda
```

Benchmark mode selects the first input/field and first timepoint, center-crops
XY to at most 1024×1024, retains full Z, and writes one multichannel label
OME-Zarr. Each successful model is stored as a separately named label channel.

## Models

`tools/download_models.py` prepares an idempotent cache containing all
registered Cellpose 3 models, Cellpose-SAM, `SD_Nuclei_Versatile`,
`SD_Foci_Aggregates`, `SD_Foci_Finn`, all three InstanSeg models, and all six
Spotiflow models. The custom StarDist H5 weights are fetched from pinned commit
`b280dfeb` of `cistardist_pytorch` and converted to PyTorch checkpoints.

Docker builds execute this downloader once and bundle the validated cache, so
runtime jobs do not need network access.

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

The scripts under `tools/omero_import_metadata_probe` reuse the same local
OMERO/BIOMERO probe contract as `cideconvolve`: export an image or plate to the
Slurm-input OME-Zarr representation, run `wrapper.py`, import the labeled
output through direct and BIOMERO paths, compare metadata, and optionally clean
up imports. See the tool README for prerequisites and commands.
