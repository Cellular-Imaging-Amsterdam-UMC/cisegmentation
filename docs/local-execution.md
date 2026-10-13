# Local execution and OMERO round trip

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
**Run Docker** uses the locally built, release-pinned `w_cisegmentation:v0.6.4`
image; the organization-qualified image in `config.yaml` is reserved for BIOMERO registry
metadata.

During execution, the job log reports the selected workflow and tuning values,
input T/C/Z/Y/X dimensions, datatype, physical scales, and channel names. Every
model call reports its resolved CPU/GPU device, dimensional mode, runtime,
timing breakdown, and model-cache status. Enable **Labels Log Info** to also
calculate label counts, foreground fractions, and size distributions for model
and final post-processed outputs. It is disabled by default because those full
label-array scans can be costly.
## Direct and benchmark examples

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
