# Run all real model checkpoints on the local Slurm GPU node

From a PowerShell terminal in this repository:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_all_models.ps1
```

This submits one Slurm GPU job and watches it. It uses the existing workflow
image with an immutable copy of the current source mounted at `/app`; it does
not build an image or change the installed BIOMERO workflow. No model/image
download is initiated by the runner. Missing cached checkpoints are reported.

The default SIF filename follows `version.txt`. Use `-WorkflowImage` with an
explicit cluster path to test another installed image. Resume retains the
original image path, source snapshot and configuration.

The default plan covers **all 39 checkpoints in 115 cases**:

- 73 small cases: all supported targets, tiled versus untiled comparison,
  and native 3D tests for 3D models, in addition to forced slice mode.
- 42 large cases: one streamed 8192 x 8192 crop per model/target, plus label
  pyramid generation/validation. Native 3D Spotiflow uses eight synthetic Z
  slices; other large cases are 2D. Cellpose/SAM native 3D small cases use 16
  synthetic slices so the rescaling step has sufficient depth.

Each case gets a new process. Failed cases and timeouts are recorded and do
not stop the rest. CUDA is required and CPU fallback is rejected. Reports
include GPU allocations, live resource headroom, CPU/RAM limits, memory
retries, effective overlap, object counts and comparison scores.

`oom_retries` counts memory allocation failures. `size_limit_retries` counts
known operation size limits, such as InstanSeg percentile normalization on
large rescaled planes. Both cause smaller tiles while retaining completed cores.
The latter is a separate failure type and does not establish a GPU OOM.

The local c1 test profile currently has four CPUs and a hard 16 GiB Docker
memory cap. The job requests four CPUs, 16 GiB RAM and one GPU. The physical
GPU is 24 GB; a per-process PyTorch allocator cap simulates 12 GiB. It excludes
some CUDA/driver allocations. The Slurm GPU node is reserved during the run.

## Other run sizes

```powershell
# Inspect the plan and cached weights without submitting a job:
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_all_models.ps1 -PlanOnly

# All small real-model cases, skipping the large stage:
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_all_models.ps1 -QuickOnly

# Run the full 40K image for every model/target in the large stage:
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_all_models.ps1 -Full40K

# One checkpoint, useful for checking the setup:
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_all_models.ps1 -Models cellpose3:nuclei -QuickOnly
```

Default case timeout is 30 minutes and job wall limit is 48 hours. Full40K
defaults to four hours per case and 72 hours for the job; it may require
multiple days and resume. Override with `-ModelTimeoutMinutes` and
`-WallHours` if needed. `-LargeSize` changes the square crop (default 8192).
All large tests read the separate Docker copy of the 40K source; they never
modify the original file or OMERO image. The large stage tests inference,
stitching and label pyramids, not a complete measurement export for every model.

## Results and interruption

Each launch prints a unique run folder under:

```text
V:\BIOMERO-local\tests\results\all-models\<run-id>
```

`results\summary.csv` is the main overview. `summary.json`, per-case
`report.json` and `case.log`, small `comparison.npz` arrays, the coordinator
log, configuration and source snapshot/hashes are also saved. A compressed
evidence archive is collected at the end. Large raw labels/pyramid overlays
remain in the Docker shared volume; their exact paths appear in reports.
They are diagnostic labels-only overlays, requiring source pixels for
standalone viewing. Keeping them in Docker avoids copying many gigabytes.

Ctrl+C stops the Windows watcher, **not** the Slurm job. The launcher prints
the job ID and a cancellation command. To reconnect/collect, or continue
unfinished cases after a job timeout:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_all_models.ps1 -ResumeRun 'V:\BIOMERO-local\tests\results\all-models\<run-id>'
```

Resume preserves the original code/configuration, skips completed cases and
keeps prior attempts. Add `-RetryFailures` to rerun failed/timed-out cases in
new attempt folders; passed cases stay skipped. If code changed, start a new
run rather than resuming the old snapshot. `-SubmitOnly` submits without
watching; reconnect using ResumeRun when you want the collected evidence.

## Interpret the results

`execution_status` records whether a real checkpoint loaded, ran on CUDA and
produced structurally valid output. `quality_status` is separate:

- `agreement_pass`: positive small reference and useful tiled agreement.
- `review_needed`: detections exist, but tiled/untiled agreement is weaker.
- `inconclusive_no_reference_detections`: the reference is empty. This is
  not positive detection validation, even if execution succeeded.
- `large_execution_only`: large inference/pyramids completed; no giant
  whole-image reference was allocated.
- `lost_reference_detections`: failed comparison against a positive untiled
  reference. Counts and comparison arrays are retained for inspection.

An empty tiled result against a positive reference fails the case.
Successful execution alone does not validate tiled agreement; inspect every
`review_needed` case and treat empty references as inconclusive. Cached weight
availability also does not imply that all optional network libraries are
installed: the two Cellpose3 transformer checkpoints require
`segmentation_models_pytorch` and its dependencies in the workflow runtime.
Comparisons use foreground Dice for masks and nearest points within two
pixels for Spotiflow. No annotated biological ground truth is supplied.
The fluorescence fixture does not validate accuracy on yeast, bacteria,
phase contrast or brightfield images. The large source has one channel;
models requiring RGB receive repeated channels, explicitly flagged in the
report. Native 3D uses a Gaussian-weighted repeated-plane fixture, also
flagged. Specialized models will need suitable real images for a biological
accuracy assessment.
