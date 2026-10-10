# Slurm allocation checks and MIG acceptance test

Version `v0.6.4-beta` writes JSON lines prefixed with
`[CISEGMENTATION_RESOURCE_CHECK]` to the workflow stdout captured by Slurm/BIOMERO.
They are always enabled, independently of label logging. Search the job log for
that prefix to compare the requested allocation with actual scheduling decisions.

The `startup` event precedes input discovery and segmentation. It includes:

- Workflow version, requested device and worker caps.
- Slurm job, step, partition, node list, CPU, RAM and GPU environment values.
- Host logical CPUs, affinity, cgroup v1/v2 CPU quota and memory limit/usage,
  and effective CPU/RAM budgets after intersecting the available limits.
- CUDA-visible device index, name, UUID, CUDA runtime, visible memory and MIG
  detection when the driver exposes a MIG name/UUID. `not-reported` means the
  available properties did not identify MIG; it does not prove MIG is disabled.
- Explicit warnings for unknown GPU memory or missing Slurm RAM environment
  budgets. Only allocation-related environment values are logged.

The job/step allocation within the selected partition controls concurrency; the
partition's entire node, CPU count or GPU capacity is not a worker budget.
`SLURM_MEM_PER_GPU` is **system RAM per locally allocated GPU**, not GPU VRAM.
An unknown local GPU count with this variable set fails before segmentation
instead of assuming unrestricted host RAM. Zero Slurm memory values do not
impose a positive limit; cgroups and host headroom still apply.

`inference_sizing`, `tile_sizing`, `finalization_sizing`, `measurement_sizing` and `extension_sizing`
events report selected workers and effective budgets. Inference sizing also
reports probe RSS/VRAM, the requested cap and GPU safety/reserve values. Each
spawned numerical worker reports `worker_threads` with its PID and measured BLAS
thread counts. NumPy/SciPy backends already imported by spawn are limited to one
thread, along with environment limits for later numerical imports.
Parent numerical backends are limited to the effective CPU allocation, including
serial and benchmark execution; their measured counts appear in `startup`.

`database_budget` events report DuckDB threads and its actual configured memory
limit in MiB. Numerical workers use one DuckDB thread and a buffer budget of at
most 256 MiB, reduced further by live RAM headroom. Parent database operations
use the effective CPU allocation. The 512 MiB worker-RSS sizing estimate includes
more than the database buffer; it is an estimate rather than a hard limit on all
Python/model/array allocations. Slurm/container cgroups remain the process memory
boundary.

CUDA memory discovery never falls back to the first physical GPU in
`nvidia-smi`. A CUDA-visible MIG slice is the authoritative device; an unavailable
query restricts automatic GPU concurrency to one. The current implementation
uses the current CUDA device; it does not distribute fields across multiple GPUs.
Worker VRAM profiling includes a **512 MiB context/library allowance** in addition
to the Torch allocator peak, or the NVIDIA process-memory reading when larger.
This allowance is a conservative estimate, not a complete measurement of all
non-Torch allocations. Worker selection also keeps at least 2 GiB/20% free and
uses a 1.5 multiplier on the probe peak.

Known CUDA allocation errors, including `CUBLAS_STATUS_ALLOC_FAILED`, stop the
affected pool. Owned workers and nested tile workers are terminated and joined
before retrying; other jobs/processes are untouched. `pool_released`,
`gpu_recovery`, `inference_retry`, `probe_retry` and `tile_retry` events record
recovery and fresh budgets. Retries reduce concurrency, keep confirmed successful
fields/committed tiles, and remove only failed field staging results. Unknown or
permanently insufficient GPU headroom cannot enable a larger retry pool. A GPU
recovery wait is bounded to 30 seconds; persistent pressure raises a clear error.

## Acceptance cluster validation

Pull the beta explicitly; stable `latest` remains unchanged:

```sh
docker pull cellularimagingcf/w_cisegmentation:v0.6.4-beta
```

Configure the acceptance workflow/SIF to use this tag, then run inside the
normally allocated Slurm job/container so its GPU visibility, CPU affinity and
cgroup limits are preserved. Do not replace a Slurm MIG visibility mask with the
physical parent GPU's UUID. The existing optional
`CISEGMENTATION_GPU_MEMORY_LIMIT_MB` can impose a smaller allocator budget for a
separate controlled recovery test; it never increases the visible allocation.

For issue #3, collect the log and verify:

1. The startup version is `0.6.4-beta`, the intended partition/job/step is logged,
   and effective CPUs/RAM are no larger than the job allocation.
2. On the original approximately 12 GiB MIG allocation, GPU memory reflects that
   slice rather than the 80 GiB parent; identify the visible device/UUID.
3. Automatic workers respect the logged CPU, RAM, VRAM and explicit caps, and
   worker BLAS backends each report one thread.
4. A representative plate completes and its labels/databases match the expected
   scientific output. Also compare workers=1 with automatic workers on a small
   deterministic sample.
5. In a separate controlled allocation-failure run, logs show released pools,
   shrinking concurrency, fresh headroom checks and preservation of completed
   fields. Verify no orphan GPU processes remain for the completed job.

Local CPU tests simulate MIG discovery and allocation failures. They do not
replace validation on a real Slurm MIG allocation. Keep
[issue #3](https://github.com/Cellular-Imaging-Amsterdam-UMC/cisegmentation/issues/3)
open until this acceptance test succeeds.

Allocation semantics: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html) and
[Slurm GPU/MIG GRES documentation](https://slurm.schedmd.com/gres.html).
