# Resource evidence and timing

Automatic worker caps use zero; a positive cap limits concurrency. GPU sizing
uses the CUDA-visible device, including a MIG slice, never the first physical
GPU's larger capacity. It intersects CPU allocation/affinity, live RAM and
VRAM budgets. Unknown GPU memory limits automatic inference to one worker.
The implementation is for the current CUDA device, not multi-GPU distribution.

Logs prefixed `[CISEGMENTATION_RESOURCE_CHECK]` contain JSON events:

- `startup`: software version, effective CPUs/RAM, visible GPU memory and
  allocation-related environment values.
- `inference_sizing`, `tile_sizing`, `finalization_sizing`,
  `measurement_sizing`, `extension_sizing`: probes, budgets and selected counts.
- `worker_threads`: actual numerical-backend thread counts.
- `database_budget`: DuckDB threads and database buffer budget.
- `pool_released`, `gpu_recovery`, `inference_retry`, `probe_retry`,
  `tile_retry`: cleanup, refreshed headroom and recovery decisions.

An inference probe includes at least a 512 MiB context/library allowance and
a 1.5 safety factor. Sizing reserves the greater of 2 GiB and 20% of VRAM.
This is conservative budgeting, not authoritative GPU peak accounting.
Likewise, available RAM headroom and a DuckDB buffer limit are not whole-job
MaxRSS. `SLURM_MEM_PER_GPU` is system RAM per allocated GPU, not GPU VRAM.

Known CUDA/cuBLAS allocation failures release owned worker pools before
retrying with fewer workers and fresh budgets. A successful run with zero
retries validates that allocation for that workload; it does not exercise
forced-OOM recovery or prove cleanup of failed pools on the real device.
Do not reuse observed model worker counts as universal settings.

## Reconstruct a timeline

Keep the workflow ID, scheduler ID, software revision and timezone together.
Distinguish whole-workflow elapsed time from the segmentation process and
individual phases. Use recorded event boundaries where available; label
derived times and missing intervals explicitly. Directory names and write
timestamps are observations, not reliable transfer/import boundaries.

Model pass and measurement durations can be read from output provenance;
verify them against the corresponding job log. Source-data copy can overlap
CPU label finalization. Timing counters may aggregate work across workers;
they cannot be summed to infer wall time. `import_seconds` measures Python
library imports. Use the process completion line for process elapsed time.

For output completeness, verify enabled label sets across expected fields and
match `output_store_uuid` between the output store and measurement provenance.
Counts establish structural completeness, not scientific mask accuracy.
Document absent continuous utilization, peak memory or outer-stage timestamps
instead of estimating them from allocation sizes or leftover workflow time.
