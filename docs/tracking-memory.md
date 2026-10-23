# Tracking memory and incomplete tracking

Production tracking spools observations, links, assignments and lineage state to
a temporary SQLite database beside the other staged outputs. Only the two
frames being compared and their sparse matching graph are loaded into RAM.
Track summaries and database output rows are streamed from the spool. This
preserves the existing two-stage LAP linking, velocity prediction, gap closing,
division policy and deterministic object order.

Non-spatial optional measurements read object rows and native point coordinates
in batches of 512. Spatial neighbor measurements still require a complete frame
and keep their separate memory guard.

Frame and candidate-graph guards follow the effective job allocation and live
headroom discovered by `resources.snapshot`: Slurm memory variables, cgroup
limits, process memory and host availability. No 16 GB limit is assumed in the
tracking code. A large individual frame or dense candidate graph can still
exceed the live budget. Scratch storage must accommodate the tracking spool.

Ordinary measurements and geometry commit before optional tracking runs. Each
label set's tracking rows commit in a separate transaction. On a catchable
tracking error, including a `MemoryError`, that label set's tracking rows are
rolled back; segmentation, ordinary measurements, geometry and successful
tracking for other label sets remain available. The log states `Tracking FAILED
due to memory` for memory failures. Other tracking exceptions retain their own
error type and message. A partial tracking result is never presented as complete.

Extension schema version 2 adds `tracking_status` with one status row per label
set. `details_json` records the failure reason and, when tracking starts, the
effective RAM limit and headroom. The output OME-Zarr provenance also records
these statuses under `parallelism.measurement_extensions.tracking`. A successful
segmentation workflow can therefore contain `failed` optional tracking: check
the log or status before relying on tracks. Base measurement schema version 5
and the existing tracking result tables retain their column definitions.

An operating-system OOM kill, power loss or full output disk cannot be caught by
Python and is outside this recovery guarantee.

For Slurm, use a partition's configured memory default instead of embedding a
5 GB request in a generic job template. A 16 GiB partition can set
`DefMemPerNode=16384 MaxMemPerNode=16384`, with node `RealMemory` configured to
match its usable capacity. Explicit job memory requests still take precedence.
Slurm passes the actual allocation to the workflow; a different partition can
provide a different memory budget without changing the tracking code. See the
[Slurm memory configuration](https://slurm.schedmd.com/slurm.conf.html) and
[sbatch memory options](https://slurm.schedmd.com/sbatch.html#OPT_mem).
