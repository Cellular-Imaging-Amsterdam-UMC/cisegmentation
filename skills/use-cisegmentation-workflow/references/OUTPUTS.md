# Outputs

## Normal runs

For each top-level input, expect one final OME-Zarr named:

```text
<source>__cisegmentation.ome.zarr
```

With native labels enabled, original pixels and datatype remain at the root and
each segmentation is an OME-Zarr 0.4 image-label under `labels/`. Possible label
sets include cells, nuclei, cytoplasm, spots, foci, and bacteria. Duplicate
Step 3 selections receive collision-safe group names. Regular images remain
images; HCS inputs retain their plate/well/field organization.

When `measurements_database` is not `skip`, also expect exactly one database
file per top-level input:

```text
<source>__cisegmentation_measurements.duckdb
<source>__cisegmentation_measurements.sqlite
```

Only the selected database format is produced. The database contains workflow
provenance, image/channel metadata, explicit label-producing input channels,
object morphology and locations, per-original-channel intensity statistics,
and mask relationships. Use the separate
`analyze-cisegmentation-measurements` skill for substantive analysis.

## Benchmark runs

Benchmark mode produces only:

```text
benchmark_gallery_<image>.ome.zarr
```

It is a rendered model-comparison gallery from the first deterministic
image/field and first timepoint, not a normal instance-label result. It does not
produce a measurements database.

## Completion checks

Treat a run as successful only when:

1. the workflow execution interface reports a successful final status for the
   retained run ID;
2. the expected final OME-Zarr store is discoverable in the output location;
3. the result is associated with the intended source input;
4. expected label sets or the benchmark gallery can be enumerated;
5. the requested database file is present and has the expected
   extension, unless measurements were skipped.

Report missing or partial outputs even if the scheduler job ended. Do not claim
that optional labels exist when their corresponding steps were skipped.
