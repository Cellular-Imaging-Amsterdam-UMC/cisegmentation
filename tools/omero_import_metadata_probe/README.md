# CI Segmentation OMERO round-trip probe

This runner uses the established `omero_import_metadata_probe` from the local
`cideconvolve` checkout to test CI Segmentation outputs through both direct
OMERO and BIOMERO import paths. The existing probe remains the source of truth
for importer-container access, metadata comparison, cleanup, and reports.

Prerequisites:

- The local NL-BIOMERO Docker stack is running.
- The `cisegmentation` Conda environment and model cache are prepared.
- `C:\rahoebe\Python\cideconvolve\tools\omero_import_metadata_probe` exists,
  or `--probe-dir` points to another checkout containing the probe.

Export an existing OMERO image to Slurm-input OME-Zarr, segment it, and import
the result through both paths:

```cmd
tools\omero_import_metadata_probe\run.cmd ^
  --existing-image 12345 ^
  --target Dataset:123 ^
  --model cellpose3:nuclei ^
  --target-type nuclei
```

Test an existing regular or HCS OME-Zarr:

```cmd
tools\omero_import_metadata_probe\run.cmd ^
  --input tests\data\nuclei-small.ome.zarr ^
  --target Dataset:123
```

For an HCS plate, use a `Screen:ID` target. The runner writes the segmentation,
the direct/BIOMERO probe reports, command logs, and a combined summary below
the requested report directory. `--cleanup success` is passed through to the
underlying probe.
