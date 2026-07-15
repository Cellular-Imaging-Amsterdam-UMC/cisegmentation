# CI Segmentation OMERO round-trip probe

This runner uses the established `omero_import_metadata_probe` from the local
`cideconvolve` checkout to test CI Segmentation outputs through both direct
OMERO and BIOMERO import paths. The existing probe remains the source of truth
for importer-container access, metadata comparison, cleanup, and reports.

`omero_import_metadata_probe.py` is a local compatibility shim around that
probe. It permits OMERO label channels without an assigned rendering color;
the color is omitted from generated OME-XML instead of failing during export.
It also forces NGFF 0.4 exports to use Zarr v2 because the current probe's
Zarr dependency otherwise defaults to Zarr v3, which Bio-Formats 8.4 and
QuPath 0.7 cannot display. The shim also writes the NGFF `_ARRAY_DIMENSIONS`
attribute required for Bio-Formats to identify the axes.
OMERO length-unit enum names such as `MICROMETER` are converted to valid
OME-XML unit symbols such as `µm`.

For registered Zarr input, `--retain-container-staging` keeps a host-staged
store available after registration. The roundtrip runner uses this because
OMERO references that path lazily, and removes it only after successful export
and OMERO cleanup.
Copied inputs are placed below `/data/.cisegmentation_roundtrip`, a mount shared
by the importer and OMERO Server containers; container-private `/tmp` paths
cannot serve registered Zarr pixels to OMERO.

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
