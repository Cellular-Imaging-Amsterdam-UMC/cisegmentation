# Optional measurements and geometry

These additions are disabled by default. Base measurement schema 5, views,
frame-local label values, object IDs and original CLI flags retain their behavior.
Extension schema 1 is recorded separately in `measurement_extensions`.

The workflow UI always writes complete source pixels with final labels. BIOMERO's
shallower may remove verified duplicate arrays before transfer. The original
source is retained. Legacy `--include-original-data false` and
`--include-original-channels false` still request a labels-only overlay.

Enable individual features with `--spatial-measurements true`,
`--colocalization true`, `--tracking true` or `--export-geometry true`.
All require `--measurements-database duckdb` or `sqlite`. Spatial/tracking features
also require positive spatial calibration with units. Calibrated NGFF axes take
priority; missing or unusable axes fall back to `PhysicalSizeX/Y/Z` from the
matching image in the embedded `OME/METADATA.ome.xml`. OME-XML length units are
converted to micrometers. HCS fields are matched through their plate/well and
`WellSample`/`ImageRef`, rather than using the first XML image for every field.
All input fields are checked using metadata only before any segmentation starts.
If neither source supplies usable calibration, the error identifies the input
and the required axes. X/Y are required for 2D; X/Y/Z for 3D. The input metadata
is not rewritten by this fallback. Missing physical time
calibration uses frames; a time scale without units is not assumed to be seconds.
Default physical controls are a 50 µm neighbor radius, 20 µm tracking displacement
and two missed frames. All channel selectors are one-based.

Spot-only example (from the repository root, in a configured inference runtime):

```sh
python wrapper.py --local --infolder input --outfolder output \
  --cell-model skip --nucleus-model skip --foci-model-1 spotiflow:general \
  --foci-channel-1 1 --measurements-database duckdb \
  --tracking true --export-geometry true
```

Tracking uses sparse distance-gated assignment with observed-velocity prediction,
followed by cell/nucleus division hypotheses and missed-frame recovery. Only cells
and nuclei may divide; all other objects have one-to-one trajectories, and merging
is disabled. Division size/distance heuristics are recorded in the analysis reference;
they require biological review and do not guarantee correct lineage reconstruction.
Segmentation itself already processes every timepoint independently.

Colocalization uses original intensities, automatic sampled Otsu thresholds and
directional Manders coefficients; there is no manual threshold override. Pearson
is undefined for constant signals or fewer than two finite paired pixels. Negative
intensity or zero denominators can make Manders undefined; NULL reasons are stored.

Geometry is exported to `<source>__cisegmentation_geometry.duckdb` or `.sqlite`,
matching the main database format. Exact pixel-edge Polygon/MultiPolygon preserves
holes and disconnected components. Volumes use per-Z outlines; point-only Spotiflow
uses native subpixel Point/Point Z through tile origins and final label remapping.
Raster masks remain authoritative. Shared UUIDs/field/label/time keys link files.
Viewer polygon display is a separate integration.

See the updated [measurement analysis reference](../skills/analyze-cisegmentation-measurements/references/EXTENSIONS.md),
[geometry analysis skill](../skills/analyze-cisegmentation-geometry/SKILL.md), and
[workflow parameter reference](../skills/use-cisegmentation-workflow/references/PARAMETERS.md).

## Repeatable validation

```sh
python -m pytest -q
python -m tools.validate_measurement_extensions --output test-results \
  --public-lineage-zip publicdata/Fluo-N2DL-HeLa.zip \
  --public-spots-zip publicdata/ISBI-SubVFI/SIMULATED-test.zip \
  --public-colocalization publicdata/ome-zarr
```

Optional `--gpu-fixture tests/data/nuclei-spots-cytoplasm.ome.zarr` runs positive
2D and native 3D Spotiflow time-series cases using the existing inference runtime.
`--large-input`, `--large-labels`, and `--large-measurements` reuse a completed 40K
segmentation to export spatial extensions over all its objects. This avoids
repeating model inference and does not validate full 40K geometry export. The
script records dynamically discovered resources, peak process-tree RSS, output
counts and scientific checks. It never builds an image.

An uncalibrated synthetic 40K source can be used for a resource benchmark on
Linux with `--synthetic-pixel-um 1`. This creates a clearly marked metadata view
and read-only data links under the test output; it leaves the original untouched.
Its distances are test quantities and must not be used biologically. Production
calibration validation remains strict.

Public datasets are retained under `publicdata/`, excluded from Git and Docker
build context. The [Cell Tracking Challenge Fluo-N2DL-HeLa](https://celltrackingchallenge.net/2d-datasets/)
training archive is downloaded from its official source, with a checksum recorded
in the validation output. Its documented 0.645 µm pixel size and 30-minute interval
calibrate the reference-marker tracking test. Reference IDs are used only for
scoring, never for assignment; this checks tracking, not segmentation accuracy.
The dataset's [conditions of use and citation](https://celltrackingchallenge.net/datasets/)
apply to publications.

The retained spot simulations provide independent ISBI reference trajectories;
real CCR5/EB1/lysosome movies also retain their authors' TrackMate trajectories.
The colocalization benchmark checks bounded per-object coefficients against
independent full-frame calculations, including the constant/empty blue channel.
Read [publicdata.md](publicdata.md) for the converted image catalog, source links,
calibration limitations, and the preparation command.
