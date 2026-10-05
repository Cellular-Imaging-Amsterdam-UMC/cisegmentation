# Random fields of real nuclei

`create_seamless_nuclei_tilescan.py` generates a synthetic image from the real
single-channel TIFF. The default experiment has the dimensions of four by
four source tiles, but the nuclei are distributed over the entire canvas.
For `1_B02__cells.tif`, each source tile is 2008 × 2008 pixels and the test
output is **8032 × 8032**.

## Default: whole-canvas random placement

1. Run the repository's bundled StarDist nuclei adapter locally on the entire
   small source image, or accept an existing instance mask with `--mask`.
2. Select complete, connected real nuclei away from the source boundary, with
   a broad range of sizes. Exclude other nuclei from each cutout's feathered
   halo, allowing nuclei with nearby neighbours to contribute their own texture.
3. Sample positions uniformly across the whole output, with random rotations
   and reflections. Accept only placements with nonoverlapping nucleus masks
   and the requested minimum gap (one pixel by default). The target object
   count matches the source object density scaled to the output area.
4. Render a continuous, stationary background whose level and noise statistics
   are estimated from the source. Coordinate-based noise is independent of
   writer chunks and has no tile boundaries.
5. Blend the six-pixel fluorescence halos into the background, protecting every
   nucleus's original core pixels. Neighbouring halos may overlap; nucleus
   cores cannot overlap or be overwritten by another patch's halo.

There are no fixed border strips, seam-centred rows, or forced nuclei at grid
junctions. The original source arrangement is not repeated. Whole nuclei
naturally cross the former tile coordinates and writer chunk boundaries.
Only complete objects inside the outer image boundary are placed.

The older `--layout tile-repair` method is retained for comparison. It copies
rotated source interiors, removes whole edge nuclei, and inserts shared donor
patches at joins. Its cleared strips and packing constraints can still make
the underlying grid visible; `random-field` is the default for new images.

This tool rearranges real nucleus cutouts on an estimated synthetic background.
The accompanying reference masks are derived from the source model
segmentation and donor placements, not manually annotated biological ground
truth. The source TIFF is not modified. Spatial calibration is written only
when `--pixel-size-um` is provided; TIFF print resolution is not interpreted
as microscopy calibration.

## Run the 4 × 4 experiment

Use the cisegmentation environment, not an unqualified `python`. On this host
the existing environment is at `V:\BIOMERO-local\tests\cisegmentation`:

```powershell
& 'V:\BIOMERO-local\tests\cisegmentation\python.exe' `
  tools/create_seamless_nuclei_tilescan.py `
  'V:\NL-BIOMERO\web\L-Drive\Project B\1_B02__cells.tif' `
  'V:\NL-BIOMERO\web\L-Drive\Project B\1_B02__nuclei-random-field-4x4.ome.zarr' `
  --rows 4 --columns 4 --layout random-field --export-tiff
```

The default model is `stardist:SD_Nuclei_Versatile`, using the checked-in
`bundled_models` directory. CPU inference works; `--device cuda` can be used
in a CUDA-enabled environment. Model inference uses the original pixel grid.
The optional `--models-root` overrides the model-cache root.

For repeat experiments, `--mask PATH` accepts the saved source mask TIFF or
a NumPy `.npy` instance mask with background zero. Sparse source IDs are
compacted; the synthetic output receives globally unique uint32 IDs.

Output and diagnostics directories must not already exist. Choose another
output name for a different seed or parameter set.

## Outputs and review

- OME-Zarr with uint16/uint8 intensity pixels and an area-averaged
  pyramid. Writing and TIFF export use bounded-memory chunks/strips.
- `labels/labels_nuclei_reference`: uint32 source/donor reference masks with
  unique IDs, a nearest-neighbour pyramid, and an explicit display window.
- `<output-stem>_diagnostics/source-nuclei-mask.tif` and
  `border-cleaned-source.tif`, plus the cleaned source mask.
- `source-before-after.png`, `eight-cleaned-orientations.png`,
  `tilescan-overview.png`, and `seam-before-after.png`. The seam contact sheet
  compares the old uncorrected arrangement against the corrected one at the
  same contrast and includes reference-mask outlines.
- `placement-plan.json` records each donor's source ID, location, orientation,
  and assigned output ID. `report.json` records the source inference settings,
  removal counts, donor counts, and validation results. `grid_visibility`
  compares mask foreground coverage around former joins against the interior,
  excluding the outer canvas edge. A ratio near 1 indicates no density deficit
  along those coordinates. The Zarr also carries
  the summary and source checksum.
- `tilescan.tif` when `--export-tiff` is requested.

The geometry checks verify that every donor mask remains connected, its
original nucleus pixels remain unchanged, and no reference nuclei are
truncated at the outer canvas boundary. Unit tests also check all eight
orientations, four-way junctions, partial tiles, arbitrary chunk boundaries,
non-overwriting behavior, high uint32 IDs, odd-sized pyramids, and TIFF export.
Additional tests verify preservation of core pixels despite overlapping halos,
reproducible random fields, and the absence of grid-boundary density deficits
or rows locked to the grid.

```powershell
& 'V:\BIOMERO-local\tests\cisegmentation\python.exe' `
  -m pytest tools/test_seamless_nuclei_tilescan.py -q
```

## Generate 40000 × 40000 after reviewing the test

```powershell
& 'V:\BIOMERO-local\tests\cisegmentation\python.exe' `
  tools/create_seamless_nuclei_tilescan.py `
  'V:\NL-BIOMERO\web\L-Drive\Project B\1_B02__cells.tif' `
  'V:\NL-BIOMERO\web\L-Drive\Project B\1_B02__nuclei-random-field-40000x40000.ome.zarr' `
  --height 40000 --width 40000 --layout random-field `
  --mask 'V:\NL-BIOMERO\web\L-Drive\Project B\1_B02__nuclei-random-field-4x4_diagnostics\source-nuclei-mask.tif'
```

The original `create_tilescan_ome_zarr.py` retains its existing CLI behavior.
Its Python writer now also accepts a custom renderer and an optional uint32
reference-mask renderer; the new seam-aware tool reuses that writer.
