# Public microscopy inputs

Original downloads and converted stores live under `publicdata/`, excluded from
Git and Docker build context. Preserve original images, checksums, source
attribution, dataset licences and conversion provenance.

```sh
python -m tools.prepare_publicdata --publicdata publicdata
python -m tools.validate_measurement_extensions --output test-results \
  --public-colocalization publicdata/ome-zarr
```

`--copy-to` copies validated stores into an importer-visible directory. Existing
complete stores are checked against their sources. Inspect `.partial` stores
after interrupted conversion before retrying.

| Dataset | Images | Purpose |
| --- | ---: | --- |
| ISBI SNR 7 spot simulations | 6 | Low/high-density spot detection |
| Real CCR5, EB1 and lysosome movies | 3 | Spot detection on experimental images |
| CBS001–010 red/green images | 10 | Pearson/Manders coefficients and empty blue controls |
| Visium mouse-brain DAPI/NeuN/GFAP crop | 1 | Compartments and marker relationships |

The converter writes NGFF 0.4 / Zarr v2 TCZYX stores with 256² XY chunks,
Blosc/Zstd compression, `_ARRAY_DIMENSIONS`, OME-XML and intensity pyramids.
Original pixels, dtype and frame order are validated. The catalog records shapes,
levels and calibration. These source TIFFs have no verified physical microscopy
pixel sizes: converted stores therefore retain pixels and frame indices. Never
interpret TIFF print resolution as microscopy calibration. Physical spatial
measurements require calibrated inputs; colocalization and pixel geometry do not.

CBS nominal overlap differs from thresholded directional Manders. NULL reasons
record constant/empty channels. Distinct neuronal and glial markers need not
colocalize biologically. Reader compatibility does not establish biological accuracy.

## Sources

- [Authors' Zenodo record](https://zenodo.org/records/14043236/):
  `SIMULATED-test.zip` and `REAL.zip`, CC BY 4.0.
- [CBS downloads](https://colocalization-benchmark.com/downloads/):
  `CBS001RGM-CBS010RGM.zip`, CC BY-NC-SA 4.0.
- [Squidpy Visium fluorescence tutorial](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_visium_fluo.html):
  original 10x Genomics mouse-brain image, also available through
  [Figshare](https://ndownloader.figshare.com/files/26098364).

`downloads.json` records URLs, sizes and hashes. Preserve the source licence
metadata. Use `tools/ValidatePublicZarr.java` in an OMERO Server environment
with the Bio-Formats ZarrReader and native Blosc library for pixel reader checks.
