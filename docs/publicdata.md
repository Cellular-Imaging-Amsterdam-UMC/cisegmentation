# Retained public microscopy benchmarks

Original downloads and converted images live under `publicdata/`, which is
excluded from Git and the Docker build context. Retain originals, checksums,
reference tracks, citation/licence information and conversion provenance.

Prepare the already downloaded files from the repository root:

```sh
python -m tools.prepare_publicdata --publicdata publicdata
python -m tools.validate_measurement_extensions --output test-results \
  --public-lineage-zip publicdata/Fluo-N2DL-HeLa.zip \
  --public-spots-zip publicdata/ISBI-SubVFI/SIMULATED-test.zip \
  --public-colocalization publicdata/ome-zarr
```

An optional `--copy-to` copies validated stores and reference sidecars to an
importer-visible directory. Existing complete stores are checked against their
sources and never regenerated. A `.partial` store requires inspection after an
interrupted conversion. ZIP members always use POSIX paths, including on Windows.

## Collection

| Dataset | Images | Dimensions | Use |
| --- | ---: | --- | --- |
| ISBI simulations, SNR 7 | 6 | 100 timepoints, 512 × 512, one channel | Microtubule/receptor/vesicle motion, each at low and high density; independent simulation reference tracks |
| Real CCR5, EB1 and lysosome test movies | 3 | 50/34/19 timepoints; 496² / 288 × 352 / 512² | Spot-only detection on experimental data; authors' TrackMate tracks retained separately |
| Colocalization Benchmark Source, CBS001–010 | 10 | 1024², three original RGB channels | Known nominal overlap from 0–90%; red/green Pearson and directional Manders; empty blue control |
| CTC Fluo-N2DL-HeLa sequence 01 | 1 | 92 timepoints, 1100 × 700, one channel | Nuclear tracking, missed frames and divisions; original TRA masks/lineage text retained |
| Visium mouse-brain fluorescence crop | 1 | 7272², DAPI / anti-NeuN / anti-GFAP | Nuclear compartments, neuronal/glial tissue neighborhoods and per-object marker relationships |

The images are NGFF **0.4**, Zarr **v2**, TCZYX with separate channel planes,
256² XY chunks, Blosc/Zstd compression, `_ARRAY_DIMENSIONS`, display channels,
OME-XML and XY intensity pyramids down to at most 128 pixels. Integer pyramid
pixels use rounded 2×2 area means; odd edges are replicated. Original pixels,
dtype and frame ordering are unchanged. Conversion validates every original
plane and pyramid windows. `ome-zarr-catalog.json` records all shapes and levels.

OME-Zarr format compatibility is verified with the installed OMERO Server
`loci.formats.in.ZarrReader`, including pixel windows from every channel and
resolution at the first and last timepoint. This is reader validation, not a
database import. `tools/ValidatePublicZarr.java` repeats that check using the
server's existing Java libraries. The standalone importer Java environment lacks
the native Blosc library needed for pixel decoding; run the pixel probe in the
OMERO Server runtime, which already contains it. This does not require a rebuild.

## Calibration and interpretation

CTC documents 0.645 µm XY pixels and 1800 seconds between frames. The other
downloaded TIFFs do not record physical microscopy pixel size. Those copies
therefore omit spatial units and physical timing; their coordinates remain pixels
and frame indices. TIFF print resolution and TrackMate's placeholder units are
not microscopy calibration. The user's 0.5 µm calibration applies to the separate
40K tilescan, not these public images.

The production spatial/tracking flags require validated physical calibration;
the ISBI assignment benchmark deliberately uses a 20-**pixel** gate on reference
positions and clearly records that distinction. Colocalization and pixel geometry
can be tested without physical units. Do not treat the mouse-brain markers as
expected positive colocalization: neuronal and glial expression may be distinct.

CBS nominal overlap is not the same statistic as directional Manders after Otsu
thresholding. Whole-frame benchmark ROIs include background; per-cell biological
interpretation requires appropriate masks. The empty blue channel produces NULL
Pearson/Manders with recorded reasons. Tracking on reference centroids assesses
assignment only, not detector or segmentation accuracy. CTC TRA markers are not
complete nuclear masks. Division hypotheses need biological review.

## Sources

- [Authors' Zenodo record](https://zenodo.org/records/14043236/):
  `SIMULATED-test.zip` and `REAL.zip`, CC BY 4.0. ISBI simulation XML is independent
  reference truth; `_Tracks.xml` represents the authors' TrackMate output and must
  not be substituted as simulation ground truth.
- [CBS download page](https://colocalization-benchmark.com/downloads/):
  `CBS001RGM-CBS010RGM.zip`, CC BY-NC-SA 4.0. Preserve the source attribution and
  distinguish nominal overlap from measured coefficients.
- [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/):
  `Fluo-N2DL-HeLa.zip`, official training archive. Follow its
  [conditions of use and citation](https://celltrackingchallenge.net/datasets/).
- [Squidpy Visium fluorescence tutorial](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_visium_fluo.html):
  10x Genomics adult mouse-brain coronal section 2. The image is available from the
  [published Figshare file](https://ndownloader.figshare.com/files/26098364), also
  listed in Squidpy's dataset registry. Its checksum matches the registry. Retain
  Squidpy and original 10x dataset attribution.

`downloads.json` records download URLs, bytes and SHA-256 hashes.
`ISBI-SubVFI/zenodo-metadata.json` retains the public source metadata and licence.
