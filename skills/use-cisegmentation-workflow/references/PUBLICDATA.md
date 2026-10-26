# Public benchmark inputs

Public downloads and converted stores belong under `publicdata/`, excluded from
Git and image build context. Preserve originals, checksums and source licences.
`tools.prepare_publicdata` writes NGFF 0.4 / Zarr v2 TCZYX images with pyramids
and validates original pixels. Consult `docs/publicdata.md` and the generated
catalog for the source list and calibration.

ISBI simulations and CCR5/EB1/lysosome movies exercise spot detection. CBS images
provide nominal red/green overlap plus an empty blue control. The Visium crop
provides DAPI, NeuN and GFAP channels for compartment and marker relationships.
Nominal overlap does not equal directional Manders. Distinct markers do not imply
positive colocalization. These TIFFs lack verified physical calibration and remain
in pixels/frames; never copy calibration from a different specimen or use DPI.
Spatial features require validated physical units. Colocalization and pixel
geometry can be tested without them.
