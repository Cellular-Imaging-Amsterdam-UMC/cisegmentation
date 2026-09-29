# Public benchmark inputs

Public downloads and converted stores belong under `publicdata/`, excluded from
Git and image build context. Preserve originals, checksums and reference sidecars.
The repository's `tools.prepare_publicdata` prepares NGFF 0.4/Zarr v2 TCZYX images
with pyramids and validates original pixels. The catalog records calibration.

CTC HeLa sequence 01 has nuclear reference markers and lineage text. ISBI SNR 7
simulations provide independent spot reference trajectories for microtubule,
receptor and vesicle motion, with low/high density. Real CCR5/EB1/lysosome movies
retain author-generated TrackMate trajectories separately from simulation truth.
CBS red/green benchmarks provide known nominal overlap plus an empty blue control.
The Visium DAPI/NeuN/GFAP tissue crop tests compartment and marker relationships;
distinct cell-type markers do not imply expected positive colocalization.

Never substitute author-generated `_Tracks.xml` for independent ISBI simulation
truth. Do not treat CTC TRA markers as complete nuclear masks. Report tracking
accuracy on reference positions separately from image detection accuracy.
Nominal CBS overlap is not expected to equal thresholded directional Manders.

CTC calibration is 0.645 micrometers and 1800 seconds per frame. The other retained
TIFFs lack verified physical calibration; their copies remain pixels/frames.
Production spatial/tracking features require validated physical units. The
reference-coordinate spot benchmark uses an explicitly recorded pixel gate.
Do not copy calibration from a different specimen or interpret TIFF DPI as
microscopy calibration. Colocalization and pixel geometry can be tested without
physical units. Check the current catalog before selecting physical parameters.
