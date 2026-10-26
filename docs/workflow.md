# Workflow and parameters

## Input and output contract

- Input: one or more top-level `.ome.zarr` stores in `/data/in`, including HCS plates.
- Normal output: `<source>__cisegmentation.ome.zarr` in `/data/out`, containing
  native OME-Zarr label groups. By default the source store is copied with its
  original pixels and remains untouched. A configured BIOMERO shallower can remove
  verified duplicate arrays before transfer. The UI no longer offers Include
  Original Data; legacy CLI flags retain their labels-only overlay behavior.
- Benchmark output: **only** `benchmark_gallery_<image>.ome.zarr`.
- Axes are normalized to `TCZYX`; time and Z are preserved in normal runs.
- Output OME-Zarrs and schema-v5 measurement databases share an
  `output_store_uuid`; `object_navigation` supplies portable field, label, and
  ROI coordinates without embedding deployment-specific OMERO IDs.
- Spotiflow points become uniquely numbered single pixels or voxels by default.
  Advanced local refinement can replace 2D or slice-wise points with bounded,
  locally thresholded instance masks.

The internal `wrapper.py` is the container/Bilayers entrypoint. It is not
packaged as an end-user CLI. `launcher.py` is the supported local frontend and
constructs the Docker invocation from `config.yaml`.

## Parameters

All channel numbers shown to users are one-based. Physical parameters are
converted internally using the OME-Zarr XY scale metadata.

| Parameter | Use |
| --- | --- |
| Step 1: Cell Detection (`--cell-model`) / Cyto Channel / Nucleus Channel | Selects `Skip`, a direct cell model, or `Cell expansion using …`. Direct models receive the Cyto Channel and, when the optional nucleus channel is greater than zero and different, that channel as a second input. For expansion, the seed channel may be entered in either Step 1 channel field. A channel above 1 is preferred over a default-like 0 or 1, and the Nucleus Channel wins when both are above 1. This input does not independently create matched nucleus labels; select Step 2 for that. |
| Step 1 Expansion Distance | Sets the maximum XY expansion distance in µm. Physical X/Y scales are read from OME-Zarr metadata. Expansion produces matched cell, nucleus, and cytoplasm channels directly. |
| Step 2: Nuclei Detection (`--nucleus-model`) / Channel | Selects `Skip` or an independent nucleus model. When cells and nuclei are both available, they are matched by overlap; only the largest nucleus per cell is retained, cells without nuclei are removed, and cytoplasm is written with shared IDs. Step 2 may repeat the nucleus model used for Step 1 expansion. |
| Step 3a–3d: Foci Detection (`--foci-model-1` … `--foci-model-4`) / Channel | Step 3a is a beginner selector; Steps 3b–3d appear first in the advanced options. Each offers `Skip`, Spotiflow, `SD_Foci_*` StarDist, and Cellpose 3 `bact` models. Repeating models or channels is allowed. StarDist outputs are named `foci`; Cellpose bacterial outputs are named `bacteria`. |
| Original data (legacy CLI only: `--include-original-data`) | Removed from the UI. Normal output copies source pixels and final labels, retaining the input. The CLI still accepts false for a labels-only overlay; `--include-original-channels` retains its legacy mapping. |
| Existing Labels (`--existing-labels`) | `overwrite` (default) replaces generated-name collisions while preserving unrelated labels; `remove` replaces the complete labels tree; `append` preserves all groups and assigns collision-safe suffixes consistently across the plate. |
| Maximum Inference / Measurement Workers | Advanced caps for automatically sized inference and measurement pools. Zero means automatic based on CUDA-visible GPU/MIG memory, CPU allocation/affinity, and RAM. GPU sizing uses the greater of the PyTorch peak plus a 512 MiB context/library allowance and NVIDIA's complete worker-process memory, reserves the greater of 2 GiB or 20% of VRAM, and applies a 50% per-worker safety margin. |
| Labels Log Info (`--labels-log-info`) | Advanced option, disabled by default. Calculate and log per-step and final label counts, foreground fraction, and label size statistics. Leave disabled for faster processing of large fields. |
| Smooth Rescaled StarDist Labels (`--smooth-stardist-labels`) | Advanced option, enabled by default. When StarDist inference downsamples a high-magnification image, its polygons are scaled and rasterized directly on the source grid for smooth boundaries. Disable it to reproduce nearest-neighbor label-map restoration. |
| Remove Border Cells (`--remove-border-cells`) | Advanced option, enabled by default. Removes cells touching an XY image edge and propagates removal to matched nuclei and derived cytoplasm. Z-stack endpoints are not treated as image borders. |
| Compute Device (`--device`) | `auto` selects CUDA when available; `cuda` requires a GPU; `cpu` forces CPU inference. |
| Dimension Mode (`--dimension-mode`) | `auto` uses native 3D where supported; `slice-2d` independently segments and relabels every Z plane. |
| Cellpose Diameter (`--diameter`) | Object diameter in µm, converted using mean XY pixel size. `0` resolves to 12 µm for nuclei or 25 µm for cells; a negative value uses the model default. |
| Cellpose Probability Threshold (`--cellprob-threshold`) | Cellpose cell-probability acceptance threshold. Higher values generally produce fewer masks. |
| Cellpose Flow Threshold (`--flow-threshold`) | Cellpose flow-consistency error threshold. |
| StarDist Probability Threshold (`--stardist-prob-threshold`) | Minimum object probability. `-1` loads `prob` from the selected model's `thresholds.json`. |
| StarDist NMS Threshold (`--stardist-nms-threshold`) | Allowed overlap during non-maximum suppression. `-1` loads `nms` from `thresholds.json`. |
| Spotiflow Probability Threshold (`--spotiflow-prob-threshold`) | Spot acceptance threshold. `-1` uses the checkpoint default. |
| Spotiflow Minimum Distance (`--spotiflow-min-distance`) | Minimum separation in µm, converted to pixels from the mean XY pixel size. |
| Spotiflow Local Mask Refinement (`--spotiflow-local-refinement`) | Advanced checkbox, disabled by default. Lightly smooths the selected channel, estimates background and noise around every Spotiflow point, and grows only the seed-connected signal. Growth is bounded to a 1.0 µm radius, overlaps are assigned by local signal-to-noise score, and weak points remain single pixels. Native-3D Spotiflow requires Force slice-wise 2D. The former `--spotiflow-microsam-refinement` flag remains a hidden compatibility alias. |
| Create Measurements Database (`--measurements-database`) | Beginner selector: `duckdb` (default), `sqlite`, or `skip`. Writes one database per top-level image or HCS screen containing final-object shape/location features, per-original-channel intensity statistics, and pairwise mask relationships. See the [measurements database reference](measurements.md). |
| Benchmark Gallery (`--benchmark`) | Advanced option. Processes the first deterministic image/field and first timepoint, runs all selectable models for every enabled step, then writes only a 2D XY OME-Zarr gallery. |

Optional spatial measurements (`--spatial-measurements`), channel colocalization
(`--colocalization`) and a separate geometry
database (`--export-geometry`) are disabled by default. They require DuckDB or
SQLite measurements and add extension schema 3 while preserving base schema 5
and raster label values. See [measurement extensions](measurement-extensions.md) for units,
controls, scientific limits and repeatable validation. Spot-only runs can skip
both cell and nucleus detection. Retained public inputs and their pyramidal
OME-Zarr conversion are documented in [public datasets](publicdata.md).

`SD_Nuclei_Versatile` is automatically downsampled to 0.5 µm/px per XY axis
when the source resolution is finer. By default, its predicted polygons are
rasterized directly on the source grid for smooth boundaries; the advanced
toggle can restore the previous nearest-neighbor behavior. The model is based
on DSB2018 nuclei data described by the
[official StarDist project](https://github.com/stardist/stardist),
and 0.5 µm/px matches the detection resolution in the
[official QuPath StarDist example](https://qupath.readthedocs.io/en/latest/docs/deep/stardist.html).
Other StarDist checkpoints retain their native input scale.
InstanSeg always reads pixel size directly from OME-Zarr metadata.

Normal label outputs store one channel per result type and use a rendering
window from zero through that channel's maximum label gray value. The metadata
requests OMERO's `glasbey_inverted.lut`; its value zero is black. Because NGFF
0.4 does not standardize LUT selection, non-OMERO readers also receive
non-black semantic fallback colors. The launcher OMERO roundtrip explicitly
applies and saves the Glasbey LUT after BIOMERO imports each result.

Normal inference always uses this optional-step workflow; there is no separate
single-model mode. If Cell Detection, Nuclei Detection, and all four Foci
Detection slots are disabled, the workflow stops with a clear validation
error. Original data channels do not receive the Glasbey label LUT and retain
their source names, colors, and display windows when those are available.
