# Parameters

Streaming parameters (advanced): `streaming_mode` is `auto` (default) or `on`.
`tile_size` (default 1024, minimum 64) is an upper bound on XY core size;
`tile_overlap` (default 96) supplies context on each XY side. Native 3D models
also use `tile_depth` (default 32) and `tile_overlap_z` (default 8).
Native 3D Spotiflow enforces minimum halos of 64 slices in Z and 192 pixels
in XY; effective overlaps are included in provenance and memory estimates.
`tile_match_threshold` (default 0.5, greater than 0 and at most 1) controls
mutual instance matching in overlapping predictions. Allocation discovery and
live memory checks can reduce cores below the configured upper bound.

Use the descriptor from the configured workflow revision as the executable
contract. Reject undocumented values and do not silently introduce parameters.
All channel numbers are one-based and must not exceed the selected image's
channel count. At least one of Step 1, Step 2, or Step 3a–3d must be enabled.

## Segmentation steps

| Name | Type | Required | Default | Constraints | Meaning |
| --- | --- | --- | --- | --- | --- |
| `cell_model` | choice | no | `cellpose3:cyto3` | `skip`, `cellpose3:cyto3`, `cellpose-sam:cpsam_v2`, `cellpose-sam:cpsam`, `instanseg:fluorescence_nuclei_and_cells`, or an `expand:` choice listed below | Step 1 direct cell segmentation or nucleus-seeded cell expansion. |
| `cell_channel` | integer | no | `1` | `>=1`; valid input channel | Primary cell/cytoplasm channel, or an expansion seed channel. |
| `cell_nuclei_channel` | integer | no | `0` | `>=0`; `0` omits it | Optional nucleus channel for a direct cell model. For expansion, a value above `1` takes precedence over `cell_channel`. |
| `cell_expansion_distance` | number, µm | no | `10.0` | `>=0` | Maximum XY distance for nucleus-seeded cell expansion. |
| `nucleus_model` | choice | no | `skip` | `skip` or a Step 2 model listed below | Independent Step 2 nuclei detection and cell/nucleus matching. |
| `nucleus_channel` | integer | no | `1` | `>=1`; valid input channel | Signal channel for Step 2 nuclei detection. |
| `foci_model_1` | choice | no | `skip` | `skip` or a Step 3 model listed below | Step 3a spot, focus, or bacterium detection. |
| `foci_channel_1` | integer | no | `1` | `>=1`; valid input channel | Input channel for Step 3a. |
| `foci_model_2` | choice | no | `skip` | same as Step 3a | Advanced Step 3b detection slot. |
| `foci_channel_2` | integer | no | `1` | `>=1`; valid input channel | Input channel for Step 3b. |
| `foci_model_3` | choice | no | `skip` | same as Step 3a | Advanced Step 3c detection slot. |
| `foci_channel_3` | integer | no | `1` | `>=1`; valid input channel | Input channel for Step 3c. |
| `foci_model_4` | choice | no | `skip` | same as Step 3a | Advanced Step 3d detection slot. |
| `foci_channel_4` | integer | no | `1` | `>=1`; valid input channel | Input channel for Step 3d. |

Step 1 expansion choices are
`expand:cellpose3:nuclei`, `expand:cellpose-sam:cpsam_v2`,
`expand:cellpose-sam:cpsam`, `expand:stardist:SD_Nuclei_Versatile`, and
`expand:instanseg:single_channel_nuclei`.

Step 2 choices are `cellpose3:nuclei`, `cellpose-sam:cpsam_v2`,
`cellpose-sam:cpsam`, `stardist:SD_Nuclei_Versatile`,
`instanseg:single_channel_nuclei`, and
`instanseg:fluorescence_nuclei_and_cells`.

Step 3 choices are `spotiflow:general`, `spotiflow:hybiss`,
`spotiflow:synth_complex`, `spotiflow:synth_3d`,
`spotiflow:smfish_3d`, `spotiflow:fluo_live`,
`stardist:SD_Foci_Aggregates`, `stardist:SD_Foci_Finn`,
`cellpose3:bact_phase_cp3`, and `cellpose3:bact_fluor_cp3`.

## Outputs and post-processing

| Name | Type | Required | Default | Constraints | Meaning |
| --- | --- | --- | --- | --- | --- |
| `remove_border_cells` | boolean | no | `true` | — | Remove cells touching an XY border and their matched nuclei/cytoplasm. |
| `existing_labels` | choice | no | `overwrite` | `remove`, `overwrite`, `append` | Replace the complete labels tree, overwrite generated-name collisions while preserving unrelated labels, or append collision-safe names. |
| `measurements_database` | choice | no | `duckdb` | `duckdb`, `sqlite`, `skip` | Create object, intensity, and relationship measurements. |
| `labels_log_info` | boolean | no | `false` | Advanced | Calculate extra label statistics; increases full-array scanning. |
| `benchmark` | boolean | no | `false` | Advanced; changes the output contract | Benchmark eligible models on the first image/field, first timepoint, and a centered XY region up to 1024×1024. |

Generated segmentations are always native OME-Zarr label groups. The legacy
`include_original_channels` argument maps to `include_original_data`, and the
legacy `write_ome_zarr_labels` argument is accepted but ignored for one
compatibility period. Benchmark mode emits only a comparison gallery and no
measurements database.

The UI always produces a complete copy of the original data with final native
labels, retaining the source. A transfer service may remove verified duplicate
arrays afterwards. `--include-original-data` remains a legacy CLI flag with its
original behavior: true copies source pixels; false writes a labels-only overlay.
The UI no longer offers an Include Original Data checkbox.

## Optional measurements, tracking and geometry

All features below are disabled by default and require DuckDB or SQLite
measurements. Inspect the configured descriptor: older executables will not
accept these new options.

| Name | Default | Meaning |
| --- | --- | --- |
| `spatial_measurements` | `false` | Same-label-set/frame nearest neighbors, radius counts, touching neighbors, shared boundaries and spot counts in available cell/nucleus masks. Requires calibrated spatial units. |
| `spatial_radius_um` | `50.0` | Positive neighbor radius in micrometers. |
| `colocalization` | `false` | Per-object Pearson and directional Manders values from original channel intensities. |
| `colocalization_pairs` | `all` | All distinct channel pairs, or comma-separated one-based pairs such as `1:2,1:3`. Thresholds use automatic Otsu; there is no manual override. |
| `tracking` | `false` | Sparse distance-gated two-stage assignment within each field and label set. Requires calibrated spatial units; timing falls back to frames. |
| `tracking_divisions` | `true` | Applied only when tracking is enabled, and only to cells/nuclei. Spots, foci, bacteria and cytoplasm remain one-to-one. No merging. |
| `tracking_distance_um` | `20.0` | Positive maximum displacement for adjacent or gap assignments. |
| `tracking_max_gap` | `2` | Nonnegative maximum missed frames; two means a three-frame link is permitted. |
| `export_geometry` | `false` | Separate matching-format database of final-mask polygons/per-Z outlines or native Spotiflow subpixel points. |

Spot-only configuration: `cell_model=skip`, `nucleus_model=skip`,
`foci_model_1=spotiflow:general`, and the requested `foci_channel_1`.
Tracking and geometry work without cell/nucleus segmentation. Native subpixel
coordinates remain available after overlapping tile inference and label remapping.
Local mask refinement exports final mask polygons instead of native point geometry.
Segmentation already supports timepoints; tracking adds trajectories to that output.

Memory budgets are discovered from the current allocation and checked live.
Objects and polygons use bounded crops and database batches, polygon edges spill
to disk, and assignment uses sparse candidate graphs. If one graph or polygon
still cannot fit, report the resource error; do not silently change scientific
distance gates or discard vertices. Additional output requires temporary disk space.

## Runtime and model tuning

| Name | Type | Required | Default | Constraints | Meaning and resource effect |
| --- | --- | --- | --- | --- | --- |
| `device` | choice | no | `auto` | `auto`, `cuda`, `cpu` | `cuda` requires an available compatible GPU; `cpu` is usually slower. |
| `max_inference_workers` | integer | no | `0` | `>=0` | Cap model workers; zero uses conservative automatic sizing from PyTorch and NVIDIA process memory, reserving at least 2 GiB or 20% of VRAM with a 50% worker margin. |
| `max_measurement_workers` | integer | no | `0` | `>=0` | Cap spawned CPU measurement workers; zero uses the available allocation while reserving one coordinator. |
| `dimension_mode` | choice | no | `auto` | `auto`, `slice-2d` | Native 3D where supported or independent slice-wise 2D; 3D generally uses more memory. |
| `diameter` | number, µm | no | `0.0` | `>=-1` | Cellpose expected diameter. `0` uses workflow defaults; `-1` uses the model default. |
| `cellprob_threshold` | number | no | `0.0` | finite | Cellpose object probability threshold. |
| `flow_threshold` | number | no | `0.4` | `>=0` | Cellpose flow error threshold. |
| `stardist_prob_threshold` | number | no | `-1.0` | finite | StarDist probability threshold; `-1` reads the checkpoint value. |
| `stardist_nms_threshold` | number | no | `-1.0` | finite | StarDist non-maximum-suppression threshold; `-1` reads the checkpoint value. |
| `smooth_stardist_labels` | boolean | no | `true` | — | Rasterize rescaled StarDist polygons on the source grid. |
| `spotiflow_prob_threshold` | number | no | `-1.0` | finite | Spotiflow probability threshold; `-1` uses the checkpoint default. |
| `spotiflow_min_distance` | number, µm | no | `1.0` | `>=0` | Minimum detected-point separation. |
| `spotiflow_local_refinement` | boolean | no | `false` | Native 3D Spotiflow requires `slice-2d` | Grow detected points into bounded local signal masks; adds processing. |

## Conditional validation

- Reject channel `0` except for `cell_nuclei_channel`, where it means omitted.
- For direct Step 1 models, pass a distinct positive
  `cell_nuclei_channel` as the optional second channel.
- For Step 1 expansion, use `cell_nuclei_channel` when it is above `1`;
  otherwise use `cell_channel`.
- Allow repeated Step 3 models and channels; each enabled slot remains a
  separate output label set.
- Reject `device=cuda` when the consumer reports no compatible GPU resource.
- Keep the defaults for model thresholds unless the user has a reason to tune
  them. A conservative normal-run example is Step 1 `cellpose3:cyto3` on
  channel `1`, all other detection steps skipped, `device=auto`,
  `existing_labels=overwrite`, and DuckDB measurements enabled.
