from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time
from dataclasses import replace
from collections.abc import Callable

import numpy as np

from .adapters import segment_czyx
from .benchmark import run_benchmark
from .ome_zarr_io import (
    LabelResult,
    discover_ome_zarrs,
    enumerate_resources,
    read_image,
    write_hcs_plate,
    write_label_image,
)
from .registry import get_model_spec
from .reporting import (
    emit,
    format_label_statistics,
    format_step_record,
    input_report_lines,
    label_statistics,
    step_record,
    workflow_report_lines,
)
from .settings import SKIP, SegmentationSettings


_INFERENCE_CACHE_SETTING_NAMES = (
    "model",
    "target",
    "primary_channel",
    "nuclei_channel",
    "device",
    "dimension_mode",
    "diameter",
    "cellprob_threshold",
    "flow_threshold",
    "stardist_prob_threshold",
    "stardist_nms_threshold",
    "smooth_stardist_labels",
    "spotiflow_prob_threshold",
    "spotiflow_min_distance",
    "spotiflow_local_refinement",
)


def _cache_value(value):
    """Return a stable, hashable representation, including unknown scales."""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _inference_cache_key(spec, settings: SegmentationSettings, scales: dict) -> tuple:
    """Describe every input that can affect one adapter invocation.

    The source CZYX array is intentionally absent because each cache instance is
    scoped to exactly one image timepoint.
    """
    setting_values = tuple(
        _cache_value(getattr(settings, name))
        for name in _INFERENCE_CACHE_SETTING_NAMES
    )
    scale_values = tuple(
        sorted((str(name), _cache_value(value)) for name, value in scales.items())
    )
    return (spec.id, setting_values, scale_values)


def _aggregate_timings(records: list[dict[str, float]]) -> dict[str, float]:
    keys = {
        "startup_seconds",
        "zarr_read_seconds",
        "import_seconds",
        "device_setup_seconds",
        "model_load_seconds",
        "inference_seconds",
    }
    detail_keys = {
        "spot_detection_seconds",
        "local_refinement_seconds",
    }
    keys.update(
        key for key in detail_keys if any(key in record for record in records)
    )
    return {
        key: (
            max((float(record.get(key, 0.0)) for record in records), default=0.0)
            if key == "startup_seconds"
            else sum(float(record.get(key, 0.0)) for record in records)
        )
        for key in sorted(keys)
    }


def _border_ids(labels: np.ndarray) -> set[int]:
    """Return nonzero labels touching an XY image edge (not a Z boundary)."""
    edges = np.concatenate(
        (
            labels[..., 0, :].ravel(),
            labels[..., -1, :].ravel(),
            labels[..., :, 0].ravel(),
            labels[..., :, -1].ravel(),
        )
    )
    return {int(value) for value in np.unique(edges) if value}


def _offset_labels(labels: np.ndarray, offset: int) -> tuple[np.ndarray, int]:
    labels = np.asarray(labels, dtype=np.uint32)
    if offset:
        labels = np.where(labels > 0, labels.astype(np.uint64) + offset, 0).astype(
            np.uint32
        )
    return labels, int(labels.max(initial=offset))


def _match_cells_and_nuclei(
    cells: np.ndarray,
    nuclei: np.ndarray,
    *,
    first_id: int = 1,
    remove_border_cells: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Match one nucleus to each cell and give all compartments the cell ID.

    A nucleus is first assigned to the cell with which it has the greatest
    overlap. If a cell contains multiple assigned nuclei, only the largest
    nucleus is retained. Cells without a nucleus are removed.
    """
    cells = np.asarray(cells, dtype=np.uint32).copy()
    nuclei = np.asarray(nuclei, dtype=np.uint32)
    if cells.shape != nuclei.shape:
        raise ValueError("Cell and nucleus label arrays must have the same shape")
    if remove_border_cells:
        touching = _border_ids(cells)
        if touching:
            cells[np.isin(cells, list(touching))] = 0

    overlap_mask = (cells > 0) & (nuclei > 0)
    nucleus_to_cell: dict[int, int] = {}
    if np.any(overlap_mask):
        pairs, counts = np.unique(
            np.column_stack((nuclei[overlap_mask], cells[overlap_mask])),
            axis=0,
            return_counts=True,
        )
        for nucleus_id in np.unique(pairs[:, 0]):
            candidates = np.flatnonzero(pairs[:, 0] == nucleus_id)
            best = candidates[np.argmax(counts[candidates])]
            nucleus_to_cell[int(nucleus_id)] = int(pairs[best, 1])

    nucleus_ids, nucleus_counts = np.unique(nuclei[nuclei > 0], return_counts=True)
    nucleus_sizes = {
        int(label): int(count) for label, count in zip(nucleus_ids, nucleus_counts)
    }
    cell_candidates: dict[int, list[int]] = {}
    for nucleus_id, cell_id in nucleus_to_cell.items():
        cell_candidates.setdefault(cell_id, []).append(nucleus_id)

    chosen = {
        cell_id: max(ids, key=lambda value: nucleus_sizes[value])
        for cell_id, ids in cell_candidates.items()
    }
    matched_cells = np.zeros_like(cells)
    matched_nuclei = np.zeros_like(nuclei)
    next_id = first_id - 1
    for cell_id in sorted(chosen):
        next_id += 1
        matched_cells[cells == cell_id] = next_id
        matched_nuclei[nuclei == chosen[cell_id]] = next_id
    cytoplasm = np.where(matched_nuclei > 0, 0, matched_cells).astype(np.uint32)
    return matched_cells, matched_nuclei, cytoplasm, next_id


def _expand_nuclei_to_cells(
    nuclei: np.ndarray,
    distance_um: float,
    scales: dict[str, float],
) -> np.ndarray:
    """Expand each nucleus to its nearest-label XY territory within a radius.

    This is the vectorized equivalent of the CellExpansion KD-tree approach.
    SciPy's exact Euclidean distance transform also handles anisotropic XY
    pixel sizes without constructing or querying a large coordinate tree.
    """
    from scipy.ndimage import distance_transform_edt

    nuclei = np.asarray(nuclei, dtype=np.uint32)
    cells = np.zeros_like(nuclei)
    y_scale = float(scales.get("y") or scales.get("x") or 1.0)
    x_scale = float(scales.get("x") or scales.get("y") or 1.0)
    for z_index, plane in enumerate(nuclei):
        if not np.any(plane):
            continue
        distances, nearest = distance_transform_edt(
            plane == 0,
            sampling=(y_scale, x_scale),
            return_indices=True,
        )
        expanded = plane[nearest[0], nearest[1]]
        expanded[distances > distance_um] = 0
        cells[z_index] = expanded
    return cells


def _step_settings(
    settings: SegmentationSettings,
    *,
    model: str,
    target: str,
    primary_channel: int,
    nuclei_channel: int = 0,
) -> SegmentationSettings:
    return replace(
        settings,
        model=model,
        target=target,
        primary_channel=primary_channel,
        nuclei_channel=nuclei_channel,
        benchmark=False,
    )


def _spot_model_dispatch(model_id: str):
    """Resolve a model allowed in the repeated spot/foci channel step."""
    spec = get_model_spec(model_id)
    if spec.family == "spotiflow" and "spots" in spec.targets:
        return spec, "spots", "spots"
    if (
        spec.family == "stardist"
        and spec.checkpoint.startswith("SD_Foci")
        and "foci" in spec.targets
    ):
        return spec, "foci", "foci"
    if (
        spec.family == "cellpose3"
        and "bact" in spec.checkpoint.lower()
        and "cells" in spec.targets
    ):
        return spec, "cells", "bacteria"
    raise ValueError(
        "Multi-step spot models must be Spotiflow, a StarDist SD_Foci model, "
        "or a Cellpose 3 model containing 'bact' in its checkpoint name"
    )


def _run_reported_segmentation(
    czyx: np.ndarray,
    spec,
    settings: SegmentationSettings,
    scales: dict[str, float],
    *,
    step: str,
    timepoint: int,
    log: Callable[[str], None] | None,
    result_cache: dict[tuple, tuple[np.ndarray, dict, str]],
) -> tuple[np.ndarray, dict, dict]:
    cache_key = _inference_cache_key(spec, settings, scales)
    cached = result_cache.get(cache_key)
    if cached is None:
        labels, info = segment_czyx(czyx, spec, settings, scales)
        info = dict(info)
        info["result_cache_hit"] = False
        info["reused_from_step"] = None
        result_cache[cache_key] = (labels.copy(), deepcopy(info), step)
    else:
        cached_labels, cached_info, source_step = cached
        labels = cached_labels.copy()
        info = deepcopy(cached_info)
        info["runtime_seconds"] = 0.0
        info["timings"] = {}
        info["model_cache_hit"] = False
        info["model_cache_hits"] = 0
        info["model_cache_misses"] = 0
        info["result_cache_hit"] = True
        info["reused_from_step"] = source_step
    record = step_record(
        step=step,
        timepoint=timepoint,
        model=spec.id,
        target=settings.target,
        primary_channel=settings.primary_channel,
        nuclei_channel=settings.nuclei_channel,
        labels=labels,
        info=info,
        scales=scales,
    )
    for line in format_step_record(record):
        emit(log, line)
    return labels, info, record


def _segment_multistep_image(
    image,
    settings: SegmentationSettings,
    *,
    startup_seconds: float = 0.0,
    zarr_read_seconds: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> LabelResult:
    settings.validate_steps()
    foci_steps = settings.enabled_foci_steps()
    for slot, _model, channel in foci_steps:
        if channel < 1 or channel > image.data.shape[1]:
            raise ValueError(
                f"Step 3{chr(96 + slot)} channel {channel} is outside input "
                f"channel count {image.data.shape[1]}"
            )
    per_time: list[np.ndarray] = []
    infos: list[dict] = []
    step_runs: list[dict] = []
    output_statistics: list[dict] = []
    channel_labels: list[str] = []
    next_id = 0
    for time_index in range(image.data.shape[0]):
        czyx = image.data[time_index]
        result_cache: dict[tuple, tuple[np.ndarray, dict, str]] = {}
        cell_labels = nucleus_labels = cell_step_nuclei = None
        if settings.cell_model != SKIP:
            expansion_model = settings.cell_expansion_model()
            if expansion_model is not None:
                expansion_settings = _step_settings(
                    settings,
                    model=expansion_model,
                    target="nuclei",
                    primary_channel=settings.cell_expansion_channel(),
                )
                cell_step_nuclei, info, record = _run_reported_segmentation(
                    czyx,
                    get_model_spec(expansion_model),
                    expansion_settings,
                    image.scales,
                    step="Step 1 expansion nuclei",
                    timepoint=time_index,
                    log=log,
                    result_cache=result_cache,
                )
                infos.append(info)
                step_runs.append(record)
                cell_labels = _expand_nuclei_to_cells(
                    cell_step_nuclei,
                    settings.cell_expansion_distance,
                    image.scales,
                )
            else:
                cell_settings = _step_settings(
                    settings,
                    model=settings.cell_model,
                    target="cells",
                    primary_channel=settings.cell_channel,
                    nuclei_channel=settings.cell_nuclei_channel,
                )
                cell_labels, info, record = _run_reported_segmentation(
                    czyx,
                    get_model_spec(settings.cell_model),
                    cell_settings,
                    image.scales,
                    step="Step 1 cells",
                    timepoint=time_index,
                    log=log,
                    result_cache=result_cache,
                )
                infos.append(info)
                step_runs.append(record)
        if settings.nucleus_model != SKIP:
            nucleus_settings = _step_settings(
                settings,
                model=settings.nucleus_model,
                target="nuclei",
                primary_channel=settings.nucleus_channel,
            )
            nucleus_labels, info, record = _run_reported_segmentation(
                czyx,
                get_model_spec(settings.nucleus_model),
                nucleus_settings,
                image.scales,
                step="Step 2 nuclei",
                timepoint=time_index,
                log=log,
                result_cache=result_cache,
            )
            infos.append(info)
            step_runs.append(record)

        time_channels: list[np.ndarray] = []
        time_channel_labels: list[str] = []
        matching_nuclei = (
            nucleus_labels if nucleus_labels is not None else cell_step_nuclei
        )
        if cell_labels is not None and matching_nuclei is not None:
            cells, nuclei, cytoplasm, next_id = _match_cells_and_nuclei(
                cell_labels,
                matching_nuclei,
                first_id=next_id + 1,
                remove_border_cells=settings.remove_border_cells,
            )
            time_channels.extend((cells, nuclei, cytoplasm))
            time_channel_labels.extend(
                ("labels_cells", "labels_nuclei", "labels_cytoplasm")
            )
        elif cell_labels is not None:
            if settings.remove_border_cells:
                touching = _border_ids(cell_labels)
                if touching:
                    cell_labels = np.where(
                        np.isin(cell_labels, list(touching)), 0, cell_labels
                    )
            cell_labels, next_id = _offset_labels(cell_labels, next_id)
            time_channels.append(cell_labels)
            time_channel_labels.append("labels_cells")
        elif nucleus_labels is not None:
            nucleus_labels, next_id = _offset_labels(nucleus_labels, next_id)
            time_channels.append(nucleus_labels)
            time_channel_labels.append("labels_nuclei")

        for slot, model_id, channel in foci_steps:
            spot_spec, spot_target, spot_label = _spot_model_dispatch(model_id)
            spot_settings = _step_settings(
                settings,
                model=model_id,
                target=spot_target,
                primary_channel=channel,
            )
            if spot_label == "bacteria" and settings.diameter == 0:
                spot_settings = replace(spot_settings, diameter=-1.0)
            spots, info, record = _run_reported_segmentation(
                czyx,
                spot_spec,
                spot_settings,
                image.scales,
                step=f"Step 3{chr(96 + slot)} {spot_label}",
                timepoint=time_index,
                log=log,
                result_cache=result_cache,
            )
            infos.append(info)
            step_runs.append(record)
            spots, next_id = _offset_labels(spots, next_id)
            time_channels.append(spots)
            time_channel_labels.append(f"labels_{spot_label}_channel_{channel}")
        if not time_channels:
            raise ValueError("Multi-step mode produced no output channels")
        emit(log, f"  Post-processing | T{time_index + 1}")
        for name, labels in zip(time_channel_labels, time_channels):
            statistics = label_statistics(labels, image.scales)
            output_statistics.append(
                {
                    "timepoint": time_index,
                    "channel": name,
                    "locations_only": bool(
                        name.startswith("labels_spots_channel_")
                        and not settings.spotiflow_local_refinement
                    ),
                    "label_statistics": statistics,
                }
            )
            emit(
                log,
                f"    {name}: "
                + format_label_statistics(
                    statistics,
                    locations_only=(
                        name.startswith("labels_spots_channel_")
                        and not settings.spotiflow_local_refinement
                    ),
                ),
            )
        if not channel_labels:
            channel_labels = time_channel_labels
        per_time.append(np.stack(time_channels, axis=0))

    tczyx = np.stack(per_time, axis=0)
    timing_records = [dict(info.get("timings", {})) for info in infos]
    timing_records.append(
        {
            "startup_seconds": startup_seconds,
            "zarr_read_seconds": zarr_read_seconds,
        }
    )
    timings = _aggregate_timings(timing_records)
    unique_ids = np.unique(tczyx)
    return LabelResult(
        tczyx,
        image,
        "multi-step",
        "multi-step",
        {
            "device": infos[-1].get("device") if infos else None,
            "runtime_seconds": sum(
                float(info.get("runtime_seconds", 0.0)) for info in infos
            ),
            "object_count": int(np.count_nonzero(unique_ids)),
            "model_cache_hits": sum(
                0
                if bool(info.get("result_cache_hit"))
                else int(
                    info.get("model_cache_hits", bool(info.get("model_cache_hit")))
                )
                for info in infos
            ),
            "model_cache_misses": sum(
                0
                if bool(info.get("result_cache_hit"))
                else int(
                    info.get(
                        "model_cache_misses", not bool(info.get("model_cache_hit"))
                    )
                )
                for info in infos
            ),
            "result_cache_hits": sum(
                bool(info.get("result_cache_hit")) for info in infos
            ),
            "timings": timings,
            "step_runs": step_runs,
            "output_statistics": output_statistics,
            "parameters": settings.to_dict(),
            "shared_instance_ids": ["cells", "nuclei", "cytoplasm"]
            if settings.cell_model != SKIP
            and (
                settings.nucleus_model != SKIP
                or settings.cell_expansion_model() is not None
            )
            else [],
        },
        channel_labels=channel_labels,
        include_original_channels=settings.include_original_channels,
        write_ome_zarr_labels=settings.write_ome_zarr_labels,
    )
def _segment_image(
    image,
    settings: SegmentationSettings,
    *,
    startup_seconds: float = 0.0,
    zarr_read_seconds: float = 0.0,
) -> LabelResult:
    spec = get_model_spec(settings.model)
    per_time = []
    infos = []
    offset = 0
    for time_index in range(image.data.shape[0]):
        labels, info = segment_czyx(
            image.data[time_index], spec, settings, image.scales
        )
        if offset:
            labels = np.where(labels > 0, labels.astype(np.uint64) + offset, 0).astype(
                np.uint32
            )
        offset = int(labels.max(initial=offset))
        per_time.append(labels[:, None])  # Z,C,Y,X
        infos.append(info)
    tczyx = np.stack(per_time, axis=0).transpose(0, 2, 1, 3, 4)
    timing_records = [dict(info.get("timings", {})) for info in infos]
    timing_records.append(
        {
            "startup_seconds": startup_seconds,
            "zarr_read_seconds": zarr_read_seconds,
        }
    )
    timings = _aggregate_timings(timing_records)
    latest = infos[-1] if infos else {}
    return LabelResult(
        tczyx,
        image,
        spec.id,
        settings.target,
        {
            "device": latest.get("device"),
            "dimension_mode": latest.get("dimension_mode"),
            "runtime_seconds": sum(
                float(info.get("runtime_seconds", 0.0)) for info in infos
            ),
            "object_count": int(tczyx.max(initial=0)),
            "model_cache_hits": sum(
                int(info.get("model_cache_hits", bool(info.get("model_cache_hit"))))
                for info in infos
            ),
            "model_cache_misses": sum(
                int(
                    info.get(
                        "model_cache_misses", not bool(info.get("model_cache_hit"))
                    )
                )
                for info in infos
            ),
            "result_cache_hits": 0,
            "timings": timings,
            "parameters": settings.to_dict(),
        },
        write_ome_zarr_labels=settings.write_ome_zarr_labels,
    )


def run_workflow(
    input_dir: str | Path,
    output_dir: str | Path,
    settings: SegmentationSettings,
    *,
    startup_seconds: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings.validate_steps()
    for line in workflow_report_lines(settings):
        emit(log, line)
    stores = discover_ome_zarrs(input_dir)
    if not stores:
        raise FileNotFoundError(f"No top-level NGFF .zarr inputs found in {input_dir}")
    outputs: list[Path] = []
    emit(log, f"Discovered {len(stores)} top-level OME-Zarr input(s).")
    startup_remaining = float(startup_seconds)
    if settings.benchmark:
        first_resource = enumerate_resources(stores[0])[0]
        read_started = time.perf_counter()
        image = read_image(first_resource)
        read_seconds = time.perf_counter() - read_started
        for line in input_report_lines(image, read_seconds):
            emit(log, line)
        gallery, failed = run_benchmark(
            image,
            settings,
            output_dir,
            base_timings={
                "startup_seconds": startup_remaining,
                "zarr_read_seconds": read_seconds,
            },
            log=log,
        )
        if failed:
            raise RuntimeError(
                f"One or more eligible benchmark models failed; gallery retained at {gallery}"
            )
        return [gallery]
    model_name = "multistep"
    for store in stores:
        resources = enumerate_resources(store)
        results = []
        for resource in resources:
            read_started = time.perf_counter()
            image = read_image(resource)
            read_seconds = time.perf_counter() - read_started
            for line in input_report_lines(image, read_seconds):
                emit(log, line)
            results.append(
                _segment_multistep_image(
                    image,
                    settings,
                    startup_seconds=startup_remaining,
                    zarr_read_seconds=read_seconds,
                    log=log,
                )
            )
            startup_remaining = 0.0
        source_name = store.name.removesuffix(".ome.zarr").removesuffix(".zarr")
        output_path = (
            output_dir
            / (
                f"{source_name}_{model_name}.ome.zarr"
            )
        )
        if resources[0].plate_path is None:
            emit(log, f"Writing output: {output_path}")
            write_label_image(results[0], output_path)
        else:
            emit(log, f"Writing HCS output: {output_path}")
            write_hcs_plate(results, output_path)
        emit(log, f"Finished output: {output_path}")
        outputs.append(output_path)
        if settings.measurements_database != "skip":
            from .measurements import (
                measurement_database_path,
                write_measurements_database,
            )

            database_path = measurement_database_path(
                output_dir, source_name, settings.measurements_database
            )
            emit(
                log,
                f"Writing measurements database ({settings.measurements_database}): "
                f"{database_path}",
            )
            measurement_summary = write_measurements_database(
                results,
                database_path,
                settings.measurements_database,
                output_ome_zarr=output_path,
                log=log,
            )
            emit(
                log,
                "Finished measurements: "
                f"images={measurement_summary['images']}, "
                f"label sets={measurement_summary['label_sets']}, "
                f"objects={measurement_summary['objects']}, "
                f"intensity rows={measurement_summary['intensities']}, "
                f"relationships={measurement_summary['relationships']}, "
                f"runtime={measurement_summary['runtime_seconds']:.2f}s",
            )
            outputs.append(database_path)
    return outputs
