from __future__ import annotations

from pathlib import Path
import time
from dataclasses import replace

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
from .settings import SegmentationSettings


def _safe_model_name(model_id: str) -> str:
    return model_id.replace(":", "_").replace("/", "_")


def _aggregate_timings(records: list[dict[str, float]]) -> dict[str, float]:
    keys = {
        "startup_seconds",
        "zarr_read_seconds",
        "import_seconds",
        "device_setup_seconds",
        "model_load_seconds",
        "inference_seconds",
    }
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
        multi_step=False,
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


def _segment_multistep_image(
    image,
    settings: SegmentationSettings,
    *,
    startup_seconds: float = 0.0,
    zarr_read_seconds: float = 0.0,
) -> LabelResult:
    if not any((settings.cell_step, settings.nucleus_step, settings.spot_step)):
        raise ValueError("Multi-step mode requires at least one enabled step")
    spot_channels = (
        settings.selected_spot_channels(image.data.shape[1])
        if settings.spot_step
        else []
    )
    spot_spec = spot_target = spot_label = None
    if settings.spot_step:
        spot_spec, spot_target, spot_label = _spot_model_dispatch(settings.spot_model)
    per_time: list[np.ndarray] = []
    infos: list[dict] = []
    channel_labels: list[str] = []
    next_id = 0
    for time_index in range(image.data.shape[0]):
        czyx = image.data[time_index]
        cell_labels = nucleus_labels = None
        if settings.cell_step:
            cell_settings = _step_settings(
                settings,
                model=settings.cell_model,
                target="cells",
                primary_channel=settings.cell_channel,
                nuclei_channel=settings.cell_nuclei_channel,
            )
            cell_labels, info = segment_czyx(
                czyx, get_model_spec(settings.cell_model), cell_settings, image.scales
            )
            infos.append(info)
        if settings.nucleus_step:
            nucleus_settings = _step_settings(
                settings,
                model=settings.nucleus_model,
                target="nuclei",
                primary_channel=settings.nucleus_channel,
            )
            nucleus_labels, info = segment_czyx(
                czyx,
                get_model_spec(settings.nucleus_model),
                nucleus_settings,
                image.scales,
            )
            infos.append(info)

        time_channels: list[np.ndarray] = []
        time_channel_labels: list[str] = []
        if cell_labels is not None and nucleus_labels is not None:
            cells, nuclei, cytoplasm, next_id = _match_cells_and_nuclei(
                cell_labels,
                nucleus_labels,
                first_id=next_id + 1,
                remove_border_cells=settings.remove_border_cells,
            )
            time_channels.extend((cells, nuclei))
            time_channel_labels.extend(("cells", "nuclei"))
            if settings.derive_cytoplasm:
                time_channels.append(cytoplasm)
                time_channel_labels.append("cytoplasm")
        elif cell_labels is not None:
            if settings.remove_border_cells:
                touching = _border_ids(cell_labels)
                if touching:
                    cell_labels = np.where(
                        np.isin(cell_labels, list(touching)), 0, cell_labels
                    )
            cell_labels, next_id = _offset_labels(cell_labels, next_id)
            time_channels.append(cell_labels)
            time_channel_labels.append("cells")
        elif nucleus_labels is not None:
            nucleus_labels, next_id = _offset_labels(nucleus_labels, next_id)
            time_channels.append(nucleus_labels)
            time_channel_labels.append("nuclei")

        for occurrence, channel in enumerate(spot_channels, start=1):
            spot_settings = _step_settings(
                settings,
                model=settings.spot_model,
                target=spot_target,
                primary_channel=channel + 1,
            )
            if spot_label == "bacteria" and settings.diameter == 0:
                spot_settings = replace(spot_settings, diameter=-1.0)
            spots, info = segment_czyx(
                czyx, spot_spec, spot_settings, image.scales
            )
            infos.append(info)
            spots, next_id = _offset_labels(spots, next_id)
            time_channels.append(spots)
            time_channel_labels.append(
                f"{spot_label} channel {channel + 1} ({occurrence})"
            )
        if not time_channels:
            raise ValueError("Multi-step mode produced no output channels")
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
            "model_cache_hits": sum(bool(info.get("model_cache_hit")) for info in infos),
            "model_cache_misses": sum(
                not bool(info.get("model_cache_hit")) for info in infos
            ),
            "timings": timings,
            "parameters": settings.to_dict(),
            "shared_instance_ids": ["cells", "nuclei", "cytoplasm"]
            if settings.cell_step and settings.nucleus_step
            else [],
        },
        channel_labels=channel_labels,
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
                bool(info.get("model_cache_hit")) for info in infos
            ),
            "model_cache_misses": sum(
                not bool(info.get("model_cache_hit")) for info in infos
            ),
            "timings": timings,
            "parameters": settings.to_dict(),
        },
    )


def run_workflow(
    input_dir: str | Path,
    output_dir: str | Path,
    settings: SegmentationSettings,
    *,
    startup_seconds: float = 0.0,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stores = discover_ome_zarrs(input_dir)
    if not stores:
        raise FileNotFoundError(f"No top-level NGFF .zarr inputs found in {input_dir}")
    outputs: list[Path] = []
    startup_remaining = float(startup_seconds)
    if settings.benchmark:
        first_resource = enumerate_resources(stores[0])[0]
        read_started = time.perf_counter()
        image = read_image(first_resource)
        read_seconds = time.perf_counter() - read_started
        gallery, failed = run_benchmark(
            image,
            settings,
            output_dir,
            base_timings={
                "startup_seconds": startup_remaining,
                "zarr_read_seconds": read_seconds,
            },
        )
        if failed:
            raise RuntimeError(
                f"One or more eligible benchmark models failed; gallery retained at {gallery}"
            )
        return [gallery]
    model_name = "multistep" if settings.multi_step else _safe_model_name(settings.model)
    for store in stores:
        resources = enumerate_resources(store)
        results = []
        for resource in resources:
            read_started = time.perf_counter()
            image = read_image(resource)
            read_seconds = time.perf_counter() - read_started
            segmenter = _segment_multistep_image if settings.multi_step else _segment_image
            results.append(
                segmenter(
                    image,
                    settings,
                    startup_seconds=startup_remaining,
                    zarr_read_seconds=read_seconds,
                )
            )
            startup_remaining = 0.0
        source_name = store.name.removesuffix(".ome.zarr").removesuffix(".zarr")
        output_path = (
            output_dir
            / (
                f"{source_name}_{model_name}.ome.zarr"
                if settings.multi_step
                else f"{source_name}_{model_name}_{settings.target}.ome.zarr"
            )
        )
        if resources[0].plate_path is None:
            write_label_image(results[0], output_path)
        else:
            write_hcs_plate(results, output_path)
        outputs.append(output_path)
    return outputs
