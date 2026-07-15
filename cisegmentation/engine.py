from __future__ import annotations

from pathlib import Path

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


def _segment_image(image, settings: SegmentationSettings) -> LabelResult:
    spec = get_model_spec(settings.model)
    per_time = []
    provenance = None
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
        provenance = info
    tczyx = np.stack(per_time, axis=0).transpose(0, 2, 1, 3, 4)
    return LabelResult(
        tczyx,
        image,
        spec.id,
        settings.target,
        {**(provenance or {}), "parameters": settings.to_dict()},
    )


def run_workflow(
    input_dir: str | Path, output_dir: str | Path, settings: SegmentationSettings
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stores = discover_ome_zarrs(input_dir)
    if not stores:
        raise FileNotFoundError(f"No top-level NGFF .zarr inputs found in {input_dir}")
    outputs: list[Path] = []
    if settings.benchmark:
        first_resource = enumerate_resources(stores[0])[0]
        gallery, failed = run_benchmark(
            read_image(first_resource), settings, output_dir
        )
        if failed:
            raise RuntimeError(
                f"One or more eligible benchmark models failed; gallery retained at {gallery}"
            )
        return [gallery]
    model_name = _safe_model_name(settings.model)
    for store in stores:
        resources = enumerate_resources(store)
        results = [
            _segment_image(read_image(resource), settings) for resource in resources
        ]
        source_name = store.name.removesuffix(".ome.zarr").removesuffix(".zarr")
        output_path = (
            output_dir
            / f"{source_name}_{model_name}_{settings.target}.ome.zarr"
        )
        if resources[0].plate_path is None:
            write_label_image(results[0], output_path)
        else:
            write_hcs_plate(results, output_path)
        outputs.append(output_path)
    return outputs
