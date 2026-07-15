from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .registry import ModelSpec
from .settings import SegmentationSettings


def resolve_device(requested: str) -> str:
    requested = str(requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _selected_image(czyx: np.ndarray, settings: SegmentationSettings) -> np.ndarray:
    channels = settings.selected_channels(czyx.shape[0])
    selected = czyx[channels]
    return np.moveaxis(selected, 0, -1)  # Z,Y,X,C


def _unique_plane_labels(planes: list[np.ndarray]) -> np.ndarray:
    output: list[np.ndarray] = []
    offset = 0
    for plane in planes:
        labels = np.asarray(plane, dtype=np.uint32)
        if offset:
            labels = np.where(labels > 0, labels.astype(np.uint64) + offset, 0).astype(
                np.uint32
            )
        offset = int(labels.max(initial=offset))
        output.append(labels)
    return np.stack(output, axis=0)


def points_to_labels(points: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.uint32)
    for label, point in enumerate(np.asarray(points), start=1):
        coords = tuple(int(round(float(value))) for value in point[: len(shape)])
        coords = tuple(
            min(max(value, 0), size - 1) for value, size in zip(coords, shape)
        )
        if labels[coords] == 0:
            labels[coords] = label
        else:
            # Min-distance normally avoids collisions; find the first empty neighbor if rounding collides.
            placed = False
            for axis in range(len(shape)):
                for delta in (-1, 1):
                    neighbor = list(coords)
                    neighbor[axis] = min(
                        max(neighbor[axis] + delta, 0), shape[axis] - 1
                    )
                    neighbor_tuple = tuple(neighbor)
                    if labels[neighbor_tuple] == 0:
                        labels[neighbor_tuple] = label
                        placed = True
                        break
                if placed:
                    break
    return labels


def _segment_cellpose(
    czyx: np.ndarray, spec: ModelSpec, settings: SegmentationSettings, device: str
) -> np.ndarray:
    image = _selected_image(czyx, settings)
    gpu = device == "cuda"
    if spec.family == "cellpose3":
        from cellpose3_legacy import models
    else:
        from cellpose import models
    model = models.CellposeModel(gpu=gpu, pretrained_model=spec.checkpoint)
    kwargs: dict[str, Any] = {
        "diameter": settings.diameter or None,
        "flow_threshold": settings.flow_threshold,
        "cellprob_threshold": settings.cellprob_threshold,
        "channel_axis": -1,
    }
    native_3d = (
        image.shape[0] > 1
        and spec.dimensions == "3d"
        and settings.dimension_mode != "slice-2d"
    )
    if native_3d:
        masks, *_ = model.eval(image, z_axis=0, do_3D=True, **kwargs)
        return np.asarray(masks, dtype=np.uint32)
    planes = []
    for z_index in range(image.shape[0]):
        masks, *_ = model.eval(image[z_index], **kwargs)
        planes.append(np.asarray(masks))
    return _unique_plane_labels(planes)


def _stardist_model_path(checkpoint: str) -> Path:
    root = Path(
        os.environ.get(
            "CISEGMENTATION_MODELS", Path.home() / ".cisegmentation" / "models"
        )
    )
    return root / "stardist" / checkpoint


def _predict_stardist_tiled(
    model, image: np.ndarray, prob: float | None, nms: float | None
) -> np.ndarray:
    tile_size, halo = 1024, 64
    height, width = image.shape[-2:]
    if height <= tile_size and width <= tile_size:
        labels, _ = model.predict_instances(
            image, prob_thresh=prob, nms_thresh=nms, normalize=True
        )
        return np.asarray(labels, dtype=np.uint32)
    output = np.zeros((height, width), dtype=np.uint32)
    offset = 0
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1, x1 = min(y0 + tile_size, height), min(x0 + tile_size, width)
            ry0, rx0 = max(0, y0 - halo), max(0, x0 - halo)
            ry1, rx1 = min(height, y1 + halo), min(width, x1 + halo)
            labels, _ = model.predict_instances(
                image[ry0:ry1, rx0:rx1],
                prob_thresh=prob,
                nms_thresh=nms,
                normalize=True,
            )
            core = np.asarray(labels)[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0]
            if offset:
                core = np.where(core > 0, core.astype(np.uint64) + offset, 0)
            output[y0:y1, x0:x1] = core.astype(np.uint32)
            offset = int(output.max(initial=offset))
    return output


def _segment_stardist(
    czyx: np.ndarray, spec: ModelSpec, settings: SegmentationSettings, device: str
) -> np.ndarray:
    from cistardist_pytorch import StarDist2D

    channel = settings.selected_channels(czyx.shape[0])[0]
    model = StarDist2D.from_folder(_stardist_model_path(spec.checkpoint), device=device)
    prob = (
        None
        if settings.stardist_prob_threshold < 0
        else settings.stardist_prob_threshold
    )
    nms = (
        None if settings.stardist_nms_threshold < 0 else settings.stardist_nms_threshold
    )
    return _unique_plane_labels(
        [
            _predict_stardist_tiled(model, czyx[channel, z], prob, nms)
            for z in range(czyx.shape[1])
        ]
    )


def _segment_instanseg(
    czyx: np.ndarray,
    spec: ModelSpec,
    settings: SegmentationSettings,
    pixel_size_um: float,
) -> np.ndarray:
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            "InstanSeg requires a positive XY pixel size in OME-Zarr metadata"
        )
    from instanseg import InstanSeg

    channels = settings.selected_channels(czyx.shape[0])
    if len(channels) < spec.min_channels:
        raise ValueError(
            f"{spec.id} requires at least {spec.min_channels} input channels"
        )
    model_root = (
        Path(
            os.environ.get(
                "CISEGMENTATION_MODELS", Path.home() / ".cisegmentation" / "models"
            )
        )
        / "instanseg"
    )
    try:
        model = InstanSeg(str(model_root / spec.checkpoint), verbosity=0)
    except Exception:
        model = InstanSeg(spec.checkpoint, verbosity=0)
    target_index = (
        1
        if settings.target == "cells"
        and spec.checkpoint == "fluorescence_nuclei_and_cells"
        else 0
    )
    planes = []
    for z_index in range(czyx.shape[1]):
        image = czyx[channels, z_index]
        labels, _ = model.eval_small_image(image, pixel_size_um)
        array = np.asarray(labels)
        while array.ndim > 2:
            if array.shape[0] > target_index:
                array = array[target_index]
            else:
                array = np.squeeze(array, axis=0)
        planes.append(array)
    return _unique_plane_labels(planes)


def _segment_spotiflow(
    czyx: np.ndarray, spec: ModelSpec, settings: SegmentationSettings, device: str
) -> np.ndarray:
    from spotiflow.model import Spotiflow

    cache = Path(
        os.environ.get(
            "SPOTIFLOW_CACHE_DIR",
            Path(
                os.environ.get(
                    "CISEGMENTATION_MODELS", Path.home() / ".cisegmentation" / "models"
                )
            )
            / "spotiflow",
        )
    )
    model = Spotiflow.from_pretrained(
        spec.checkpoint, cache_dir=cache, map_location=device, verbose=False
    )
    channel = settings.selected_channels(czyx.shape[0])[0]
    volume = czyx[channel]
    threshold = (
        None
        if settings.spotiflow_prob_threshold < 0
        else settings.spotiflow_prob_threshold
    )
    native_3d = (
        volume.shape[0] > 1
        and spec.dimensions == "3d"
        and settings.dimension_mode != "slice-2d"
    )
    if native_3d:
        points, _ = model.predict(
            volume,
            prob_thresh=threshold,
            min_distance=settings.spotiflow_min_distance,
            device=device,
            verbose=False,
        )
        return points_to_labels(points, volume.shape)
    return _unique_plane_labels(
        [
            points_to_labels(
                model.predict(
                    volume[z],
                    prob_thresh=threshold,
                    min_distance=settings.spotiflow_min_distance,
                    device=device,
                    verbose=False,
                )[0],
                volume[z].shape,
            )
            for z in range(volume.shape[0])
        ]
    )


def segment_czyx(
    czyx: np.ndarray,
    spec: ModelSpec,
    settings: SegmentationSettings,
    scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    if settings.target not in spec.targets:
        raise ValueError(
            f"Model {spec.id} does not support target {settings.target!r}; supported: {', '.join(spec.targets)}"
        )
    settings.selected_channels(czyx.shape[0])
    device = resolve_device(settings.device)
    start = time.perf_counter()
    if spec.family in {"cellpose3", "cellpose-sam"}:
        labels = _segment_cellpose(czyx, spec, settings, device)
    elif spec.family == "stardist":
        labels = _segment_stardist(czyx, spec, settings, device)
    elif spec.family == "instanseg":
        labels = _segment_instanseg(czyx, spec, settings, scales.get("x", float("nan")))
    elif spec.family == "spotiflow":
        labels = _segment_spotiflow(czyx, spec, settings, device)
    else:
        raise ValueError(f"No adapter for model family {spec.family}")
    elapsed = time.perf_counter() - start
    return np.asarray(labels, dtype=np.uint32), {
        "device": device,
        "runtime_seconds": elapsed,
        "object_count": int(np.max(labels, initial=0)),
        "dimension_mode": "native-3d"
        if spec.dimensions == "3d"
        and labels.shape[0] > 1
        and settings.dimension_mode != "slice-2d"
        else "slice-2d",
    }
