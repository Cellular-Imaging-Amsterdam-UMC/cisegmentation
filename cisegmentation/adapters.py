from __future__ import annotations

import os
from pathlib import Path
import logging
import sys
import time
from typing import Any, Callable

import numpy as np

from .registry import ModelSpec
from .settings import SegmentationSettings


_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_TRITON_FLOP_WARNING = "triton not found; flop counting will not work for triton kernels"
_SPOT_REFINEMENT_MAX_RADIUS_UM = 1.0
_SPOT_REFINEMENT_THRESHOLD_FRACTION = 0.30
_SPOT_REFINEMENT_NOISE_SIGMAS = 3.0


class _TritonFlopWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _TRITON_FLOP_WARNING not in record.getMessage()


_TRITON_FLOP_WARNING_FILTER = _TritonFlopWarningFilter()


def _configure_torch_runtime():
    """Apply explicit PyTorch runtime choices used by every model adapter."""
    flop_logger = logging.getLogger("torch.utils.flop_counter")
    if _TRITON_FLOP_WARNING_FILTER not in flop_logger.filters:
        flop_logger.addFilter(_TRITON_FLOP_WARNING_FILTER)

    import torch

    # Cellpose creates sparse tensors without requesting invariant validation.
    # Explicit opt-out preserves that fast path and prevents PyTorch from
    # warning that the implicit default may change.
    torch.sparse.check_sparse_tensor_invariants.disable()
    return torch


def clear_model_cache() -> None:
    """Clear process-local models, primarily for tests and controlled teardown."""
    _MODEL_CACHE.clear()


def _cached_model(
    model_id: str,
    device: str,
    importer: Callable[[], Any],
    constructor: Callable[[Any], Any],
) -> tuple[Any, dict[str, Any]]:
    """Load a model once per stable model ID and device in this process."""
    key = (model_id, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], {
            "model_cache_hit": True,
            "import_seconds": 0.0,
            "model_load_seconds": 0.0,
        }
    started = time.perf_counter()
    imported = importer()
    import_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = constructor(imported)
    model_load_seconds = time.perf_counter() - started
    _MODEL_CACHE[key] = model
    return model, {
        "model_cache_hit": False,
        "import_seconds": import_seconds,
        "model_load_seconds": model_load_seconds,
    }


def _models_root() -> Path:
    configured = os.environ.get("CISEGMENTATION_MODELS")
    if configured:
        return Path(configured)
    repository_cache = Path(__file__).resolve().parents[1] / "models"
    if repository_cache.exists():
        return repository_cache
    return Path.home() / ".cisegmentation" / "models"


def _configure_model_cache() -> Path:
    root = _models_root()
    os.environ.setdefault("CISEGMENTATION_MODELS", str(root))
    os.environ.setdefault("CELLPOSE3_LEGACY_LOCAL_MODELS_PATH", str(root / "cellpose3"))
    os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(root / "cellpose-sam"))
    os.environ.setdefault("SPOTIFLOW_CACHE_DIR", str(root / "spotiflow"))
    os.environ.setdefault("SPOTIFLOW_LOCAL_MODELS_PATH", str(root / "spotiflow"))
    return root


def resolve_device(requested: str) -> str:
    requested = str(requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cpu":
        return "cpu"
    torch = _configure_torch_runtime()

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _selected_image(czyx: np.ndarray, settings: SegmentationSettings) -> np.ndarray:
    channels = settings.selected_channels(czyx.shape[0])
    selected = czyx[channels]
    return np.moveaxis(selected, 0, -1)  # Z,Y,X,C


def _xy_pixel_sizes_um(
    scales: dict[str, float], parameter: str
) -> tuple[float, float]:
    y_um = float(scales.get("y", float("nan")))
    x_um = float(scales.get("x", float("nan")))
    if not np.isfinite(y_um) or y_um <= 0 or not np.isfinite(x_um) or x_um <= 0:
        raise ValueError(
            f"{parameter} requires positive XY pixel sizes in OME-Zarr metadata"
        )
    return y_um, x_um


def _cellpose_diameter_pixels(
    diameter_um: float, target: str, scales: dict[str, float]
) -> float | None:
    if diameter_um < 0:
        return None
    resolved_um = diameter_um or (12.0 if target == "nuclei" else 25.0)
    y_um, x_um = _xy_pixel_sizes_um(scales, "Cellpose diameter")
    return resolved_um / ((y_um + x_um) / 2.0)


def _spotiflow_min_distance_pixels(
    distance_um: float, scales: dict[str, float]
) -> int:
    y_um, x_um = _xy_pixel_sizes_um(scales, "Spotiflow minimum distance")
    return max(1, int(round(float(distance_um) / ((y_um + x_um) / 2.0))))


def _resize_2d(
    array: np.ndarray, shape: tuple[int, int], *, labels: bool
) -> np.ndarray:
    if array.shape == shape:
        return np.asarray(array)
    import torch
    import torch.nn.functional as functional

    source = np.asarray(array)
    if not source.dtype.isnative:
        source = source.astype(source.dtype.newbyteorder("="), copy=False)
    tensor = torch.as_tensor(source, dtype=torch.float32)[None, None]
    kwargs = {"size": shape, "mode": "nearest" if labels else "bilinear"}
    if not labels:
        kwargs.update({"align_corners": False, "antialias": True})
    resized = functional.interpolate(tensor, **kwargs)[0, 0].cpu().numpy()
    return resized.astype(np.uint32 if labels else np.float32, copy=False)


def _stardist_versatile_input(
    image: np.ndarray, scales: dict[str, float]
) -> tuple[np.ndarray, tuple[int, int]]:
    y_um, x_um = _xy_pixel_sizes_um(scales, "StarDist versatile rescaling")
    original_shape = tuple(image.shape[-2:])
    factors = (min(1.0, y_um / 0.5), min(1.0, x_um / 0.5))
    target_shape = tuple(
        max(1, int(round(size * factor)))
        for size, factor in zip(original_shape, factors)
    )
    return _resize_2d(image, target_shape, labels=False), original_shape


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


def _smooth_spot_plane(image: np.ndarray) -> np.ndarray:
    """Apply a small dependency-free Gaussian-like filter to one image plane."""
    array = np.asarray(image, dtype=np.float32)
    finite = array[np.isfinite(array)]
    fill = float(np.median(finite)) if finite.size else 0.0
    clean = np.nan_to_num(array, nan=fill, posinf=fill, neginf=fill)
    padded = np.pad(clean, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + 2 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, :-2]
        + 4 * padded[1:-1, 1:-1]
        + 2 * padded[1:-1, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 16.0


def _connected_seed_component(binary: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """Return the 8-connected foreground component containing ``seed``."""
    output = np.zeros(binary.shape, dtype=bool)
    if not binary[seed]:
        return output
    height, width = binary.shape
    stack = [seed]
    output[seed] = True
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and binary[ny, nx]
                    and not output[ny, nx]
                ):
                    output[ny, nx] = True
                    stack.append((ny, nx))
    return output


def _local_spot_candidate(
    smoothed: np.ndarray,
    point_yx: np.ndarray,
    radius_y: float,
    radius_x: float,
) -> dict[str, Any]:
    """Grow one seed inside a physical ellipse using local background statistics."""
    height, width = smoothed.shape
    seed_y = min(max(int(round(float(point_yx[0]))), 0), height - 1)
    seed_x = min(max(int(round(float(point_yx[1]))), 0), width - 1)
    outer_y, outer_x = max(2, int(np.ceil(radius_y * 1.5))), max(
        2, int(np.ceil(radius_x * 1.5))
    )
    y0, y1 = max(0, seed_y - outer_y), min(height, seed_y + outer_y + 1)
    x0, x1 = max(0, seed_x - outer_x), min(width, seed_x + outer_x + 1)
    patch = smoothed[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = ((yy - seed_y) / radius_y) ** 2 + ((xx - seed_x) / radius_x) ** 2
    allowed = distance <= 1.0
    annulus = (distance >= 1.0) & (distance <= 2.25)
    background_values = patch[annulus]
    if not background_values.size:
        background_values = patch[~allowed]
    if not background_values.size:
        background_values = patch.reshape(-1)
    background = float(np.median(background_values))
    noise = float(1.4826 * np.median(np.abs(background_values - background)))
    peak_y0, peak_y1 = max(0, seed_y - 1), min(height, seed_y + 2)
    peak_x0, peak_x1 = max(0, seed_x - 1), min(width, seed_x + 2)
    peak = float(np.max(smoothed[peak_y0:peak_y1, peak_x0:peak_x1]))
    contrast = max(0.0, peak - background)
    threshold = max(
        background + _SPOT_REFINEMENT_NOISE_SIGMAS * noise,
        background + _SPOT_REFINEMENT_THRESHOLD_FRACTION * contrast,
    )
    binary = allowed & (patch >= threshold)
    local_seed = (seed_y - y0, seed_x - x0)
    component = _connected_seed_component(binary, local_seed)
    fallback = not np.any(component)
    if fallback:
        component[local_seed] = True
    return {
        "mask": component,
        "bbox": (y0, x0, y1, x1),
        "seed_yx": (seed_y, seed_x),
        "score": contrast / max(noise, 1e-6),
        "fallback": fallback,
    }


def _merge_local_spot_candidates(
    candidates: list[dict[str, Any]], shape: tuple[int, int]
) -> tuple[np.ndarray, dict[str, int]]:
    """Assign overlapping candidates by local signal-to-noise score."""
    labels = np.zeros(shape, dtype=np.uint32)
    overlap_pixels = 0
    suppressed = 0
    grown = 0
    fallbacks = 0
    next_label = 0
    for candidate in sorted(candidates, key=lambda item: -float(item["score"])):
        seed_y, seed_x = candidate["seed_yx"]
        if labels[seed_y, seed_x] != 0:
            suppressed += 1
            continue
        y0, x0, y1, x1 = candidate["bbox"]
        target = labels[y0:y1, x0:x1]
        overlap_pixels += int(np.count_nonzero(candidate["mask"] & (target != 0)))
        available = candidate["mask"] & (target == 0)
        if not np.any(available):
            suppressed += 1
            continue
        next_label += 1
        target[available] = next_label
        if candidate["fallback"]:
            fallbacks += 1
        elif np.count_nonzero(available) > 1:
            grown += 1
    return labels, {
        "refined_masks": next_label,
        "grown_masks": grown,
        "single_pixel_fallbacks": fallbacks,
        "suppressed_duplicate_seeds": suppressed,
        "overlap_pixels_removed": overlap_pixels,
    }


def _refine_spotiflow_points(
    volume: np.ndarray,
    points_by_plane: list[np.ndarray],
    scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Convert Spotiflow points to bounded, locally thresholded instance masks."""
    y_um, x_um = _xy_pixel_sizes_um(scales, "Spotiflow local mask refinement")
    radius_y = max(1.0, _SPOT_REFINEMENT_MAX_RADIUS_UM / y_um)
    radius_x = max(1.0, _SPOT_REFINEMENT_MAX_RADIUS_UM / x_um)
    started = time.perf_counter()
    planes = []
    totals = {
        "detected_points": sum(len(points) for points in points_by_plane),
        "refined_masks": 0,
        "grown_masks": 0,
        "single_pixel_fallbacks": 0,
        "suppressed_duplicate_seeds": 0,
        "overlap_pixels_removed": 0,
        "refinement_max_radius_um": _SPOT_REFINEMENT_MAX_RADIUS_UM,
        "refinement_radius_y_pixels": radius_y,
        "refinement_radius_x_pixels": radius_x,
        "refinement_threshold_fraction": _SPOT_REFINEMENT_THRESHOLD_FRACTION,
        "refinement_noise_sigmas": _SPOT_REFINEMENT_NOISE_SIGMAS,
        "refinement_smoothing": "3x3-binomial",
    }
    for z_index, points in enumerate(points_by_plane):
        smoothed = _smooth_spot_plane(volume[z_index])
        candidates = [
            _local_spot_candidate(smoothed, point, radius_y, radius_x)
            for point in points
        ]
        labels, summary = _merge_local_spot_candidates(
            candidates, volume[z_index].shape
        )
        for name, value in summary.items():
            totals[name] += value
        planes.append(labels)
    labels = _unique_plane_labels(planes) if planes else np.zeros(volume.shape, np.uint32)
    return labels, totals, {
        "local_refinement_seconds": time.perf_counter() - started,
    }


def _segment_cellpose(
    czyx: np.ndarray,
    spec: ModelSpec,
    settings: SegmentationSettings,
    device: str,
    scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    _configure_model_cache()
    image = _selected_image(czyx, settings)
    gpu = device == "cuda"

    def importer():
        if spec.family == "cellpose3":
            from cellpose3_legacy import models
        else:
            from cellpose import models
        return models

    model, timing = _cached_model(
        spec.id,
        device,
        importer,
        lambda models: models.CellposeModel(
            gpu=gpu, pretrained_model=spec.checkpoint
        ),
    )
    inference_started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "diameter": _cellpose_diameter_pixels(
            settings.diameter, settings.target, scales
        ),
        "flow_threshold": settings.flow_threshold,
        "cellprob_threshold": settings.cellprob_threshold,
        "channel_axis": -1,
    }
    diameter_pixels = kwargs["diameter"]
    diameter_source = "configured"
    if settings.diameter == 0:
        diameter_source = "automatic target default"
    elif settings.diameter < 0:
        diameter_source = "model default"
        model_diameter = getattr(model, "diam_mean", None)
        if hasattr(model_diameter, "item"):
            model_diameter = model_diameter.item()
        diameter_pixels = float(model_diameter) if model_diameter is not None else None
    y_um = float(scales.get("y", float("nan")))
    x_um = float(scales.get("x", float("nan")))
    physical_scale_known = (
        np.isfinite(y_um) and y_um > 0 and np.isfinite(x_um) and x_um > 0
    )
    effective = {
        "adapter": "cellpose",
        "diameter_source": diameter_source,
        "diameter_pixels": diameter_pixels,
        "diameter_um": float(diameter_pixels * (y_um + x_um) / 2.0)
        if diameter_pixels is not None and physical_scale_known
        else None,
        "cellprob_threshold": float(settings.cellprob_threshold),
        "flow_threshold": float(settings.flow_threshold),
    }
    native_3d = (
        image.shape[0] > 1
        and spec.dimensions == "3d"
        and settings.dimension_mode != "slice-2d"
    )
    if native_3d:
        masks, *_ = model.eval(image, z_axis=0, do_3D=True, **kwargs)
        labels = np.asarray(masks, dtype=np.uint32)
        return labels, {
            **timing,
            "effective_parameters": effective,
            "inference_seconds": time.perf_counter() - inference_started,
        }
    planes = []
    for z_index in range(image.shape[0]):
        masks, *_ = model.eval(image[z_index], **kwargs)
        planes.append(np.asarray(masks))
    labels = _unique_plane_labels(planes)
    return labels, {
        **timing,
        "effective_parameters": effective,
        "inference_seconds": time.perf_counter() - inference_started,
    }


def _stardist_model_path(checkpoint: str) -> Path:
    return _configure_model_cache() / "stardist" / checkpoint


def _predict_stardist_tiled(
    model,
    image: np.ndarray,
    prob: float | None,
    nms: float | None,
    *,
    collect_polygons: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    tile_size, halo = 1024, 64
    height, width = image.shape[-2:]
    if height <= tile_size and width <= tile_size:
        labels, details = model.predict_instances(
            image, prob_thresh=prob, nms_thresh=nms, normalize=True
        )
        return np.asarray(labels, dtype=np.uint32), details
    output = np.zeros((height, width), dtype=np.uint32)
    offset = 0
    distances: list[np.ndarray] = []
    points: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1, x1 = min(y0 + tile_size, height), min(x0 + tile_size, width)
            ry0, rx0 = max(0, y0 - halo), max(0, x0 - halo)
            ry1, rx1 = min(height, y1 + halo), min(width, x1 + halo)
            labels, details = model.predict_instances(
                image[ry0:ry1, rx0:rx1],
                prob_thresh=prob,
                nms_thresh=nms,
                normalize=True,
            )
            tile_points = np.asarray(details.get("points", []), dtype=np.float32)
            tile_distances = np.asarray(details.get("dist", []), dtype=np.float32)
            tile_probabilities = np.asarray(details.get("prob", []), dtype=np.float32)
            if collect_polygons and tile_points.size:
                if (
                    tile_points.ndim != 2
                    or tile_points.shape[1] != 2
                    or tile_distances.ndim != 2
                    or len(tile_distances) != len(tile_points)
                    or len(tile_probabilities) != len(tile_points)
                ):
                    raise RuntimeError(
                        "StarDist polygon details are malformed; disable Smooth "
                        "Rescaled StarDist Labels to use nearest-neighbor restoration"
                    )
                keep = (
                    (tile_points[:, 0] >= y0 - ry0)
                    & (tile_points[:, 0] < y1 - ry0)
                    & (tile_points[:, 1] >= x0 - rx0)
                    & (tile_points[:, 1] < x1 - rx0)
                )
                if np.any(keep):
                    global_points = tile_points[keep].copy()
                    global_points[:, 0] += ry0
                    global_points[:, 1] += rx0
                    points.append(global_points)
                    distances.append(tile_distances[keep])
                    probabilities.append(tile_probabilities[keep])
            core = np.asarray(labels)[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0]
            if offset:
                core = np.where(core > 0, core.astype(np.uint64) + offset, 0)
            output[y0:y1, x0:x1] = core.astype(np.uint32)
            offset = int(output.max(initial=offset))
    details = {
        "points": np.concatenate(points, axis=0)
        if points
        else np.zeros((0, 2), dtype=np.float32),
        "dist": np.concatenate(distances, axis=0)
        if distances
        else np.zeros((0, 0), dtype=np.float32),
        "prob": np.concatenate(probabilities, axis=0)
        if probabilities
        else np.zeros((0,), dtype=np.float32),
    }
    return output, details


def _restore_stardist_labels(
    labels: np.ndarray,
    details: dict[str, Any],
    source_shape: tuple[int, int],
    *,
    smooth: bool,
) -> tuple[np.ndarray, str]:
    """Restore model-grid labels while preserving StarDist polygon geometry."""
    model_labels = np.asarray(labels, dtype=np.uint32)
    model_shape = tuple(model_labels.shape)
    if model_shape == source_shape:
        return model_labels, "none"
    if not smooth:
        return _resize_2d(model_labels, source_shape, labels=True), "nearest-neighbor"
    try:
        points = np.asarray(details["points"], dtype=np.float32)
        distances = np.asarray(details["dist"], dtype=np.float32)
        probabilities = np.asarray(details["prob"], dtype=np.float32)
        valid = (
            points.ndim == 2
            and points.shape[1] == 2
            and distances.ndim == 2
            and len(distances) == len(points)
            and probabilities.ndim == 1
            and len(probabilities) == len(points)
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError(
            "StarDist polygon details are unavailable or malformed; disable Smooth "
            "Rescaled StarDist Labels to use nearest-neighbor restoration"
        )
    if not len(points):
        return np.zeros(source_shape, dtype=np.uint32), "scaled-polygons"
    scale = np.asarray(
        [source_shape[0] / model_shape[0], source_shape[1] / model_shape[1]],
        dtype=np.float32,
    )
    from cistardist_pytorch.model import polygons_to_label

    restored = polygons_to_label(
        distances,
        points * scale,
        shape=source_shape,
        prob=probabilities,
        scale_dist=(float(scale[0]), float(scale[1])),
    )
    return np.asarray(restored, dtype=np.uint32), "scaled-polygons"


def _segment_stardist(
    czyx: np.ndarray,
    spec: ModelSpec,
    settings: SegmentationSettings,
    device: str,
    scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    channel = settings.selected_channels(czyx.shape[0])[0]
    model, timing = _cached_model(
        spec.id,
        device,
        lambda: __import__("cistardist_pytorch", fromlist=["StarDist2D"]).StarDist2D,
        lambda StarDist2D: StarDist2D.from_folder(
            _stardist_model_path(spec.checkpoint), device=device
        ),
    )
    inference_started = time.perf_counter()
    prob = (
        None
        if settings.stardist_prob_threshold < 0
        else settings.stardist_prob_threshold
    )
    nms = (
        None if settings.stardist_nms_threshold < 0 else settings.stardist_nms_threshold
    )
    model_thresholds = getattr(model, "thresholds", {})
    effective_prob = float(model_thresholds.get("prob", 0.5) if prob is None else prob)
    effective_nms = float(model_thresholds.get("nms", 0.4) if nms is None else nms)
    source_shape = tuple(czyx[channel, 0].shape)
    model_shape = source_shape
    restoration = "none"
    planes = []
    for z_index in range(czyx.shape[1]):
        plane = czyx[channel, z_index]
        original_shape = tuple(plane.shape)
        if spec.checkpoint == "SD_Nuclei_Versatile":
            plane, original_shape = _stardist_versatile_input(plane, scales)
            model_shape = tuple(plane.shape)
        labels, details = _predict_stardist_tiled(
            model,
            plane,
            prob,
            nms,
            collect_polygons=(
                settings.smooth_stardist_labels
                and tuple(plane.shape) != original_shape
            ),
        )
        labels, restoration = _restore_stardist_labels(
            labels,
            details,
            original_shape,
            smooth=settings.smooth_stardist_labels,
        )
        planes.append(labels)
    labels = _unique_plane_labels(planes)
    y_factor = model_shape[0] / source_shape[0]
    x_factor = model_shape[1] / source_shape[1]
    effective = {
        "adapter": "stardist",
        "probability_threshold": effective_prob,
        "probability_source": "thresholds.json" if prob is None else "configured",
        "nms_threshold": effective_nms,
        "nms_source": "thresholds.json" if nms is None else "configured",
        "source_yx_shape": list(source_shape),
        "model_yx_shape": list(model_shape),
        "rescale_y_factor": float(y_factor),
        "rescale_x_factor": float(x_factor),
        "model_y_um": float(scales["y"]) / y_factor if "y" in scales else None,
        "model_x_um": float(scales["x"]) / x_factor if "x" in scales else None,
        "labels_restored_to_source_grid": model_shape != source_shape,
        "label_restoration": restoration,
    }
    return labels, {
        **timing,
        "effective_parameters": effective,
        "inference_seconds": time.perf_counter() - inference_started,
    }


def _segment_instanseg(
    czyx: np.ndarray,
    spec: ModelSpec,
    settings: SegmentationSettings,
    pixel_size_um: float,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            "InstanSeg requires a positive XY pixel size in OME-Zarr metadata"
        )
    channels = (
        list(range(3))
        if spec.checkpoint == "brightfield_nuclei" and czyx.shape[0] >= 3
        else settings.selected_channels(czyx.shape[0])
    )
    if len(channels) < spec.min_channels:
        raise ValueError(
            f"{spec.id} requires at least {spec.min_channels} input channels"
        )
    model_root = _configure_model_cache() / "instanseg"
    checkpoint = model_root / spec.checkpoint / "instanseg.pt"
    def constructor(InstanSeg):
        if checkpoint.exists():
            import torch

            return InstanSeg(
                torch.jit.load(str(checkpoint), map_location="cpu"), verbosity=0
            )
        return InstanSeg(spec.checkpoint, verbosity=0)

    model, timing = _cached_model(
        spec.id,
        device,
        lambda: __import__("instanseg", fromlist=["InstanSeg"]).InstanSeg,
        constructor,
    )
    inference_started = time.perf_counter()
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
    labels = _unique_plane_labels(planes)
    return labels, {
        **timing,
        "effective_parameters": {
            "adapter": "instanseg",
            "pixel_size_um": float(pixel_size_um),
        },
        "inference_seconds": time.perf_counter() - inference_started,
    }


def _segment_spotiflow(
    czyx: np.ndarray,
    spec: ModelSpec,
    settings: SegmentationSettings,
    device: str,
    scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    root = _configure_model_cache()
    cache = Path(os.environ.get("SPOTIFLOW_CACHE_DIR", root / "spotiflow"))
    channel = settings.selected_channels(czyx.shape[0])[0]
    volume = czyx[channel]
    native_3d = (
        volume.shape[0] > 1
        and spec.dimensions == "3d"
        and settings.dimension_mode != "slice-2d"
    )
    if native_3d and settings.spotiflow_local_refinement:
        raise ValueError(
            "Spotiflow Local Mask Refinement supports slice-wise 2D; select "
            "Force slice-wise 2D"
        )
    model, timing = _cached_model(
        spec.id,
        device,
        lambda: __import__("spotiflow.model", fromlist=["Spotiflow"]).Spotiflow,
        lambda Spotiflow: Spotiflow.from_folder(
            cache / spec.checkpoint,
            inference_mode=True,
            which="best",
            map_location=device,
        ),
    )
    threshold = (
        None
        if settings.spotiflow_prob_threshold < 0
        else settings.spotiflow_prob_threshold
    )
    min_distance = _spotiflow_min_distance_pixels(
        settings.spotiflow_min_distance, scales
    )
    model_threshold = getattr(model, "_prob_thresh", [0.5])
    if isinstance(model_threshold, (list, tuple, np.ndarray)):
        model_threshold = model_threshold[0]
    effective_threshold = float(model_threshold if threshold is None else threshold)
    y_um, x_um = _xy_pixel_sizes_um(scales, "Spotiflow minimum distance reporting")
    effective = {
        "adapter": "spotiflow",
        "probability_threshold": effective_threshold,
        "probability_source": "checkpoint thresholds.yaml"
        if threshold is None
        else "configured",
        "minimum_distance_pixels": int(min_distance),
        "minimum_distance_um": float(min_distance * (y_um + x_um) / 2.0),
        "local_refinement": bool(settings.spotiflow_local_refinement),
    }
    detection_started = time.perf_counter()
    if native_3d:
        points, _ = model.predict(
            volume,
            prob_thresh=threshold,
            min_distance=min_distance,
            device=device,
            verbose=False,
        )
        detection_seconds = time.perf_counter() - detection_started
        labels = points_to_labels(points, volume.shape)
        effective.update(
            {
                "output_mode": "point-locations",
                "detected_points": int(len(points)),
            }
        )
        return labels, {
            **timing,
            "model_cache_hits": int(bool(timing.get("model_cache_hit"))),
            "model_cache_misses": int(not bool(timing.get("model_cache_hit"))),
            "locations_only": True,
            "effective_parameters": effective,
            "spot_detection_seconds": detection_seconds,
            "inference_seconds": detection_seconds,
        }
    points_by_plane = [
        np.asarray(
            model.predict(
                volume[z],
                prob_thresh=threshold,
                min_distance=min_distance,
                device=device,
                verbose=False,
            )[0],
            dtype=np.float32,
        ).reshape(-1, 2)
        for z in range(volume.shape[0])
    ]
    detection_seconds = time.perf_counter() - detection_started
    detected_points = sum(len(points) for points in points_by_plane)
    if not settings.spotiflow_local_refinement:
        labels = _unique_plane_labels(
            [
                points_to_labels(points, volume[z].shape)
                for z, points in enumerate(points_by_plane)
            ]
        )
        effective.update(
            {
                "output_mode": "point-locations",
                "detected_points": int(detected_points),
            }
        )
        return labels, {
            **timing,
            "model_cache_hits": int(bool(timing.get("model_cache_hit"))),
            "model_cache_misses": int(not bool(timing.get("model_cache_hit"))),
            "locations_only": True,
            "effective_parameters": effective,
            "spot_detection_seconds": detection_seconds,
            "inference_seconds": detection_seconds,
        }

    labels, refinement, refinement_timing = _refine_spotiflow_points(
        volume, points_by_plane, scales
    )
    effective.update(
        {
            **refinement,
            "output_mode": "instance-masks",
            "refinement_method": "bounded-local-intensity",
        }
    )
    local_refinement_seconds = float(
        refinement_timing.get("local_refinement_seconds", 0.0)
    )
    return labels, {
        "model_cache_hit": bool(timing.get("model_cache_hit")),
        "model_cache_hits": int(bool(timing.get("model_cache_hit"))),
        "model_cache_misses": int(not bool(timing.get("model_cache_hit"))),
        "locations_only": False,
        "import_seconds": float(timing.get("import_seconds", 0.0)),
        "model_load_seconds": float(timing.get("model_load_seconds", 0.0)),
        "effective_parameters": effective,
        "spot_detection_seconds": detection_seconds,
        "local_refinement_seconds": local_refinement_seconds,
        "inference_seconds": detection_seconds + local_refinement_seconds,
    }


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
    start = time.perf_counter()
    torch_was_loaded = "torch" in sys.modules
    device_started = time.perf_counter()
    device = resolve_device(settings.device)
    torch = sys.modules.get("torch")
    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if device == "cuda" and torch is not None
        else "CPU"
    )
    device_seconds = time.perf_counter() - device_started
    device_import_seconds = (
        device_seconds if not torch_was_loaded and "torch" in sys.modules else 0.0
    )
    device_setup_seconds = device_seconds - device_import_seconds
    if spec.family in {"cellpose3", "cellpose-sam"}:
        labels, timing = _segment_cellpose(czyx, spec, settings, device, scales)
    elif spec.family == "stardist":
        labels, timing = _segment_stardist(czyx, spec, settings, device, scales)
    elif spec.family == "instanseg":
        labels, timing = _segment_instanseg(
            czyx,
            spec,
            settings,
            scales.get("x", float("nan")),
            device,
        )
    elif spec.family == "spotiflow":
        labels, timing = _segment_spotiflow(czyx, spec, settings, device, scales)
    else:
        raise ValueError(f"No adapter for model family {spec.family}")
    elapsed = time.perf_counter() - start
    effective_parameters = timing.pop("effective_parameters", {})
    model_cache_hit = bool(timing.pop("model_cache_hit", False))
    model_cache_hits = int(timing.pop("model_cache_hits", int(model_cache_hit)))
    model_cache_misses = int(
        timing.pop("model_cache_misses", int(not model_cache_hit))
    )
    locations_only = bool(
        timing.pop("locations_only", spec.family == "spotiflow")
    )
    timing = {
        **timing,
        "import_seconds": float(timing.get("import_seconds", 0.0))
        + device_import_seconds,
        "device_setup_seconds": device_setup_seconds,
    }
    return np.asarray(labels, dtype=np.uint32), {
        "device": device,
        "device_name": device_name,
        "runtime_seconds": elapsed,
        "object_count": int(np.max(labels, initial=0)),
        "dimension_mode": "native-3d"
        if spec.dimensions == "3d"
        and labels.shape[0] > 1
        and settings.dimension_mode != "slice-2d"
        else "slice-2d",
        "model_cache_hit": model_cache_hit,
        "model_cache_hits": model_cache_hits,
        "model_cache_misses": model_cache_misses,
        "locations_only": locations_only,
        "effective_parameters": effective_parameters,
        "timings": timing,
    }
