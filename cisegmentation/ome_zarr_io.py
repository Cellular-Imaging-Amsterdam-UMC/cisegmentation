from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable

import numpy as np


@dataclass
class ImageResource:
    store_path: Path
    image_path: str = ""
    plate_path: tuple[str, str, str] | None = None
    plate_attrs: dict[str, Any] | None = None
    well_attrs: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.image_path.replace("/", "_") or self.store_path.name.removesuffix(
            ".ome.zarr"
        )


@dataclass
class ImageData:
    data: np.ndarray
    axes: tuple[str, ...]
    scales: dict[str, float]
    attrs: dict[str, Any]
    resource: ImageResource
    source_dtype: str


@dataclass
class LabelResult:
    labels: np.ndarray  # T, 1, Z, Y, X
    source: ImageData
    model_id: str
    target: str
    provenance: dict[str, Any] = field(default_factory=dict)
    channel_labels: list[str] | None = None
    include_original_channels: bool = False
    write_ome_zarr_labels: bool = False


def _attrs(group) -> dict[str, Any]:
    return dict(group.attrs.asdict() if hasattr(group.attrs, "asdict") else group.attrs)


_PHASE_TIMING_KEYS = (
    "startup_seconds",
    "zarr_read_seconds",
    "import_seconds",
    "device_setup_seconds",
    "model_load_seconds",
    "inference_seconds",
    "zarr_write_seconds",
)


def _finalize_timings(
    timings: dict[str, Any] | None, zarr_write_seconds: float
) -> dict[str, float]:
    result = {
        key: float((timings or {}).get(key, 0.0)) for key in _PHASE_TIMING_KEYS
    }
    result["zarr_write_seconds"] = float(zarr_write_seconds)
    result["total_seconds"] = sum(result[key] for key in _PHASE_TIMING_KEYS)
    return result


def _set_write_timing(group, write_started: float) -> None:
    metadata = dict(_attrs(group).get("cisegmentation", {}))
    metadata["timings"] = _finalize_timings(
        metadata.get("timings"), time.perf_counter() - write_started
    )
    group.attrs["cisegmentation"] = metadata


def _install_store(temporary: Path, output_path: Path) -> None:
    """Atomically install a completed store, tolerating brief Windows locks."""
    for attempt in range(20):
        try:
            os.replace(temporary, output_path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def discover_ome_zarrs(input_dir: str | Path) -> list[Path]:
    def is_ngff_store(path: Path) -> bool:
        if not path.is_dir() or not path.name.lower().endswith(".zarr"):
            return False
        attrs_path = path / ".zattrs"
        try:
            attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(attrs.get("plate"), dict) or bool(attrs.get("multiscales"))

    root = Path(input_dir)
    if is_ngff_store(root):
        return [root]
    return sorted(
        path
        for path in root.iterdir()
        if is_ngff_store(path)
    )


def enumerate_resources(store_path: str | Path) -> list[ImageResource]:
    import zarr

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="r")
    root_attrs = _attrs(root)
    plate = root_attrs.get("plate")
    if not isinstance(plate, dict):
        return [ImageResource(store_path)]
    resources: list[ImageResource] = []
    for well in plate.get("wells", []):
        well_path = str(well.get("path", "")).strip("/")
        if not well_path:
            continue
        well_group = root[well_path]
        well_attrs = _attrs(well_group)
        for image in (well_attrs.get("well") or {}).get("images", []):
            field = str(image.get("path", "")).strip("/")
            image_path = f"{well_path}/{field}"
            row, column = well_path.split("/", 1)
            resources.append(
                ImageResource(
                    store_path, image_path, (row, column, field), root_attrs, well_attrs
                )
            )
    if not resources:
        raise ValueError(f"OME-Zarr plate contains no fields: {store_path}")
    return resources


def _axis_names(multiscale: dict, ndim: int) -> tuple[str, ...]:
    axes = multiscale.get("axes") or []
    names = tuple(
        str(axis.get("name") if isinstance(axis, dict) else axis).lower()
        for axis in axes
    )
    if len(names) == ndim:
        return names
    defaults = {
        2: ("y", "x"),
        3: ("z", "y", "x"),
        4: ("c", "z", "y", "x"),
        5: ("t", "c", "z", "y", "x"),
    }
    if ndim not in defaults:
        raise ValueError(f"Cannot infer axes for {ndim}-D OME-Zarr array")
    return defaults[ndim]


def _scale_map(multiscale: dict, axes: tuple[str, ...]) -> dict[str, float]:
    datasets = multiscale.get("datasets") or []
    transforms = (
        (datasets[0].get("coordinateTransformations") or []) if datasets else []
    )
    values = next(
        (item.get("scale") for item in transforms if item.get("type") == "scale"), None
    )
    if not values or len(values) != len(axes):
        return {}
    return {axis: float(value) for axis, value in zip(axes, values)}


def _to_tczyx(data: np.ndarray, axes: tuple[str, ...]) -> np.ndarray:
    known = set("tczyx")
    if any(axis not in known for axis in axes):
        raise ValueError(f"Unsupported axes: {axes}")
    result = data
    current = list(axes)
    for axis in "tczyx":
        if axis not in current:
            result = np.expand_dims(result, axis=0)
            current.insert(0, axis)
    permutation = [current.index(axis) for axis in "tczyx"]
    return np.transpose(result, permutation)


def _to_native_byte_order(data: np.ndarray) -> np.ndarray:
    """Return values in the platform byte order expected by PyTorch models."""
    array = np.asarray(data)
    if array.dtype.isnative:
        return array
    return array.astype(array.dtype.newbyteorder("="), copy=False)


def read_image(resource: ImageResource) -> ImageData:
    import zarr

    root = zarr.open_group(str(resource.store_path), mode="r")
    group = root[resource.image_path] if resource.image_path else root
    attrs = _attrs(group)
    multiscales = attrs.get("multiscales") or []
    if not multiscales:
        raise ValueError(
            f"No multiscales metadata at {resource.store_path}/{resource.image_path}"
        )
    multiscale = multiscales[0]
    dataset_path = str(multiscale["datasets"][0]["path"])
    array = group[dataset_path]
    raw = np.asarray(array)
    source_dtype = str(raw.dtype)
    axes = _axis_names(multiscale, raw.ndim)
    scales = _scale_map(multiscale, axes)
    return ImageData(
        _to_native_byte_order(_to_tczyx(raw, axes)),
        axes,
        scales,
        attrs,
        resource,
        source_dtype,
    )


def _downsample_labels(data: np.ndarray) -> np.ndarray:
    return data[..., ::2, ::2]


def _output_pixels(result: LabelResult) -> tuple[np.ndarray, int]:
    """Return label-safe int32 output pixels and the original channel count."""
    labels = np.asarray(result.labels)
    label_max = int(labels.max(initial=0))
    if label_max > np.iinfo(np.int32).max:
        raise OverflowError(
            f"Maximum label ID {label_max} exceeds QuPath-compatible int32 range"
        )
    if not result.include_original_channels:
        return labels.astype(np.int32, copy=False), 0

    original = np.asarray(result.source.data)
    if not (
        np.issubdtype(original.dtype, np.integer)
        or np.issubdtype(original.dtype, np.floating)
    ):
        raise TypeError(
            "Including original channels supports integer or floating-point "
            f"source pixels; source dtype is {original.dtype}"
        )
    if original.size:
        if np.issubdtype(original.dtype, np.floating) and not np.all(
            np.isfinite(original)
        ):
            raise ValueError(
                "Floating-point original channels contain NaN or infinity and "
                "cannot be converted safely to int32"
            )
        range_values = (
            np.rint(original)
            if np.issubdtype(original.dtype, np.floating)
            else original
        )
        minimum = float(range_values.min())
        maximum = float(range_values.max())
        limits = np.iinfo(np.int32)
        if minimum < limits.min or maximum > limits.max:
            raise OverflowError(
                f"Original pixel range {minimum}..{maximum} does not fit int32"
            )
    if original.shape[0] != labels.shape[0] or original.shape[2:] != labels.shape[2:]:
        raise ValueError("Original and label pixels must have matching T, Z, Y and X")
    converted = (
        np.rint(original).astype(np.int32)
        if np.issubdtype(original.dtype, np.floating)
        else original.astype(np.int32, copy=False)
    )
    return (
        np.concatenate(
            (converted, labels.astype(np.int32, copy=False)),
            axis=1,
        ),
        original.shape[1],
    )


def _axis_metadata() -> list[dict[str, str]]:
    return [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def _scale_values(source: ImageData, xy_factor: int = 1) -> list[float]:
    return [
        source.scales.get("t", 1.0),
        1.0,
        source.scales.get("z", 1.0),
        source.scales.get("y", 1.0) * xy_factor,
        source.scales.get("x", 1.0) * xy_factor,
    ]


_LABEL_COLORS = ("00FF00", "0000FF", "FF00FF", "FFFF00", "00FFFF", "FF8000", "FF0000")


def _label_channel_color(label: str, index: int) -> str:
    text = label.lower()
    if "cytoplasm" in text:
        return "FF00FF"
    if "nucle" in text:
        return "0000FF"
    if "cell" in text:
        return "00FF00"
    if "spot" in text or "foci" in text:
        return ("FFFF00", "00FFFF", "FF8000", "FF0000")[index % 4]
    return _LABEL_COLORS[index % len(_LABEL_COLORS)]


def _ome_color_int(color: str) -> int:
    rgba = (int(color, 16) << 8) | 255
    return rgba if rgba < 2**31 else rgba - 2**32


def _source_channel_metadata(result: LabelResult, count: int) -> list[dict[str, Any]]:
    source_channels = (result.source.attrs.get("omero") or {}).get("channels") or []
    metadata = []
    for index in range(count):
        source = source_channels[index] if index < len(source_channels) else {}
        pixels = np.asarray(result.source.data[:, index])
        minimum = float(pixels.min(initial=0))
        maximum = float(pixels.max(initial=0))
        window = source.get("window") or {}
        metadata.append(
            {
                "label": str(source.get("label") or f"original channel {index + 1}"),
                "color": str(source.get("color") or _LABEL_COLORS[index % len(_LABEL_COLORS)]),
                "active": bool(source.get("active", True)),
                "window": {
                    "start": float(window.get("start", minimum)),
                    "end": float(window.get("end", maximum)),
                    "min": float(window.get("min", minimum)),
                    "max": float(window.get("max", maximum)),
                },
            }
        )
    return metadata


def _output_channel_metadata(
    result: LabelResult, pixels: np.ndarray, original_count: int
) -> list[dict[str, Any]]:
    channels = _source_channel_metadata(result, original_count)
    label_names = result.channel_labels or [f"{result.target} labels"]
    for label_index, channel_name in enumerate(label_names):
        output_index = original_count + label_index
        maximum = max(1, int(pixels[:, output_index].max(initial=0)))
        channels.append(
            {
                "label": channel_name,
                "color": _label_channel_color(channel_name, label_index),
                "lookupTable": "glasbey_inverted.lut",
                "active": True,
                "window": {
                    "start": 0.0,
                    "end": float(maximum),
                    "min": 0.0,
                    "max": float(maximum),
                },
            }
        )
    return channels


def _ome_xml(
    result: LabelResult,
    name: str,
    pixels: np.ndarray,
    channels_metadata: list[dict[str, Any]],
) -> str:
    t, c, z, y, x = pixels.shape
    px_x, px_y, px_z = (result.source.scales.get(axis, 1.0) for axis in ("x", "y", "z"))
    channels = "".join(
        f'<Channel ID="Channel:0:{index}" Name="{channel["label"]}" '
        f'Color="{_ome_color_int(channel["color"])}" '
        'SamplesPerPixel="1"/>'
        for index, channel in enumerate(channels_metadata)
    )
    dtype = np.dtype(pixels.dtype)
    ome_type = {
        np.dtype("uint8"): "uint8",
        np.dtype("int8"): "int8",
        np.dtype("uint16"): "uint16",
        np.dtype("int16"): "int16",
        np.dtype("uint32"): "uint32",
        np.dtype("int32"): "int32",
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
    }.get(dtype)
    if ome_type is None:
        raise TypeError(f"OME-XML does not support output dtype {dtype}")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0" Name="{name}"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
        f'Type="{ome_type}" SizeX="{x}" SizeY="{y}" SizeZ="{z}" SizeC="{c}" SizeT="{t}" '
        f'PhysicalSizeX="{px_x}" PhysicalSizeY="{px_y}" PhysicalSizeZ="{px_z}">'
        f"{channels}"
        "</Pixels></Image></OME>"
    )


def _write_image_group(group, result: LabelResult, name: str) -> None:
    pixels, original_count = _output_pixels(result)
    levels = [pixels]
    while min(levels[-1].shape[-2:]) >= 512 and len(levels) < 5:
        levels.append(_downsample_labels(levels[-1]))
    datasets = []
    for index, level in enumerate(levels):
        chunks = (1, 1, 1, min(512, level.shape[-2]), min(512, level.shape[-1]))
        array = group.create_dataset(
            str(index),
            shape=level.shape,
            data=level,
            chunks=chunks,
            overwrite=True,
            dimension_separator="/",
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]
        datasets.append(
            {
                "path": str(index),
                "coordinateTransformations": [
                    {"type": "scale", "scale": _scale_values(result.source, 2**index)}
                ],
            }
        )
    group.attrs["multiscales"] = [
        {"version": "0.4", "name": name, "axes": _axis_metadata(), "datasets": datasets}
    ]
    channels_metadata = _output_channel_metadata(result, pixels, original_count)
    group.attrs["omero"] = {
        "version": "0.4",
        "name": name,
        "channels": channels_metadata,
        "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"},
    }
    group.attrs["cisegmentation"] = {
        "model": result.model_id,
        "target": result.target,
        "source": str(result.source.resource.store_path.name),
        "storage_dtype": "int32",
        "original_channel_count": original_count,
        "original_source_dtype": result.source.source_dtype
        if original_count
        else None,
        "original_channels_conversion": "round-to-nearest-int32"
        if original_count
        and np.issubdtype(np.dtype(result.source.source_dtype), np.floating)
        else ("lossless-int32-cast" if original_count else None),
        "label_rendering": {
            "lookup_table": "glasbey_inverted.lut",
            "rendering_only": True,
            "pixel_values_transformed": False,
        },
        **result.provenance,
    }
    store_root = getattr(group.store, "path", None) or getattr(
        group.store, "root", None
    )
    if store_root is None:
        raise RuntimeError(
            "OME-XML sidecar writing requires a local directory-backed Zarr store"
        )
    ome = Path(store_root) / group.path / "OME"
    ome.mkdir(parents=True, exist_ok=True)
    (ome / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
    (ome / "METADATA.ome.xml").write_text(
        _ome_xml(result, name, pixels, channels_metadata), encoding="utf-8"
    )


def _label_group_names(result: LabelResult) -> list[str]:
    """Return safe, unique group names without changing displayed label names."""
    labels = result.channel_labels or [f"labels_{result.target}"]
    names: list[str] = []
    used: set[str] = set()
    for index, label in enumerate(labels, start=1):
        base = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(label)).strip("_.-")
        base = base or f"labels_{index}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        names.append(candidate)
    return names


def _write_native_ome_zarr_labels(group, result: LabelResult, name: str) -> None:
    """Write an NGFF 0.4 image with associated image-label groups."""
    source_pixels = np.asarray(result.source.data)
    source_levels = [source_pixels]
    while min(source_levels[-1].shape[-2:]) >= 512 and len(source_levels) < 5:
        source_levels.append(_downsample_labels(source_levels[-1]))

    source_datasets = []
    for index, level in enumerate(source_levels):
        chunks = (1, 1, 1, min(512, level.shape[-2]), min(512, level.shape[-1]))
        array = group.create_dataset(
            str(index),
            shape=level.shape,
            data=level,
            chunks=chunks,
            overwrite=True,
            dimension_separator="/",
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]
        source_datasets.append(
            {
                "path": str(index),
                "coordinateTransformations": [
                    {"type": "scale", "scale": _scale_values(result.source, 2**index)}
                ],
            }
        )
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": name,
            "axes": _axis_metadata(),
            "datasets": source_datasets,
        }
    ]
    source_channels = _source_channel_metadata(result, source_pixels.shape[1])
    group.attrs["omero"] = {
        "version": "0.4",
        "name": name,
        "channels": source_channels,
        "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"},
    }

    labels = np.asarray(result.labels)
    label_names = result.channel_labels or [f"labels_{result.target}"]
    if labels.shape[1] != len(label_names):
        raise ValueError(
            "The number of label channels does not match the channel label names"
        )
    group_names = _label_group_names(result)
    labels_group = group.require_group("labels")
    labels_group.attrs["labels"] = group_names
    for label_index, (group_name, display_name) in enumerate(
        zip(group_names, label_names)
    ):
        label_group = labels_group.require_group(group_name)
        label_levels = [labels[:, label_index : label_index + 1]]
        while min(label_levels[-1].shape[-2:]) >= 512 and len(label_levels) < 5:
            label_levels.append(_downsample_labels(label_levels[-1]))
        datasets = []
        for level_index, level in enumerate(label_levels):
            chunks = (
                1,
                1,
                1,
                min(512, level.shape[-2]),
                min(512, level.shape[-1]),
            )
            array = label_group.create_dataset(
                str(level_index),
                shape=level.shape,
                data=level,
                chunks=chunks,
                overwrite=True,
                dimension_separator="/",
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]
            datasets.append(
                {
                    "path": str(level_index),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": _scale_values(result.source, 2**level_index),
                        }
                    ],
                }
            )
        label_group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": str(display_name),
                "axes": _axis_metadata(),
                "datasets": datasets,
            }
        ]
        label_group.attrs["image-label"] = {
            "version": "0.4",
            "source": {"image": "../../"},
        }

    group.attrs["cisegmentation"] = {
        "model": result.model_id,
        "target": result.target,
        "source": result.source.resource.store_path.name,
        "storage_dtype": str(source_pixels.dtype),
        "label_storage_dtype": str(labels.dtype),
        "output_layout": "ome-zarr-0.4-labels",
        "label_groups": [f"labels/{group_name}" for group_name in group_names],
        **result.provenance,
    }
    store_root = getattr(group.store, "path", None) or getattr(
        group.store, "root", None
    )
    if store_root is None:
        raise RuntimeError(
            "OME-XML sidecar writing requires a local directory-backed Zarr store"
        )
    ome = Path(store_root) / group.path / "OME"
    ome.mkdir(parents=True, exist_ok=True)
    (ome / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
    (ome / "METADATA.ome.xml").write_text(
        _ome_xml(result, name, source_pixels, source_channels), encoding="utf-8"
    )


def write_label_image(result: LabelResult, output_path: str | Path) -> Path:
    import zarr

    write_started = time.perf_counter()
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    root = zarr.open_group(str(temporary), mode="w", zarr_version=2)
    writer = (
        _write_native_ome_zarr_labels
        if result.write_ome_zarr_labels
        else _write_image_group
    )
    writer(root, result, output_path.name.removesuffix(".ome.zarr"))
    _set_write_timing(root, write_started)
    root.store.close()
    if output_path.exists():
        shutil.rmtree(output_path)
    _install_store(temporary, output_path)
    return output_path


def write_rgb_gallery(
    cyx: np.ndarray,
    source: ImageData,
    provenance: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a synthetic 2D RGB benchmark montage as NGFF 0.4/Zarr v2."""
    import zarr

    write_started = time.perf_counter()
    cyx = np.asarray(cyx, dtype=np.uint8)
    if cyx.ndim != 3 or cyx.shape[0] != 3:
        raise ValueError(f"RGB gallery must have shape (3,Y,X), got {cyx.shape}")
    data = cyx[None, :, None]
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    root = zarr.open_group(str(temporary), mode="w", zarr_version=2)
    levels = [data]
    while min(levels[-1].shape[-2:]) >= 512 and len(levels) < 5:
        levels.append(levels[-1][..., ::2, ::2])
    datasets = []
    for index, level in enumerate(levels):
        array = root.create_dataset(
            str(index),
            data=level,
            shape=level.shape,
            chunks=(1, 1, 1, min(512, level.shape[-2]), min(512, level.shape[-1])),
            overwrite=True,
            dimension_separator="/",
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]
        datasets.append(
            {
                "path": str(index),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0, 2**index, 2**index]}
                ],
            }
        )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": output_path.name.removesuffix(".ome.zarr"),
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": datasets,
        }
    ]
    root.attrs["omero"] = {
        "version": "0.4",
        "name": output_path.name.removesuffix(".ome.zarr"),
        "channels": [
            {
                "label": label,
                "color": color,
                "active": True,
                "window": {"start": 0.0, "end": 255.0, "min": 0.0, "max": 255.0},
            }
            for label, color in (
                ("Red", "FF0000"),
                ("Green", "00FF00"),
                ("Blue", "0000FF"),
            )
        ],
        "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"},
    }
    root.attrs["cisegmentation"] = {
        "model": "benchmark-gallery",
        "target": "visual-comparison",
        "source": source.resource.store_path.name,
        **provenance,
    }
    ome = temporary / "OME"
    ome.mkdir(parents=True, exist_ok=True)
    (ome / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
    _, _, _, height, width = data.shape
    channels = "".join(
        f'<Channel ID="Channel:0:{index}" Name="{name}" SamplesPerPixel="1"/>'
        for index, name in enumerate(("Red", "Green", "Blue"))
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0" Name="{output_path.stem}"><Pixels ID="Pixels:0" '
        f'DimensionOrder="XYZCT" Type="uint8" SizeX="{width}" SizeY="{height}" '
        f'SizeZ="1" SizeC="3" SizeT="1">{channels}</Pixels></Image></OME>'
    )
    (ome / "METADATA.ome.xml").write_text(xml, encoding="utf-8")
    _set_write_timing(root, write_started)
    root.store.close()
    if output_path.exists():
        shutil.rmtree(output_path)
    _install_store(temporary, output_path)
    return output_path


def write_hcs_plate(results: Iterable[LabelResult], output_path: str | Path) -> Path:
    import zarr

    write_started = time.perf_counter()
    result_list = list(results)
    if not result_list:
        raise ValueError("Cannot write an empty HCS result")
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    root = zarr.open_group(str(temporary), mode="w", zarr_version=2)
    source_plate = result_list[0].source.resource.plate_attrs or {}
    root.attrs.update(
        {key: value for key, value in source_plate.items() if key != "omero"}
    )
    wells: dict[str, list[str]] = {}
    for result in result_list:
        row, column, field = result.source.resource.plate_path or ("A", "1", "0")
        well_path = f"{row}/{column}"
        wells.setdefault(well_path, []).append(field)
        group = root.require_group(f"{well_path}/{field}")
        writer = (
            _write_native_ome_zarr_labels
            if result.write_ome_zarr_labels
            else _write_image_group
        )
        writer(group, result, f"{output_path.stem}_{row}_{column}_{field}")
    for well_path, fields in wells.items():
        well = root.require_group(well_path)
        well.attrs["well"] = {
            "images": [{"path": field, "acquisition": 0} for field in sorted(fields)],
            "version": "0.4",
        }
    if "plate" not in root.attrs:
        root.attrs["plate"] = {
            "version": "0.4",
            "rows": [],
            "columns": [],
            "wells": [{"path": path} for path in sorted(wells)],
        }
    timing_records = [result.provenance.get("timings", {}) for result in result_list]
    aggregated = {
        key: (
            max((float(record.get(key, 0.0)) for record in timing_records), default=0.0)
            if key == "startup_seconds"
            else sum(float(record.get(key, 0.0)) for record in timing_records)
        )
        for key in _PHASE_TIMING_KEYS
        if key != "zarr_write_seconds"
    }
    root.attrs["cisegmentation"] = {
        "model": result_list[0].model_id,
        "target": result_list[0].target,
        "source": result_list[0].source.resource.store_path.name,
        "field_count": len(result_list),
        "model_cache_hits": sum(
            int(result.provenance.get("model_cache_hits", 0))
            for result in result_list
        ),
        "model_cache_misses": sum(
            int(result.provenance.get("model_cache_misses", 0))
            for result in result_list
        ),
        "timings": aggregated,
    }
    _set_write_timing(root, write_started)
    root.store.close()
    if output_path.exists():
        shutil.rmtree(output_path)
    _install_store(temporary, output_path)
    return output_path
