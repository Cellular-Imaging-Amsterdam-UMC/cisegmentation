from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable
from uuid import uuid4

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
    label_origins: list[str] | None = None


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
    "measurement_seconds",
)
_DETAIL_TIMING_KEYS = (
    "spot_detection_seconds",
    "local_refinement_seconds",
)


def _finalize_timings(
    timings: dict[str, Any] | None, zarr_write_seconds: float
) -> dict[str, float]:
    result = {
        key: float((timings or {}).get(key, 0.0)) for key in _PHASE_TIMING_KEYS
    }
    result.update(
        {
            key: float((timings or {}).get(key, 0.0))
            for key in _DETAIL_TIMING_KEYS
            if key in (timings or {})
        }
    )
    result["zarr_write_seconds"] = float(zarr_write_seconds)
    result["total_seconds"] = sum(result[key] for key in _PHASE_TIMING_KEYS)
    return result


def _set_write_timing(group, write_started: float) -> None:
    metadata = dict(_attrs(group).get("cisegmentation", {}))
    metadata["timings"] = _finalize_timings(
        metadata.get("timings"), time.perf_counter() - write_started
    )
    group.attrs["cisegmentation"] = metadata


def set_measurement_timing(
    output_path: str | Path, measurement_seconds: float
) -> None:
    """Persist database measurement time after the OME-Zarr output is complete."""
    import zarr

    root = zarr.open_group(str(output_path), mode="a")
    metadata = dict(_attrs(root).get("cisegmentation", {}))
    timings = dict(metadata.get("timings", {}))
    timings["measurement_seconds"] = float(measurement_seconds)
    metadata["timings"] = _finalize_timings(
        timings, float(timings.get("zarr_write_seconds", 0.0))
    )
    root.attrs["cisegmentation"] = metadata
    root.store.close()


def new_output_store_uuid() -> str:
    """Return the portable identity shared by an output store and its database."""
    return str(uuid4())


def _set_output_store_uuid(group, output_store_uuid: str) -> None:
    metadata = dict(_attrs(group).get("cisegmentation", {}))
    metadata["output_store_uuid"] = output_store_uuid
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


def _source_axis_metadata(source: ImageData) -> list[dict[str, str]]:
    kinds = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    return [
        {
            "name": axis,
            "type": kinds[axis],
            **(
                {"unit": "micrometer"}
                if axis in {"z", "y", "x"}
                else {}
            ),
        }
        for axis in source.axes
    ]


def _source_scale_values(source: ImageData, xy_factor: int = 1) -> list[float]:
    values = {
        "t": source.scales.get("t", 1.0),
        "c": 1.0,
        "z": source.scales.get("z", 1.0),
        "y": source.scales.get("y", 1.0) * xy_factor,
        "x": source.scales.get("x", 1.0) * xy_factor,
    }
    return [values[axis] for axis in source.axes]


def _from_tczyx(data: np.ndarray, axes: tuple[str, ...]) -> np.ndarray:
    source_axes = ("t", "c", "z", "y", "x")
    result = np.asarray(data)
    for index in reversed(range(len(source_axes))):
        if source_axes[index] not in axes:
            if result.shape[index] != 1:
                raise ValueError(
                    f"Cannot omit non-singleton {source_axes[index].upper()} "
                    "axis from a native label image"
                )
            result = np.squeeze(result, axis=index)
    remaining = [axis for axis in source_axes if axis in axes]
    permutation = [remaining.index(axis) for axis in axes]
    return np.transpose(result, permutation)


_LABEL_COLORS = ("00FF00", "0000FF", "FF00FF", "FFFF00", "00FFFF", "FF8000", "FF0000")


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


def existing_label_names(resource: ImageResource) -> list[str]:
    """Return validated native label group names for one source image."""
    import zarr

    root = zarr.open_group(str(resource.store_path), mode="r")
    group = root[resource.image_path] if resource.image_path else root
    if "labels" not in group:
        root.store.close()
        return []
    labels_group = group["labels"]
    declared = [str(name) for name in labels_group.attrs.get("labels", [])]
    available = set(labels_group.group_keys())
    missing = [name for name in declared if name not in available]
    if missing:
        root.store.close()
        raise ValueError(
            f"Declared OME-Zarr labels are missing for {resource.name}: "
            + ", ".join(missing)
        )
    source_multiscale = (_attrs(group).get("multiscales") or [{}])[0]
    source_dataset = str(
        (source_multiscale.get("datasets") or [{"path": "0"}])[0]["path"]
    )
    source_shape = tuple(group[source_dataset].shape)
    source_axes = _axis_names(source_multiscale, len(source_shape))
    for name in declared:
        label_group = labels_group[name]
        label_multiscale = (_attrs(label_group).get("multiscales") or [{}])[0]
        label_dataset = str(
            (label_multiscale.get("datasets") or [{"path": "0"}])[0]["path"]
        )
        if label_dataset not in label_group:
            root.store.close()
            raise ValueError(
                f"Label {name!r} has no declared level-0 array"
            )
        array = label_group[label_dataset]
        if not np.issubdtype(array.dtype, np.integer):
            root.store.close()
            raise ValueError(f"Label {name!r} must use an integer data type")
        label_shape = tuple(array.shape)
        label_axes = _axis_names(label_multiscale, len(label_shape))
        source_sizes = dict(zip(source_axes, source_shape))
        incompatible = any(
            axis not in source_sizes
            or size not in (1, source_sizes[axis])
            for axis, size in zip(label_axes, label_shape)
        ) or any(
            axis not in label_axes and size != 1 and axis != "c"
            for axis, size in source_sizes.items()
        )
        if incompatible:
            root.store.close()
            raise ValueError(
                f"Label {name!r} axes/shape {label_axes}/{label_shape} are "
                f"incompatible with source {source_axes}/{source_shape}"
            )
    root.store.close()
    return declared


def read_native_label(
    resource: ImageResource, group_name: str, *, store_path: str | Path | None = None
) -> np.ndarray:
    """Read one native label level as TCZYX."""
    import zarr

    root = zarr.open_group(str(store_path or resource.store_path), mode="r")
    prefix = f"{resource.image_path}/" if resource.image_path else ""
    group = root[f"{prefix}labels/{group_name}"]
    multiscale = (_attrs(group).get("multiscales") or [{}])[0]
    dataset = str((multiscale.get("datasets") or [{"path": "0"}])[0]["path"])
    raw = np.asarray(group[dataset])
    axes = _axis_names(multiscale, raw.ndim)
    result = _to_tczyx(raw, axes)
    root.store.close()
    return np.asarray(result, dtype=np.uint32)


def write_native_label_groups(
    store_path: str | Path,
    resource_path: str,
    result: LabelResult,
    generated_group_names: list[str],
    final_group_names: list[str],
) -> None:
    """Write only generated native label groups into an existing hierarchy."""
    import zarr

    if len(generated_group_names) != result.labels.shape[1]:
        raise ValueError("Generated label group names do not match label channels")
    root = zarr.open_group(str(store_path), mode="a")
    group = root[resource_path] if resource_path else root
    labels_group = group.require_group("labels")
    labels_group.attrs["labels"] = list(final_group_names)
    display_names = result.channel_labels or generated_group_names
    for label_index, (group_name, display_name) in enumerate(
        zip(generated_group_names, display_names)
    ):
        if group_name in labels_group:
            del labels_group[group_name]
        label_group = labels_group.require_group(group_name)
        label_levels = [
            _from_tczyx(
                np.asarray(result.labels[:, label_index : label_index + 1]),
                result.source.axes,
            )
        ]
        while min(label_levels[-1].shape[-2:]) >= 512 and len(label_levels) < 5:
            label_levels.append(_downsample_labels(label_levels[-1]))
        datasets = []
        for level_index, level in enumerate(label_levels):
            chunks = tuple(
                min(512, size) if index >= level.ndim - 2 else 1
                for index, size in enumerate(level.shape)
            )
            array = label_group.create_dataset(
                str(level_index),
                shape=level.shape,
                data=level,
                chunks=chunks,
                overwrite=True,
                dimension_separator="/",
            )
            array.attrs["_ARRAY_DIMENSIONS"] = list(result.source.axes)
            datasets.append(
                {
                    "path": str(level_index),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": _source_scale_values(
                                result.source, 2**level_index
                            ),
                        }
                    ],
                }
            )
        label_group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": str(display_name),
                "axes": _source_axis_metadata(result.source),
                "datasets": datasets,
            }
        ]
        label_group.attrs["image-label"] = {
            "version": "0.4",
            "source": {"image": "../../"},
        }
    metadata = dict(_attrs(group).get("cisegmentation", {}))
    metadata.update(
        {
            "model": result.model_id,
            "target": result.target,
            "source": result.source.resource.store_path.name,
            "label_storage_dtype": str(np.asarray(result.labels).dtype),
            "output_layout": "ome-zarr-0.4-labels",
            "label_groups": [f"labels/{name}" for name in final_group_names],
            **result.provenance,
        }
    )
    group.attrs["cisegmentation"] = metadata
    root.store.close()


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


def write_label_image(
    result: LabelResult,
    output_path: str | Path,
    *,
    output_store_uuid: str | None = None,
) -> Path:
    import zarr

    write_started = time.perf_counter()
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    root = zarr.open_group(str(temporary), mode="w", zarr_version=2)
    _write_native_ome_zarr_labels(
        root, result, output_path.name.removesuffix(".ome.zarr")
    )
    _set_output_store_uuid(root, output_store_uuid or new_output_store_uuid())
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


class HCSPlateWriter:
    """Incrementally write one completed segmentation field at a time."""

    def __init__(
        self,
        resources: Iterable[ImageResource],
        output_path: str | Path,
        *,
        output_store_uuid: str | None = None,
    ) -> None:
        import zarr

        self.resources = list(resources)
        if not self.resources:
            raise ValueError("Cannot write an empty HCS result")
        if any(resource.plate_path is None for resource in self.resources):
            raise ValueError("Every HCS resource must have a plate field path")
        self.output_path = Path(output_path)
        self.output_store_uuid = output_store_uuid or new_output_store_uuid()
        self.temporary = self.output_path.with_name(self.output_path.name + ".partial")
        if self.temporary.exists():
            shutil.rmtree(self.temporary)
        self.root = zarr.open_group(str(self.temporary), mode="w", zarr_version=2)
        source_plate = self.resources[0].plate_attrs or {}
        self.root.attrs.update(
            {key: value for key, value in source_plate.items() if key != "omero"}
        )
        plate_metadata = source_plate.get("plate") or {}
        valid_acquisition_ids = {
            acquisition["id"]
            for acquisition in plate_metadata.get("acquisitions", [])
            if isinstance(acquisition, dict) and "id" in acquisition
        }
        self.wells: dict[str, list[dict[str, Any]]] = {}
        for resource in self.resources:
            row, column, field = resource.plate_path  # type: ignore[misc]
            image_attrs: dict[str, Any] = {"path": field}
            source_images = ((resource.well_attrs or {}).get("well") or {}).get(
                "images", []
            )
            source_image = next(
                (
                    image
                    for image in source_images
                    if str(image.get("path", "")).strip("/") == field
                ),
                None,
            )
            if source_image is not None:
                acquisition_id = source_image.get("acquisition")
                if acquisition_id in valid_acquisition_ids:
                    image_attrs["acquisition"] = acquisition_id
            self.wells.setdefault(f"{row}/{column}", []).append(image_attrs)
        for well_path, images in self.wells.items():
            well = self.root.require_group(well_path)
            well.attrs["well"] = {
                "images": sorted(images, key=lambda image: image["path"]),
                "version": "0.4",
            }
        if "plate" not in self.root.attrs:
            self.root.attrs["plate"] = {
                "version": "0.4",
                "rows": [],
                "columns": [],
                "wells": [{"path": path} for path in sorted(self.wells)],
            }
        self._expected_paths = {
            "/".join(resource.plate_path or ()) for resource in self.resources
        }
        self._written_paths: set[str] = set()
        self._model_id: str | None = None
        self._target: str | None = None
        self._source_name: str | None = None
        self._timing_records: list[dict[str, Any]] = []
        self._model_cache_hits = 0
        self._model_cache_misses = 0
        self._result_cache_hits = 0
        self._segmentation_seconds = 0.0
        self._segmentation_count = 0
        self._write_seconds = 0.0
        self._closed = False

    def append(self, result: LabelResult) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed HCS plate writer")
        plate_path = result.source.resource.plate_path
        if plate_path is None:
            raise ValueError("HCS result is missing its plate field path")
        resource_path = "/".join(plate_path)
        if resource_path not in self._expected_paths:
            raise ValueError(f"Unexpected HCS field: {resource_path}")
        if resource_path in self._written_paths:
            raise ValueError(f"HCS field was written more than once: {resource_path}")
        row, column, field = plate_path
        group = self.root.require_group(resource_path)
        write_started = time.perf_counter()
        _write_native_ome_zarr_labels(
            group,
            result,
            f"{self.output_path.stem}_{row}_{column}_{field}",
        )
        _set_write_timing(group, write_started)
        self._write_seconds += time.perf_counter() - write_started
        self._written_paths.add(resource_path)
        if self._model_id is None:
            self._model_id = result.model_id
            self._target = result.target
            self._source_name = result.source.resource.store_path.name
        self._timing_records.append(dict(result.provenance.get("timings", {})))
        self._model_cache_hits += int(result.provenance.get("model_cache_hits", 0))
        self._model_cache_misses += int(
            result.provenance.get("model_cache_misses", 0)
        )
        self._result_cache_hits += int(result.provenance.get("result_cache_hits", 0))
        self._segmentation_seconds += float(
            result.provenance.get("runtime_seconds", 0.0)
        )
        self._segmentation_count += int(
            result.provenance.get("segmentation_count", 0)
        )

    def finalize(self) -> Path:
        if self._closed:
            raise RuntimeError("HCS plate writer is already closed")
        missing = self._expected_paths - self._written_paths
        if missing:
            raise ValueError(
                "Cannot finalize HCS output; fields were not written: "
                + ", ".join(sorted(missing))
            )
        if self._model_id is None or self._target is None or self._source_name is None:
            raise ValueError("Cannot write an empty HCS result")
        timing_records = self._timing_records
        aggregate_keys = [
            key for key in _PHASE_TIMING_KEYS if key != "zarr_write_seconds"
        ]
        aggregate_keys.extend(
            key
            for key in _DETAIL_TIMING_KEYS
            if any(key in record for record in timing_records)
        )
        aggregated = {
            key: (
                max(
                    (float(record.get(key, 0.0)) for record in timing_records),
                    default=0.0,
                )
                if key == "startup_seconds"
                else sum(float(record.get(key, 0.0)) for record in timing_records)
            )
            for key in aggregate_keys
        }
        self.root.attrs["cisegmentation"] = {
            "model": self._model_id,
            "target": self._target,
            "source": self._source_name,
            "output_store_uuid": self.output_store_uuid,
            "field_count": len(self._written_paths),
            "model_cache_hits": self._model_cache_hits,
            "model_cache_misses": self._model_cache_misses,
            "result_cache_hits": self._result_cache_hits,
            "runtime_seconds": self._segmentation_seconds,
            "segmentation_count": self._segmentation_count,
            "timings": _finalize_timings(aggregated, self._write_seconds),
        }
        self.root.store.close()
        self._closed = True
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        _install_store(self.temporary, self.output_path)
        return self.output_path

    def abort(self) -> None:
        if not self._closed:
            self.root.store.close()
            self._closed = True
        if self.temporary.exists():
            shutil.rmtree(self.temporary)


def write_hcs_plate(
    results: Iterable[LabelResult],
    output_path: str | Path,
    *,
    output_store_uuid: str | None = None,
) -> Path:
    result_list = list(results)
    if not result_list:
        raise ValueError("Cannot write an empty HCS result")
    writer = HCSPlateWriter(
        [result.source.resource for result in result_list],
        output_path,
        output_store_uuid=output_store_uuid,
    )
    try:
        for result in result_list:
            writer.append(result)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise
