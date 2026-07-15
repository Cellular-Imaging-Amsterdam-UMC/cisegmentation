from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
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


def _attrs(group) -> dict[str, Any]:
    return dict(group.attrs.asdict() if hasattr(group.attrs, "asdict") else group.attrs)


def discover_ome_zarrs(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if root.name.lower().endswith(".ome.zarr") and root.is_dir():
        return [root]
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.lower().endswith(".ome.zarr")
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
    axes = _axis_names(multiscale, raw.ndim)
    scales = _scale_map(multiscale, axes)
    return ImageData(
        _to_tczyx(raw, axes), axes, scales, attrs, resource, str(raw.dtype)
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


def _ome_xml(result: LabelResult, name: str) -> str:
    t, c, z, y, x = result.labels.shape
    px_x, px_y, px_z = (result.source.scales.get(axis, 1.0) for axis in ("x", "y", "z"))
    channel_names = result.channel_labels or [f"{result.target} labels"]
    channels = "".join(
        f'<Channel ID="Channel:0:{index}" Name="{channel_name}" SamplesPerPixel="1"/>'
        for index, channel_name in enumerate(channel_names)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0" Name="{name}"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
        f'Type="uint32" SizeX="{x}" SizeY="{y}" SizeZ="{z}" SizeC="{c}" SizeT="{t}" '
        f'PhysicalSizeX="{px_x}" PhysicalSizeY="{px_y}" PhysicalSizeZ="{px_z}">'
        f"{channels}"
        "</Pixels></Image></OME>"
    )


def _write_image_group(group, result: LabelResult, name: str) -> None:
    labels = np.asarray(result.labels, dtype=np.uint32)
    levels = [labels]
    while min(levels[-1].shape[-2:]) >= 512 and len(levels) < 5:
        levels.append(_downsample_labels(levels[-1]))
    datasets = []
    for index, level in enumerate(levels):
        chunks = (1, 1, 1, min(512, level.shape[-2]), min(512, level.shape[-1]))
        array = group.create_dataset(
            str(index), shape=level.shape, data=level, chunks=chunks, overwrite=True
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
    channel_names = result.channel_labels or [f"{result.target} labels"]
    group.attrs["omero"] = {
        "version": "0.4",
        "name": name,
        "channels": [
            {
                "label": channel_name,
                "color": f"{(index * 2654435761) & 0xFFFFFF:06X}",
                "active": index == 0,
                "window": {
                    "start": 0.0,
                    "end": float(labels[:, index].max(initial=0)),
                    "min": 0.0,
                    "max": float(labels[:, index].max(initial=0)),
                },
            }
            for index, channel_name in enumerate(channel_names)
        ],
        "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"},
    }
    group.attrs["cisegmentation"] = {
        "model": result.model_id,
        "target": result.target,
        "source": str(result.source.resource.store_path.name),
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
    (ome / "METADATA.ome.xml").write_text(_ome_xml(result, name), encoding="utf-8")


def write_label_image(result: LabelResult, output_path: str | Path) -> Path:
    import zarr

    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    root = zarr.open_group(str(temporary), mode="w", zarr_version=2)
    _write_image_group(root, result, output_path.name.removesuffix(".ome.zarr"))
    if output_path.exists():
        shutil.rmtree(output_path)
    temporary.replace(output_path)
    return output_path


def write_rgb_gallery(
    cyx: np.ndarray,
    source: ImageData,
    provenance: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a synthetic 2D RGB benchmark montage as NGFF 0.4/Zarr v2."""
    import zarr

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
    if output_path.exists():
        shutil.rmtree(output_path)
    temporary.replace(output_path)
    return output_path


def write_hcs_plate(results: Iterable[LabelResult], output_path: str | Path) -> Path:
    import zarr

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
        _write_image_group(group, result, f"{output_path.stem}_{row}_{column}_{field}")
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
    if output_path.exists():
        shutil.rmtree(output_path)
    temporary.replace(output_path)
    return output_path
