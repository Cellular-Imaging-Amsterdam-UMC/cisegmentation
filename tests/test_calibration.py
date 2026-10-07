"""Calibration precedence, OME-XML image matching and metadata-only preflight."""

from copy import deepcopy
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import zarr

from cisegmentation.engine import run_workflow
from cisegmentation.measurement_extensions import calibration
from cisegmentation.ome_zarr_io import ImageResource, enumerate_resources, read_image
from cisegmentation.settings import SegmentationSettings


def make_image(group, shape=(1, 1, 1, 8, 8), units=False):
    group.create_dataset("s0", shape=shape, chunks=(1, 1, 1, 8, 8), dtype="u2")
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": a, **({"unit": "micrometer"} if units and a in "zyx" else {})}
                for a in "tczyx"
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, 1, 1]}
                    ],
                }
            ],
        }
    ]


def write_xml(path, images, plate=None, unit="µm"):
    ome = ET.Element("OME", xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06")
    if plate is not None:
        ome.append(plate)
    for image_id, xy, shape in images:
        image = ET.SubElement(ome, "Image", ID=image_id)
        attrs = {f"Size{a.upper()}": str(v) for a, v in zip("tczyx", shape)}
        for axis in "XY":
            attrs[f"PhysicalSize{axis}"] = str(xy)
            if unit is not None:
                attrs[f"PhysicalSize{axis}Unit"] = unit
        ET.SubElement(image, "Pixels", attrs)
    folder = path / "OME"
    folder.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ome).write(
        folder / "METADATA.ome.xml", encoding="utf-8", xml_declaration=True
    )


def standalone(tmp_path, *, units=False, shape=(1, 1, 1, 8, 8), name="source"):
    path = tmp_path / f"{name}.ome.zarr"
    make_image(zarr.open_group(str(path), mode="w"), shape, units)
    return ImageResource(path)


@pytest.mark.parametrize(
    "unit,value", [("µm", 0.345), ("μm", 0.345), ("nm", 345), (None, 0.345)]
)
@pytest.mark.parametrize("lazy", [False, True])
def test_xml_fallback_units_and_input_unchanged(tmp_path, unit, value, lazy):
    resource = standalone(tmp_path)
    write_xml(resource.store_path, [("Image:0", value, (1, 1, 1, 8, 8))], unit=unit)
    original = (resource.store_path / ".zattrs").read_bytes()
    image = read_image(resource, lazy=lazy)
    assert image.scales["x"] == pytest.approx(0.345)
    assert image.scales["y"] == pytest.approx(0.345)
    assert set(image.calibration_sources) == {"x", "y"}
    valid, scales, offset, *_ = calibration(image)
    assert valid
    assert scales == pytest.approx([1, 0.345, 0.345])
    assert offset == pytest.approx([0, 0, 0])
    assert (resource.store_path / ".zattrs").read_bytes() == original


def test_ngff_precedence_and_per_axis_fallback(tmp_path):
    resource = standalone(tmp_path, units=True)
    write_xml(resource.store_path, [("Image:0", 0.345, (1, 1, 1, 8, 8))])
    root = zarr.open_group(str(resource.store_path), mode="a")
    attrs = root.attrs["multiscales"]
    attrs[0]["datasets"][0]["coordinateTransformations"][0]["scale"][-2:] = [0.5, 0.75]
    root.attrs["multiscales"] = attrs
    image = read_image(resource, lazy=True)
    assert not image.calibration_sources
    assert image.scales["y"] == 0.5
    assert image.scales["x"] == 0.75
    attrs[0]["axes"][-1].pop("unit")
    root.attrs["multiscales"] = attrs
    image = read_image(resource, lazy=True)
    assert set(image.calibration_sources) == {"x"}
    assert image.scales["y"] == 0.5
    assert image.scales["x"] == pytest.approx(0.345)
    assert calibration(image)[1] == pytest.approx([1, 0.5, 0.345])


def test_fallback_without_transforms_or_axis_metadata(tmp_path):
    resource = standalone(tmp_path)
    write_xml(resource.store_path, [("Image:0", 0.345, (1, 1, 1, 8, 8))])
    root = zarr.open_group(str(resource.store_path), mode="a")
    attrs = root.attrs["multiscales"]
    attrs[0].pop("axes")
    attrs[0]["datasets"][0].pop("coordinateTransformations")
    root.attrs["multiscales"] = attrs
    image = read_image(resource, lazy=True)
    assert image.scales["x"] == pytest.approx(0.345)
    assert calibration(image)[0]


@pytest.mark.parametrize("invalid_scale", [0, -1, float("nan"), float("inf")])
def test_xml_recovers_unusable_ngff_pixel_sizes(tmp_path, invalid_scale):
    resource = standalone(tmp_path, units=True)
    write_xml(resource.store_path, [("Image:0", 0.345, (1, 1, 1, 8, 8))])
    root = zarr.open_group(str(resource.store_path), mode="a")
    attrs = root.attrs["multiscales"]
    attrs[0]["datasets"][0]["coordinateTransformations"][0]["scale"][-2:] = [
        invalid_scale,
        invalid_scale,
    ]
    root.attrs["multiscales"] = attrs
    image = read_image(resource, lazy=True)
    assert image.scales["x"] == pytest.approx(0.345)
    assert image.scales["y"] == pytest.approx(0.345)
    assert calibration(image)[0]


def test_xml_xyz_calibration_is_usable_for_3d(tmp_path):
    resource = standalone(tmp_path, shape=(1, 1, 2, 8, 8))
    write_xml(resource.store_path, [("Image:0", 0.345, (1, 1, 2, 8, 8))])
    path = resource.store_path / "OME" / "METADATA.ome.xml"
    xml = ET.parse(path)
    pixels = xml.getroot().find("{*}Image/{*}Pixels")
    pixels.set("PhysicalSizeZ", "2000")
    pixels.set("PhysicalSizeZUnit", "nm")
    xml.write(path, encoding="utf-8")
    image = read_image(resource, lazy=True)
    assert set(image.calibration_sources) == {"x", "y", "z"}
    assert calibration(image)[0]
    assert calibration(image)[1] == pytest.approx([2, 0.345, 0.345])


def test_shared_transforms_units_offsets_and_pyramid_are_preserved(tmp_path):
    resource = standalone(tmp_path, units=True)
    write_xml(resource.store_path, [("Image:0", 0.345, (1, 1, 1, 8, 8))])
    root = zarr.open_group(str(resource.store_path), mode="a")
    attrs = root.attrs["multiscales"]
    ms = attrs[0]
    ms["axes"][-1].pop("unit")
    ms["axes"][-2]["unit"] = "nanometer"
    ms["datasets"][0]["coordinateTransformations"] = [
        {"type": "scale", "scale": [1, 1, 1, 250, 1]},
        {"type": "translation", "translation": [0, 0, 0, 1000, 2]},
    ]
    ms["coordinateTransformations"] = [{"type": "scale", "scale": [1, 1, 1, 2, 2]}]
    level = deepcopy(ms["datasets"][0])
    level["path"] = "s1"
    level["coordinateTransformations"][0]["scale"][-2:] = [500, 2]
    ms["datasets"].append(level)
    root.attrs["multiscales"] = attrs
    image = read_image(resource, lazy=True)
    assert image.scales["y"] == 0.5
    assert image.scales["x"] == pytest.approx(0.345)
    assert calibration(image)[1] == pytest.approx([1, 0.5, 0.345])
    assert calibration(image)[2] == pytest.approx([0, 2, 0.69])
    assert image.attrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][0][
        "scale"
    ][-1] == pytest.approx(0.69)


def test_hcs_matches_well_sample_image_refs_not_xml_image_order(tmp_path):
    path = tmp_path / "plate.ome.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs["plate"] = {
        "rows": [{"name": "A"}, {"name": "B"}],
        "columns": [{"name": "1"}],
        "wells": [{"path": "A/1"}, {"path": "B/1"}],
    }
    plate = ET.Element("Plate", ID="Plate:0")
    images = []
    for row, name in enumerate(["A", "B"]):
        well_group = root.require_group(f"{name}/1")
        well_group.attrs["well"] = {"images": [{"path": "0"}, {"path": "1"}]}
        well = ET.SubElement(plate, "Well", Row=str(row), Column="0")
        for field in [1, 0]:
            image_id = f"Image:{row}:{field}"
            sample = ET.SubElement(well, "WellSample", Index=str(row * 2 + field))
            ET.SubElement(sample, "ImageRef", ID=image_id)
            make_image(well_group.require_group(str(field)))
            images.append((image_id, 0.345 + row + field / 10, (1, 1, 1, 8, 8)))
    write_xml(path, images, plate)
    for resource in enumerate_resources(path):
        image = read_image(resource, lazy=True)
        row, _, field = resource.plate_path
        assert image.scales["x"] == pytest.approx(
            0.345 + (row == "B") + int(field) / 10
        )
        assert f"Image:{int(row == 'B')}:{field}" in image.calibration_sources["x"]


@pytest.mark.parametrize(
    "value,unit",
    [(0, "µm"), (-1, "µm"), ("nan", "µm"), ("inf", "µm"), (0.345, "pixel")],
)
def test_unusable_xml_does_not_supply_calibration(tmp_path, value, unit):
    resource = standalone(tmp_path)
    write_xml(resource.store_path, [("Image:0", value, (1, 1, 1, 8, 8))], unit=unit)
    assert not calibration(read_image(resource, lazy=True))[0]


def test_malformed_xml_and_wrong_series_dimensions_do_not_supply_calibration(tmp_path):
    resource = standalone(tmp_path)
    write_xml(resource.store_path, [("Image:0", 0.345, (1, 1, 1, 16, 16))])
    assert not calibration(read_image(resource, lazy=True))[0]
    (resource.store_path / "OME" / "METADATA.ome.xml").write_text(
        "<OME>", encoding="utf-8"
    )
    assert not calibration(read_image(resource, lazy=True))[0]


@pytest.mark.parametrize("feature", ["spatial_measurements", "tracking"])
@pytest.mark.parametrize("missing_z", [False, True])
def test_invalid_later_input_fails_before_inference_or_pixel_reads(
    tmp_path, monkeypatch, feature, missing_z
):
    from cisegmentation import engine

    standalone(tmp_path, units=True, name="a-valid")
    invalid = standalone(
        tmp_path, name="b-invalid", shape=(1, 1, 2 if missing_z else 1, 8, 8)
    )
    if missing_z:
        write_xml(invalid.store_path, [("Image:0", 0.345, (1, 1, 2, 8, 8))])

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Preflight must fail before inference and without reading image chunks"
        )

    monkeypatch.setattr(engine, "_run_parallel_store", forbidden)
    monkeypatch.setattr(zarr.Array, "__getitem__", forbidden)
    settings = SegmentationSettings(measurements_database="sqlite", **{feature: True})
    output = tmp_path / "output"
    with pytest.raises(
        ValueError, match=r"b-invalid.*NGFF metadata or embedded OME-XML"
    ) as error:
        run_workflow(tmp_path, output, settings)
    assert ("X/Y/Z" if missing_z else "X/Y") in str(error.value)
    assert not list(output.iterdir())


def test_xml_fallback_workflow_uses_calibration_for_models_and_measurements(
    tmp_path, monkeypatch
):
    import sqlite3

    import cisegmentation.parallel_pipeline as pipeline

    resource = standalone(tmp_path)
    write_xml(resource.store_path, [("Image:0", 0.345, (1, 1, 1, 8, 8))])
    seen = []

    def fake_spots(data, spec, settings, scales, **kwargs):
        seen.append(scales["x"])
        labels = np.zeros(data.shape[1:], dtype="u4")
        labels[0, 1, 1], labels[0, 1, 4] = 1, 2
        return labels, {
            "device": "cpu",
            "runtime_seconds": 0.01,
            "timings": {"inference_seconds": 0.01},
            "effective_parameters": {},
            "model_cache_hits": 0,
            "model_cache_misses": 1,
            "_native_points": np.array([[1, 0, 1, 1], [2, 0, 1, 4]]),
        }

    monkeypatch.setenv("CISEGMENTATION_INLINE_WORKERS", "1")
    monkeypatch.setattr(pipeline, "segment_czyx", fake_spots)
    settings = SegmentationSettings(
        cell_model="skip",
        nucleus_model="skip",
        foci_model_1="spotiflow:general",
        measurements_database="sqlite",
        spatial_measurements=True,
        tracking=True,
        max_inference_workers=1,
        max_measurement_workers=1,
    )
    logs = []
    outputs = run_workflow(
        resource.store_path, tmp_path / "output", settings, log=logs.append
    )
    assert seen and seen[0] == pytest.approx(0.345)
    assert any("OME-XML fallback used for 1 field(s)" in line for line in logs)
    database = next(p for p in outputs if "_measurements." in p.name)
    with sqlite3.connect(database) as db:
        distances = db.execute(
            "SELECT nearest_distance_um FROM spatial_measurements"
        ).fetchall()
        assert [row[0] for row in distances] == pytest.approx([1.035, 1.035])
        assert db.execute(
            "SELECT scale_x_um,scale_y_um FROM images"
        ).fetchone() == pytest.approx((0.345, 0.345))
