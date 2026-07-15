import numpy as np
import pytest

from cisegmentation.adapters import (
    _cached_model,
    _cellpose_diameter_pixels,
    _segment_instanseg,
    _spotiflow_min_distance_pixels,
    _stardist_versatile_input,
    clear_model_cache,
    points_to_labels,
)
from cisegmentation.registry import get_model_spec
from cisegmentation.benchmark import _benchmark_specs, center_crop, run_benchmark
from cisegmentation.ome_zarr_io import enumerate_resources, read_image
from cisegmentation.settings import SegmentationSettings


def test_process_model_cache_is_keyed_by_model_and_device():
    clear_model_cache()
    imports = []
    constructions = []

    def importer():
        imports.append(True)
        return object()

    def constructor(imported):
        constructions.append(imported)
        return object()

    first, first_timing = _cached_model(
        "cellpose3:cyto3", "cuda", importer, constructor
    )
    second, second_timing = _cached_model(
        "cellpose3:cyto3", "cuda", importer, constructor
    )
    cpu, cpu_timing = _cached_model(
        "cellpose3:cyto3", "cpu", importer, constructor
    )

    assert first is second
    assert cpu is not first
    assert len(imports) == len(constructions) == 2
    assert not first_timing["model_cache_hit"]
    assert second_timing == {
        "model_cache_hit": True,
        "import_seconds": 0.0,
        "model_load_seconds": 0.0,
    }
    assert not cpu_timing["model_cache_hit"]
    clear_model_cache()


def test_spotiflow_points_are_unique_single_pixels():
    points = np.array([[2.2, 3.7], [5.0, 6.0]])
    labels = points_to_labels(points, (10, 10))
    assert np.count_nonzero(labels) == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_physical_parameters_are_converted_from_ome_zarr_pixel_size():
    scales = {"y": 0.5, "x": 0.5}
    assert _cellpose_diameter_pixels(0, "nuclei", scales) == 24
    assert _cellpose_diameter_pixels(0, "cells", scales) == 50
    assert _cellpose_diameter_pixels(-1, "nuclei", scales) is None
    assert _spotiflow_min_distance_pixels(2.0, scales) == 4


def test_stardist_versatile_downsamples_finer_than_half_micron():
    image = np.zeros((80, 100), dtype=np.uint16)
    resized, original_shape = _stardist_versatile_input(
        image, {"y": 0.25, "x": 0.4}
    )
    assert original_shape == (80, 100)
    assert resized.shape == (40, 80)

    unchanged, _ = _stardist_versatile_input(image, {"y": 0.5, "x": 0.7})
    assert unchanged.shape == image.shape


def test_center_crop_is_centered_and_at_most_1024():
    image = np.zeros((2, 3, 1300, 1100), dtype=np.uint8)
    cropped, info = center_crop(image)
    assert cropped.shape == (2, 3, 1024, 1024)
    assert info == {"x": 38, "y": 138, "width": 1024, "height": 1024}


def test_instanseg_requires_metadata_pixel_size():
    with np.testing.assert_raises_regex(ValueError, "OME-Zarr metadata"):
        _segment_instanseg(
            np.zeros((1, 1, 8, 8), dtype=np.uint8),
            get_model_spec("instanseg:single_channel_nuclei"),
            SegmentationSettings(),
            float("nan"),
        )


def test_benchmark_writes_only_one_multichannel_ome_zarr(
    inputfolder, outputfolder, monkeypatch
):
    import cisegmentation.benchmark as benchmark

    data = inputfolder / "nuclei-small.ome.zarr"
    image = read_image(enumerate_resources(data)[0])
    monkeypatch.setattr(
        benchmark,
        "segment_czyx",
        lambda array, spec, settings, scales: (
            np.ones(array.shape[1:], dtype=np.uint32),
            {"runtime_seconds": 0.1, "object_count": 1, "device": "cpu"},
        ),
    )
    settings = SegmentationSettings(
        target="nuclei", benchmark=True, benchmark_models="stardist"
    )
    output, failed = run_benchmark(image, settings, outputfolder)
    assert not failed
    assert output.suffixes[-2:] == [".ome", ".zarr"]
    assert [path.name for path in outputfolder.iterdir()] == [output.name]
    import zarr

    root = zarr.open_group(str(output), mode="r")
    assert root["0"].dtype == np.uint8
    assert root["0"].shape[:3] == (1, 3, 1)
    assert root.attrs["cisegmentation"]["layout"] == (
        "2d-xy-input-and-segmentation-panels"
    )
    assert [run["target"] for run in root.attrs["cisegmentation"]["runs"]] == [
        "nuclei",
        "foci",
        "foci",
    ]


def test_all_benchmark_includes_spotiflow():
    settings = SegmentationSettings(target="nuclei", benchmark_models="all")
    ids = {spec.id for spec in _benchmark_specs(settings, 1)}
    assert "spotiflow:general" in ids
    assert "spotiflow:smfish_3d" in ids


@pytest.mark.parametrize(
    "preset,expected_families,expected_count",
    [
        (
            "all",
            {"cellpose3", "cellpose-sam", "stardist", "instanseg", "spotiflow"},
            38,
        ),
        ("cellpose", {"cellpose-sam"}, 1),
        ("cellpose3", {"cellpose3"}, 25),
        ("stardist", {"stardist"}, 3),
        ("instanseg", {"instanseg"}, 3),
        ("spotiflow", {"spotiflow"}, 6),
    ],
)
def test_benchmark_family_presets(preset, expected_families, expected_count):
    settings = SegmentationSettings(target="nuclei", benchmark_models=preset)
    specs = _benchmark_specs(settings, 1)
    assert {spec.family for spec in specs} == expected_families
    assert len(specs) == expected_count
