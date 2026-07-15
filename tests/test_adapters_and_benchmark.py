from pathlib import Path

import numpy as np

from cisegmentation.adapters import _segment_instanseg, points_to_labels
from cisegmentation.registry import get_model_spec
from cisegmentation.benchmark import _benchmark_specs, center_crop, run_benchmark
from cisegmentation.ome_zarr_io import enumerate_resources, read_image
from cisegmentation.settings import SegmentationSettings


def test_spotiflow_points_are_unique_single_pixels():
    points = np.array([[2.2, 3.7], [5.0, 6.0]])
    labels = points_to_labels(points, (10, 10))
    assert np.count_nonzero(labels) == 2
    assert set(np.unique(labels)) == {0, 1, 2}


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


def test_benchmark_writes_only_one_multichannel_ome_zarr(tmp_path, monkeypatch):
    import cisegmentation.benchmark as benchmark

    data = Path(__file__).parent / "data" / "nuclei-small.ome.zarr"
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
        target="nuclei", benchmark=True, benchmark_models="stardist:SD_Nuclei_Versatile"
    )
    output, failed = run_benchmark(image, settings, tmp_path)
    assert not failed
    assert output.suffixes[-2:] == [".ome", ".zarr"]
    assert [path.name for path in tmp_path.iterdir()] == [output.name]
    import zarr

    root = zarr.open_group(str(output), mode="r")
    assert root["0"].dtype == np.uint8
    assert root["0"].shape[:3] == (1, 3, 1)
    assert root.attrs["cisegmentation"]["layout"] == (
        "2d-xy-input-and-segmentation-panels"
    )


def test_all_benchmark_includes_spotiflow():
    settings = SegmentationSettings(target="nuclei", benchmark_models="all")
    ids = {spec.id for spec in _benchmark_specs(settings, 1)}
    assert "spotiflow:general" in ids
    assert "spotiflow:smfish_3d" in ids
