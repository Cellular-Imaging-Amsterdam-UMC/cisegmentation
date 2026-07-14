from pathlib import Path

import numpy as np

from cisegmentation.adapters import points_to_labels
from cisegmentation.benchmark import center_crop, run_benchmark
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
