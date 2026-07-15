import numpy as np

import cisegmentation.engine as engine
from cisegmentation.ome_zarr_io import enumerate_resources, read_image
from cisegmentation.settings import SegmentationSettings


def test_multitimepoint_cache_and_timings_are_aggregated(inputfolder, monkeypatch):
    image = read_image(
        enumerate_resources(inputfolder / "nuclei-small.ome.zarr")[0]
    )
    image.data = np.repeat(image.data, 2, axis=0)
    calls = 0

    def fake_segment(czyx, spec, settings, scales):
        nonlocal calls
        cache_hit = calls > 0
        calls += 1
        return np.ones(czyx.shape[1:], dtype=np.uint32), {
            "device": "cuda",
            "dimension_mode": "slice-2d",
            "runtime_seconds": 1.0,
            "object_count": 1,
            "model_cache_hit": cache_hit,
            "timings": {
                "import_seconds": 0.5 if not cache_hit else 0.0,
                "device_setup_seconds": 0.1,
                "model_load_seconds": 0.25 if not cache_hit else 0.0,
                "inference_seconds": 0.75,
            },
        }

    monkeypatch.setattr(engine, "segment_czyx", fake_segment)
    result = engine._segment_image(
        image,
        SegmentationSettings(),
        startup_seconds=0.2,
        zarr_read_seconds=0.3,
    )

    assert result.provenance["model_cache_misses"] == 1
    assert result.provenance["model_cache_hits"] == 1
    assert result.provenance["runtime_seconds"] == 2.0
    assert result.provenance["timings"] == {
        "device_setup_seconds": 0.2,
        "import_seconds": 0.5,
        "inference_seconds": 1.5,
        "model_load_seconds": 0.25,
        "startup_seconds": 0.2,
        "zarr_read_seconds": 0.3,
    }
