from pathlib import Path

import numpy as np
import pytest

from cisegmentation.adapters import clear_model_cache, segment_czyx
from cisegmentation.ome_zarr_io import enumerate_resources, read_image
from cisegmentation.registry import get_model_spec
from cisegmentation.settings import SegmentationSettings


@pytest.mark.gpu
def test_high_magnification_stardist_fixture_uses_smoother_source_polygons():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the pinned StarDist fixture comparison")
    checkpoint = Path("models/stardist/SD_Nuclei_Versatile/SD_Nuclei_Versatile.pt")
    if not checkpoint.exists():
        pytest.skip("Downloaded StarDist model is required for this comparison")

    image = read_image(
        enumerate_resources(Path("tests/data/nuclei-spots-cytoplasm.ome.zarr"))[0]
    )
    spec = get_model_spec("stardist:SD_Nuclei_Versatile")
    common = {
        "model": spec.id,
        "target": "nuclei",
        "primary_channel": 1,
        "device": "cuda",
    }
    clear_model_cache()
    try:
        smooth, smooth_info = segment_czyx(
            image.data[0],
            spec,
            SegmentationSettings(**common, smooth_stardist_labels=True),
            image.scales,
        )
        nearest, nearest_info = segment_czyx(
            image.data[0],
            spec,
            SegmentationSettings(**common, smooth_stardist_labels=False),
            image.scales,
        )
    finally:
        clear_model_cache()

    assert len(np.unique(smooth)) - 1 == len(np.unique(nearest)) - 1 == 4
    assert smooth_info["effective_parameters"]["label_restoration"] == (
        "scaled-polygons"
    )
    assert nearest_info["effective_parameters"]["label_restoration"] == (
        "nearest-neighbor"
    )
    assert np.count_nonzero(smooth != nearest) > 0
    assert np.count_nonzero(smooth) < np.count_nonzero(nearest)
