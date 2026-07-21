from pathlib import Path

import numpy as np
import pytest

from cisegmentation.adapters import segment_czyx
from cisegmentation.ome_zarr_io import enumerate_resources, read_image
from cisegmentation.registry import get_model_spec
from cisegmentation.settings import SegmentationSettings


@pytest.mark.gpu
def test_local_refinement_bounds_real_spotiflow_masks():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Spotiflow fixture")
    root = Path(__file__).resolve().parents[1]
    spotiflow = root / "models" / "spotiflow" / "general" / "best.pt"
    if not spotiflow.is_file():
        pytest.skip("Pinned Spotiflow checkpoint is not downloaded")
    image = read_image(
        enumerate_resources(root / "tests" / "data" / "nuclei-spots-cytoplasm.ome.zarr")[0]
    )
    czyx = image.data[0]
    common = dict(
        model="spotiflow:general",
        target="spots",
        primary_channel=2,
        device="cuda",
        dimension_mode="slice-2d",
    )
    point_labels, _ = segment_czyx(
        czyx,
        get_model_spec("spotiflow:general"),
        SegmentationSettings(**common, spotiflow_local_refinement=False),
        image.scales,
    )
    masks, info = segment_czyx(
        czyx,
        get_model_spec("spotiflow:general"),
        SegmentationSettings(**common, spotiflow_local_refinement=True),
        image.scales,
    )

    points = np.argwhere(point_labels > 0)
    assert len(points) > 0
    assert masks.shape == point_labels.shape
    assert masks.dtype == np.uint32
    assert 0 < np.count_nonzero(masks) < 0.05 * masks.size
    assert masks.max() == len(points)
    assert all(masks[tuple(point)] > 0 for point in points)
    effective = info["effective_parameters"]
    assert effective["detected_points"] == len(points)
    assert effective["refined_masks"] == len(points)
    maximum_area = np.pi * effective["refinement_radius_y_pixels"] * effective[
        "refinement_radius_x_pixels"
    ]
    counts = np.bincount(masks.ravel())[1:]
    assert counts.max() <= np.ceil(maximum_area)
