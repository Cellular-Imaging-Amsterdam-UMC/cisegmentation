from __future__ import annotations

import numpy as np
import pytest

import cisegmentation.adapters as adapters
from cisegmentation.registry import get_model_spec
from cisegmentation.settings import SegmentationSettings


def test_spot_smoothing_is_finite_and_preserves_shape():
    image = np.zeros((7, 9), dtype=np.float32)
    image[3, 4] = 16
    image[0, 0] = np.nan
    smoothed = adapters._smooth_spot_plane(image)
    assert smoothed.shape == image.shape
    assert np.isfinite(smoothed).all()
    assert smoothed[3, 4] > smoothed[3, 3] > 0


def test_local_candidate_is_seed_connected_and_bounded():
    yy, xx = np.ogrid[:31, :31]
    image = 10 + 200 * np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / 5)
    candidate = adapters._local_spot_candidate(
        adapters._smooth_spot_plane(image), np.array([15.0, 15.0]), 5.0, 4.0
    )
    mask = candidate["mask"]
    y0, x0, _y1, _x1 = candidate["bbox"]
    assert mask[15 - y0, 15 - x0]
    rows, columns = np.where(mask)
    assert np.all(((rows + y0 - 15) / 5.0) ** 2 + ((columns + x0 - 15) / 4.0) ** 2 <= 1)
    assert np.count_nonzero(mask) > 1
    assert candidate["fallback"] is False


def test_weak_seed_falls_back_to_one_pixel():
    image = np.arange(225, dtype=np.float32).reshape(15, 15)
    candidate = adapters._local_spot_candidate(image, np.array([7.0, 7.0]), 3.0, 3.0)
    assert candidate["fallback"] is True
    assert np.count_nonzero(candidate["mask"]) == 1


def test_candidate_merge_resolves_overlap_by_score():
    first = {
        "mask": np.ones((5, 5), dtype=bool),
        "bbox": (2, 2, 7, 7),
        "seed_yx": (4, 4),
        "score": 10.0,
        "fallback": False,
    }
    second = {
        "mask": np.ones((5, 5), dtype=bool),
        "bbox": (4, 4, 9, 9),
        "seed_yx": (7, 7),
        "score": 5.0,
        "fallback": False,
    }
    labels, summary = adapters._merge_local_spot_candidates([second, first], (12, 12))
    assert labels[4, 4] == 1
    assert labels[7, 7] == 2
    assert summary["refined_masks"] == 2
    assert summary["overlap_pixels_removed"] > 0


def test_refinement_handles_empty_points_without_models():
    labels, summary, timing = adapters._refine_spotiflow_points(
        np.zeros((2, 8, 9), dtype=np.uint16),
        [np.empty((0, 2)), np.empty((0, 2))],
        {"x": 0.5, "y": 0.5},
    )
    assert labels.shape == (2, 8, 9)
    assert summary["detected_points"] == 0
    assert summary["refined_masks"] == 0
    assert timing["local_refinement_seconds"] >= 0


def test_spotiflow_refinement_replaces_points_with_local_masks(monkeypatch):
    class FakeSpotiflow:
        _prob_thresh = [0.5]

        @staticmethod
        def predict(image, **_kwargs):
            return np.array([[8.0, 8.0], [20.0, 20.0]], dtype=np.float32), None

    monkeypatch.setattr(
        adapters,
        "_cached_model",
        lambda *_args, **_kwargs: (
            FakeSpotiflow(),
            {"model_cache_hit": False, "import_seconds": 0.1, "model_load_seconds": 0.2},
        ),
    )
    yy, xx = np.ogrid[:30, :30]
    image = (
        10
        + 500 * np.exp(-((yy - 8) ** 2 + (xx - 8) ** 2) / 3)
        + 400 * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / 4)
    ).astype(np.uint16)
    labels, timing = adapters._segment_spotiflow(
        image[None, None],
        get_model_spec("spotiflow:general"),
        SegmentationSettings(
            model="spotiflow:general",
            target="spots",
            spotiflow_local_refinement=True,
        ),
        "cpu",
        {"x": 0.5, "y": 0.5},
    )
    assert labels.shape == (1, 30, 30)
    assert labels.max() == 2
    assert np.count_nonzero(labels) > 2
    assert timing["locations_only"] is False
    assert timing["model_cache_misses"] == 1
    effective = timing["effective_parameters"]
    assert effective["refinement_method"] == "bounded-local-intensity"
    assert effective["detected_points"] == 2
    assert effective["grown_masks"] == 2
    assert effective["refinement_max_radius_um"] == 1.0


def test_disabled_refinement_preserves_single_pixel_output(monkeypatch):
    class FakeSpotiflow:
        _prob_thresh = [0.5]

        @staticmethod
        def predict(*_args, **_kwargs):
            return np.array([[2.2, 3.7], [5.0, 6.0]], dtype=np.float32), None

    monkeypatch.setattr(
        adapters,
        "_cached_model",
        lambda *_args, **_kwargs: (
            FakeSpotiflow(),
            {"model_cache_hit": False, "import_seconds": 0.0, "model_load_seconds": 0.0},
        ),
    )
    labels, timing = adapters._segment_spotiflow(
        np.zeros((1, 1, 10, 10), dtype=np.uint16),
        get_model_spec("spotiflow:general"),
        SegmentationSettings(model="spotiflow:general", target="spots"),
        "cpu",
        {"x": 0.5, "y": 0.5},
    )
    expected = adapters.points_to_labels(
        np.array([[2.2, 3.7], [5.0, 6.0]], dtype=np.float32), (10, 10)
    )[None]
    np.testing.assert_array_equal(labels, expected)
    assert timing["locations_only"] is True


def test_native_3d_spotiflow_refinement_requires_slice_mode():
    with pytest.raises(ValueError, match="Force slice-wise 2D"):
        adapters._segment_spotiflow(
            np.zeros((1, 3, 8, 8), dtype=np.uint16),
            get_model_spec("spotiflow:synth_3d"),
            SegmentationSettings(
                model="spotiflow:synth_3d",
                target="spots",
                spotiflow_local_refinement=True,
            ),
            "cpu",
            {"x": 0.5, "y": 0.5},
        )
