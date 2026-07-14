import pytest

from cisegmentation.registry import (
    eligible_benchmark_models,
    get_model_spec,
)
from cisegmentation.settings import SegmentationSettings, parse_model_selection


def test_requested_custom_stardist_models_are_registered():
    assert get_model_spec("stardist:SD_Foci_Aggregates").targets == ("foci",)
    assert get_model_spec("stardist:SD_Foci_Finn").targets == ("foci",)


def test_all_foci_benchmark_includes_both_custom_models_and_excludes_spotiflow():
    ids = {spec.id for spec in eligible_benchmark_models("foci", 1)}
    assert {"stardist:SD_Foci_Aggregates", "stardist:SD_Foci_Finn"} <= ids
    assert not any(model_id.startswith("spotiflow:") for model_id in ids)


def test_channel_selection_is_one_based_and_validated():
    settings = SegmentationSettings(primary_channel=2, nuclei_channel=1)
    assert settings.selected_channels(3) == [1, 0]
    assert SegmentationSettings(input_channels="1,3").selected_channels(3) == [0, 2]
    with pytest.raises(ValueError):
        SegmentationSettings(primary_channel=4).selected_channels(3)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("all", ["all"]),
        ("a,b", ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        (["a", "b"], ["a", "b"]),
    ],
)
def test_parse_model_selection(value, expected):
    assert parse_model_selection(value) == expected
