import pytest

from cisegmentation.registry import (
    eligible_benchmark_models,
    get_model_spec,
)
from cisegmentation.settings import SegmentationSettings, parse_model_selection


def test_requested_custom_stardist_models_are_registered():
    assert get_model_spec("stardist:SD_Foci_Aggregates").targets == ("foci",)
    assert get_model_spec("stardist:SD_Foci_Finn").targets == ("foci",)


def test_foci_eligible_models_include_both_custom_models():
    ids = {spec.id for spec in eligible_benchmark_models("foci", 1)}
    assert {"stardist:SD_Foci_Aggregates", "stardist:SD_Foci_Finn"} <= ids


def test_channel_selection_is_one_based_and_validated():
    settings = SegmentationSettings(primary_channel=2, nuclei_channel=1)
    assert settings.selected_channels(3) == [1, 0]
    with pytest.raises(ValueError):
        SegmentationSettings(primary_channel=4).selected_channels(3)


def test_repeated_spot_channels_are_retained_and_validated():
    settings = SegmentationSettings(spot_channels="2, 2;3")
    assert settings.selected_spot_channels(3) == [1, 1, 2]
    assert SegmentationSettings(spot_channels=[2, 2]).selected_spot_channels(3) == [
        1,
        1,
    ]
    with pytest.raises(ValueError, match="outside input channel count"):
        SegmentationSettings(spot_channels="4").selected_spot_channels(3)


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
