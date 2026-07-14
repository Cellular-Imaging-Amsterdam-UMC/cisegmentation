
from bilayers_cli import load_config, validate_config
from wrapper import build_parser


def test_bilayers_config_is_structurally_valid():
    assert validate_config(load_config()) == []


def test_wrapper_accepts_hyphenated_bilayers_parameters():
    args = build_parser().parse_args(
        [
            "--infolder",
            "in",
            "--outfolder",
            "out",
            "--model",
            "stardist:SD_Foci_Finn",
            "--target",
            "foci",
            "--primary-channel",
            "2",
            "--benchmark",
            "true",
        ]
    )
    assert args.model == "stardist:SD_Foci_Finn"
    assert args.primary_channel == 2
    assert args.benchmark is True
