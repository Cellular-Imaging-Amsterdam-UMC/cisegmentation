from pathlib import Path

from bilayers_cli import load_config, validate_config
from wrapper import build_parser


def test_bilayers_config_is_structurally_valid():
    config = load_config()
    assert validate_config(config) == []
    parameters = {item["name"]: item for item in config["parameters"]}
    assert "instanseg_pixel_size_um" not in parameters
    assert parameters["diameter"]["mode"] == "advanced"
    spot_values = {
        option["value"] for option in parameters["benchmark_models"]["options"]
    }
    assert "spotiflow:general" in spot_values


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


def test_environment_bootstrap_installs_launcher_dependencies():
    bootstrap = Path("create_env.cmd").read_text(encoding="utf-8")
    assert "requirements_launcher.txt" in bootstrap
