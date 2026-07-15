from pathlib import Path

from bilayers_cli import load_config, validate_config
from wrapper import build_parser


def test_bilayers_config_is_structurally_valid():
    config = load_config()
    assert validate_config(config) == []
    parameters = {item["name"]: item for item in config["parameters"]}
    assert "instanseg_pixel_size_um" not in parameters
    assert "input_channels" not in parameters
    assert parameters["diameter"]["mode"] == "advanced"
    assert parameters["diameter"]["minimum"] == -1.0
    assert parameters["spotiflow_min_distance"]["type"] == "float"
    assert parameters["multi_step"]["type"] == "checkbox"
    assert parameters["spot_channels"]["default"] == "2"
    assert parameters["remove_border_cells"]["default"] is False
    spot_models = {option["value"] for option in parameters["spot_model"]["options"]}
    assert {
        "stardist:SD_Foci_Aggregates",
        "stardist:SD_Foci_Finn",
        "cellpose3:bact_phase_cp3",
        "cellpose3:bact_fluor_cp3",
    } <= spot_models
    benchmark = parameters["benchmark_models"]
    assert benchmark["multiselect"] is False
    assert [option["value"] for option in benchmark["options"]] == [
        "all",
        "cellpose",
        "cellpose3",
        "stardist",
        "instanseg",
        "spotiflow",
    ]
    assert [option["label"] for option in benchmark["options"]] == [
        "All Algorithms, All Models",
        "Cellpose (SAM), All Models",
        "Cellpose 3 (Legacy), All Models",
        "StarDist, All Models",
        "InstanSeg, All Models",
        "Spotiflow, All Models",
    ]


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
