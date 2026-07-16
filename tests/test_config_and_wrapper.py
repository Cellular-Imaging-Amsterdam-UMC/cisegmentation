from pathlib import Path

from bilayers_cli import generate_cli_command, load_config, validate_config
from cisegmentation.settings import (
    CELL_MODELS,
    FOCI_MODELS,
    STEP1_NUCLEUS_MODELS,
    STEP2_NUCLEUS_MODELS,
)
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
    assert "multi_step" not in parameters
    assert parameters["cell_step"]["default"] is True
    assert parameters["nucleus_step"]["default"] is False
    assert all(parameters[f"foci_step_{slot}"]["default"] is False for slot in range(1, 5))
    assert parameters["include_original_channels"]["default"] is False
    assert parameters["remove_border_cells"]["default"] is True
    assert parameters["remove_border_cells"]["mode"] == "beginner"
    parameter_names = [item["name"] for item in config["parameters"]]
    assert parameter_names.index("remove_border_cells") + 1 == parameter_names.index(
        "foci_step_1"
    )
    assert parameters["include_original_channels"]["mode"] == "advanced"
    assert parameters["benchmark"]["mode"] == "advanced"
    advanced_names = [
        item["name"]
        for item in config["parameters"]
        if item.get("mode") == "advanced"
    ]
    assert advanced_names[:4] == [
        "foci_model_1",
        "foci_model_2",
        "foci_model_3",
        "foci_model_4",
    ]
    assert all(
        parameters[f"foci_model_{slot}"]["section_id"] == "advanced"
        for slot in range(1, 5)
    )
    spot_models = {option["value"] for option in parameters["foci_model_1"]["options"]}
    assert {
        "stardist:SD_Foci_Aggregates",
        "stardist:SD_Foci_Finn",
        "cellpose3:bact_phase_cp3",
        "cellpose3:bact_fluor_cp3",
    } <= spot_models
    assert "benchmark_models" not in parameters
    assert tuple(option["value"] for option in parameters["cell_model"]["options"]) == CELL_MODELS
    assert tuple(
        option["value"] for option in parameters["cell_nuclei_model"]["options"]
    ) == STEP1_NUCLEUS_MODELS
    assert tuple(
        option["value"] for option in parameters["nucleus_model"]["options"]
    ) == STEP2_NUCLEUS_MODELS
    assert tuple(
        option["value"] for option in parameters["foci_model_1"]["options"]
    ) == FOCI_MODELS


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


def test_bilayers_can_disable_default_cell_step():
    command = generate_cli_command(
        load_config(), {"cell_step": False, "nucleus_step": True}
    )
    assert "--cell-step False" in command
    assert "--nucleus-step True" in command


def test_environment_bootstrap_installs_launcher_dependencies():
    bootstrap = Path("create_env.cmd").read_text(encoding="utf-8")
    assert "requirements_launcher.txt" in bootstrap
