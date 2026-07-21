from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    checkpoint: str
    targets: tuple[str, ...]
    dimensions: str = "2d"
    min_channels: int = 1
    description: str = ""


_CP3_MODELS = (
    "cyto3",
    "nuclei",
    "cyto2_cp3",
    "tissuenet_cp3",
    "livecell_cp3",
    "yeast_PhC_cp3",
    "yeast_BF_cp3",
    "bact_phase_cp3",
    "bact_fluor_cp3",
    "deepbacs_cp3",
    "cyto2",
    "cyto",
    "CPx",
    "transformer_cp3",
    "neurips_cellpose_default",
    "neurips_cellpose_transformer",
    "neurips_grayscale_cyto2",
    "CP",
    "TN1",
    "TN2",
    "TN3",
    "LC1",
    "LC2",
    "LC3",
    "LC4",
)


def _build_registry() -> dict[str, ModelSpec]:
    specs: list[ModelSpec] = []
    for name in _CP3_MODELS:
        targets = ("nuclei",) if name == "nuclei" else ("cells",)
        specs.append(ModelSpec(f"cellpose3:{name}", "cellpose3", name, targets, "3d"))
    specs.extend(
        [
            ModelSpec(
                "cellpose-sam:cpsam_v2",
                "cellpose-sam",
                "cpsam_v2",
                ("nuclei", "cells"),
                "3d",
            ),
            ModelSpec(
                "cellpose-sam:cpsam",
                "cellpose-sam",
                "cpsam",
                ("nuclei", "cells"),
                "3d",
            ),
            ModelSpec(
                "stardist:SD_Nuclei_Versatile",
                "stardist",
                "SD_Nuclei_Versatile",
                ("nuclei",),
            ),
            ModelSpec(
                "stardist:SD_Foci_Aggregates",
                "stardist",
                "SD_Foci_Aggregates",
                ("foci",),
            ),
            ModelSpec("stardist:SD_Foci_Finn", "stardist", "SD_Foci_Finn", ("foci",)),
            ModelSpec(
                "instanseg:brightfield_nuclei",
                "instanseg",
                "brightfield_nuclei",
                ("nuclei",),
                min_channels=3,
            ),
            ModelSpec(
                "instanseg:fluorescence_nuclei_and_cells",
                "instanseg",
                "fluorescence_nuclei_and_cells",
                ("nuclei", "cells"),
            ),
            ModelSpec(
                "instanseg:single_channel_nuclei",
                "instanseg",
                "single_channel_nuclei",
                ("nuclei",),
            ),
        ]
    )
    for name in ("general", "hybiss", "synth_complex", "fluo_live"):
        specs.append(ModelSpec(f"spotiflow:{name}", "spotiflow", name, ("spots",)))
    for name in ("synth_3d", "smfish_3d"):
        specs.append(
            ModelSpec(f"spotiflow:{name}", "spotiflow", name, ("spots",), "3d")
        )
    return {spec.id: spec for spec in specs}


MODEL_REGISTRY = _build_registry()


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model {model_id!r}. Available models: {choices}"
        ) from exc


def eligible_benchmark_models(target: str, channel_count: int) -> list[ModelSpec]:
    return [
        spec
        for spec in MODEL_REGISTRY.values()
        if target in spec.targets and channel_count >= spec.min_channels
    ]
