from __future__ import annotations

from dataclasses import asdict, dataclass
import json


CELL_MODELS = (
    "cellpose3:cyto3",
    "cellpose-sam:cpsam",
    "instanseg:fluorescence_nuclei_and_cells",
)
STEP1_NUCLEUS_MODELS = (
    "cellpose3:nuclei",
    "cellpose-sam:cpsam",
    "stardist:SD_Nuclei_Versatile",
    "instanseg:single_channel_nuclei",
)
STEP2_NUCLEUS_MODELS = (
    *STEP1_NUCLEUS_MODELS,
    "instanseg:fluorescence_nuclei_and_cells",
)
FOCI_MODELS = (
    "spotiflow:general",
    "spotiflow:hybiss",
    "spotiflow:synth_complex",
    "spotiflow:synth_3d",
    "spotiflow:smfish_3d",
    "spotiflow:fluo_live",
    "stardist:SD_Foci_Aggregates",
    "stardist:SD_Foci_Finn",
    "cellpose3:bact_phase_cp3",
    "cellpose3:bact_fluor_cp3",
)

SKIP = "skip"
EXPANSION_PREFIX = "expand:"


@dataclass
class SegmentationSettings:
    model: str = "cellpose3:nuclei"
    target: str = "nuclei"
    primary_channel: int = 1
    nuclei_channel: int = 0
    device: str = "auto"
    dimension_mode: str = "auto"
    diameter: float = 0.0
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    stardist_prob_threshold: float = -1.0
    stardist_nms_threshold: float = -1.0
    smooth_stardist_labels: bool = True
    spotiflow_prob_threshold: float = -1.0
    spotiflow_min_distance: float = 1.0
    benchmark: bool = False
    cell_model: str = "cellpose3:cyto3"
    cell_channel: int = 1
    cell_nuclei_channel: int = 0
    cell_expansion_distance: float = 10.0
    nucleus_model: str = SKIP
    nucleus_channel: int = 1
    foci_model_1: str = SKIP
    foci_channel_1: int = 1
    foci_model_2: str = SKIP
    foci_channel_2: int = 1
    foci_model_3: str = SKIP
    foci_channel_3: int = 1
    foci_model_4: str = SKIP
    foci_channel_4: int = 1
    include_original_channels: bool = False
    write_ome_zarr_labels: bool = False
    remove_border_cells: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def selected_channels(self, channel_count: int) -> list[int]:
        channels = [self.primary_channel - 1]
        if self.nuclei_channel > 0 and self.nuclei_channel != self.primary_channel:
            channels.append(self.nuclei_channel - 1)
        if not channels or any(
            index < 0 or index >= channel_count for index in channels
        ):
            raise ValueError(
                f"Selected one-based channels are outside input channel count {channel_count}"
            )
        return channels

    def enabled_foci_steps(self) -> list[tuple[int, str, int]]:
        """Return enabled Step 3 slots as ``(slot, model, one-based channel)``."""
        return [
            (slot, getattr(self, f"foci_model_{slot}"), getattr(self, f"foci_channel_{slot}"))
            for slot in range(1, 5)
            if getattr(self, f"foci_model_{slot}") != SKIP
        ]

    def cell_expansion_model(self) -> str | None:
        if not self.cell_model.startswith(EXPANSION_PREFIX):
            return None
        return self.cell_model.removeprefix(EXPANSION_PREFIX)

    def cell_expansion_channel(self) -> int:
        """Use the explicit nucleus input for expansion, falling back to primary."""
        return self.cell_nuclei_channel or self.cell_channel

    def validate_steps(self) -> None:
        if not (
            self.cell_model != SKIP
            or self.nucleus_model != SKIP
            or self.enabled_foci_steps()
        ):
            raise ValueError(
                "Select at least one segmentation step: Cell Detection, "
                "Nuclei Detection, or a Foci Detection slot"
            )
        expansion_model = self.cell_expansion_model()
        if self.cell_model != SKIP and self.cell_model not in CELL_MODELS:
            if expansion_model is None:
                raise ValueError(f"Unknown Step 1 selection: {self.cell_model}")
            if not expansion_model:
                raise ValueError("Step 1 expansion selection must include a nucleus model")
            if expansion_model not in STEP1_NUCLEUS_MODELS:
                raise ValueError(f"Unknown Step 1 expansion model: {expansion_model}")
        if self.nucleus_model != SKIP and self.nucleus_model not in STEP2_NUCLEUS_MODELS:
            raise ValueError(f"Unknown Step 2 nucleus model: {self.nucleus_model}")
        for slot in range(1, 5):
            model = getattr(self, f"foci_model_{slot}")
            if model != SKIP and model not in FOCI_MODELS:
                raise ValueError(f"Unknown Step 3{chr(96 + slot)} model: {model}")
        if self.cell_nuclei_channel < 0:
            raise ValueError("Step 1 nucleus channel must be zero or greater")
        if self.cell_expansion_distance < 0:
            raise ValueError("Cell expansion distance must be zero or greater")


_LEGACY_FIELDS = {
    "cell_step",
    "cell_method",
    "cell_nuclei_model",
    "cell_expansion_nucleus_model",
    "nucleus_step",
    *(f"foci_step_{slot}" for slot in range(1, 5)),
}


def normalize_legacy_workflow_values(values: dict) -> dict:
    """Translate the former checkbox-based workflow into selector values."""
    normalized = dict(values)
    if not any(name in normalized for name in _LEGACY_FIELDS):
        return normalized

    def enabled(name: str, default: bool) -> bool:
        value = normalized.get(name, default)
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    cell_enabled = enabled("cell_step", True)
    method = normalized.get("cell_method", "deep-learning")
    if method not in {"deep-learning", "cell-expansion"}:
        raise ValueError(f"Unknown legacy cell detection method: {method}")
    if not cell_enabled:
        normalized["cell_model"] = SKIP
    elif method == "cell-expansion":
        seed_model = normalized.get(
            "cell_expansion_nucleus_model", "cellpose3:nuclei"
        )
        normalized["cell_model"] = f"{EXPANSION_PREFIX}{seed_model}"

    has_nucleus_flag = "nucleus_step" in normalized
    nucleus_enabled = enabled("nucleus_step", False)
    legacy_step1_nuclei = (
        cell_enabled
        and method != "cell-expansion"
        and int(normalized.get("cell_nuclei_channel", 1)) > 0
        and not nucleus_enabled
    )
    if legacy_step1_nuclei:
        normalized["nucleus_model"] = normalized.get(
            "cell_nuclei_model", "cellpose3:nuclei"
        )
        normalized["nucleus_channel"] = int(normalized.get("cell_nuclei_channel", 1))
    elif nucleus_enabled and "nucleus_model" not in normalized:
        normalized["nucleus_model"] = "cellpose3:nuclei"
    elif has_nucleus_flag and not nucleus_enabled:
        normalized["nucleus_model"] = SKIP

    for slot in range(1, 5):
        step_name = f"foci_step_{slot}"
        if step_name not in normalized:
            continue
        if enabled(step_name, False):
            normalized.setdefault(f"foci_model_{slot}", "spotiflow:general")
        else:
            normalized[f"foci_model_{slot}"] = SKIP

    for name in _LEGACY_FIELDS:
        normalized.pop(name, None)
    return normalized


def parse_model_selection(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [str(item).strip() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
