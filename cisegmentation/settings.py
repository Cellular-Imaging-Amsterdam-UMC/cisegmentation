from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass
class SegmentationSettings:
    model: str = "cellpose3:nuclei"
    target: str = "nuclei"
    primary_channel: int = 1
    nuclei_channel: int = 0
    input_channels: str = ""
    device: str = "auto"
    dimension_mode: str = "auto"
    diameter: float = 0.0
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    stardist_prob_threshold: float = -1.0
    stardist_nms_threshold: float = -1.0
    spotiflow_prob_threshold: float = -1.0
    spotiflow_min_distance: int = 1
    benchmark: bool = False
    benchmark_models: str = "all"

    def to_dict(self) -> dict:
        return asdict(self)

    def selected_channels(self, channel_count: int) -> list[int]:
        if self.input_channels.strip():
            raw = self.input_channels.replace(";", ",").split(",")
            channels = [int(value.strip()) - 1 for value in raw if value.strip()]
        else:
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
