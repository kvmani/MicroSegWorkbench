"""Training configuration and presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from ..core.augment import AugmentConfig


class TrainConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TrainConfig:
    dataset_id: str
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    base_channels: int = 8
    depth: int = 2
    patches_per_image: int = 8
    seed: int = 1234
    device: str = "auto"
    num_threads: int = 0  # 0 = PyTorch default
    fixed_samples: int = 3
    normalization: str = "dtype_range"
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    preset: str = "custom"

    def __post_init__(self) -> None:
        checks = [
            (self.epochs >= 1, "epochs must be at least 1"),
            (self.batch_size >= 1, "batch size must be at least 1"),
            (0 < self.learning_rate < 1, "learning rate must be between 0 and 1"),
            (self.weight_decay >= 0, "weight decay must not be negative"),
            (self.base_channels >= 4, "base channels must be at least 4"),
            (1 <= self.depth <= 5, "depth must be between 1 and 5"),
            (self.patches_per_image >= 1, "patches per image must be at least 1"),
            (self.fixed_samples >= 0, "fixed samples must not be negative"),
            (self.normalization == "dtype_range", "only 'dtype_range' normalization is implemented"),
            (self.augment.patch_size % 2 ** self.depth == 0, "patch size must be divisible by 2^depth"),
        ]
        for ok, message in checks:
            if not ok:
                raise TrainConfigError(message)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["augment"] = self.augment.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        data = dict(payload)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TrainConfigError(f"Unknown training options: {sorted(unknown)}")
        if isinstance(data.get("augment"), dict):
            data["augment"] = AugmentConfig.from_dict(data["augment"])
        return cls(**data)


PRESETS: dict[str, dict[str, Any]] = {
    "quick_cpu": dict(epochs=15, batch_size=8, base_channels=8, depth=2, patches_per_image=8, learning_rate=3e-3,
                      description="Tiny model for a fast check on any computer (minutes on CPU)."),
    "standard": dict(epochs=40, batch_size=8, base_channels=16, depth=3, patches_per_image=8, learning_rate=1e-3,
                     description="Larger model and more epochs; intended for a GPU workstation."),
}


def preset_config(name: str, dataset_id: str, **overrides: Any) -> TrainConfig:
    if name not in PRESETS:
        raise TrainConfigError(f"Unknown preset {name!r}; choose one of {sorted(PRESETS)}.")
    values = {k: v for k, v in PRESETS[name].items() if k != "description"}
    values.update(overrides)
    return replace(TrainConfig(dataset_id=dataset_id, preset=name), **values)
