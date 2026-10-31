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
    batch_size: int = 8  # 0 = automatic (largest that fits in GPU memory, max 64; 8 on CPU)
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
    loss: str = "cross_entropy_dice"  # cross_entropy | cross_entropy_dice (Dice helps rare classes)
    mixed_precision: str = "auto"  # auto = on for CUDA GPUs, off on CPU; "on" / "off" force it
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    preset: str = "custom"

    def __post_init__(self) -> None:
        checks = [
            (self.epochs >= 1, "epochs must be at least 1"),
            (self.batch_size >= 0, "batch size must be 0 (automatic) or more"),
            (self.loss in ("cross_entropy", "cross_entropy_dice"), "loss must be cross_entropy or cross_entropy_dice"),
            (self.mixed_precision in ("auto", "on", "off"), "mixed precision must be auto, on or off"),
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
    "quick_cpu": dict(epochs=15, batch_size=8, base_channels=16, depth=2, patches_per_image=8, learning_rate=3e-3,
                      description="Tiny model for a fast check on any computer (about a minute on CPU for small datasets)."),
    "standard": dict(epochs=40, batch_size=0, base_channels=16, depth=3, patches_per_image=8, learning_rate=1e-3,
                     description="Larger model and more epochs for a GPU workstation; batch size is chosen "
                                 "automatically to fit GPU memory."),
    "large_gpu": dict(epochs=80, batch_size=0, base_channels=32, depth=4, patches_per_image=16, learning_rate=1e-3,
                      patch_size=256,
                      description="Biggest model for GPUs with 8 GB or more; uses 256 px patches. Slow on CPU."),
}


def preset_config(name: str, dataset_id: str, **overrides: Any) -> TrainConfig:
    if name not in PRESETS:
        raise TrainConfigError(f"Unknown preset {name!r}; choose one of {sorted(PRESETS)}.")
    values = {k: v for k, v in PRESETS[name].items() if k != "description"}
    patch = values.pop("patch_size", None)
    if patch is not None:
        values["augment"] = replace(AugmentConfig(), patch_size=patch)
    values.update(overrides)
    return replace(TrainConfig(dataset_id=dataset_id, preset=name), **values)
