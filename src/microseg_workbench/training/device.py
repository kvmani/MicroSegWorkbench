"""Compute device discovery with a visible, explicit CPU fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda")


class DeviceError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    kind: str
    name: str
    index: int | None = None
    total_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        if self.kind == "cuda":
            return f"GPU {self.index}: {self.name} ({self.total_memory_mb:.0f} MB)"
        return f"CPU: {self.name}"


def _cpu_info() -> DeviceInfo:
    import platform

    return DeviceInfo(kind="cpu", name=platform.processor() or platform.machine() or "CPU")


def detect_devices() -> list[DeviceInfo]:
    devices = [_cpu_info()]
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(DeviceInfo("cuda", props.name, i, props.total_memory / 2**20))
    except Exception:  # broken drivers must not crash the app; CPU stays available
        pass
    return devices


def choose_device(requested: str) -> tuple[torch.device, DeviceInfo, str]:
    """Resolve ``auto|cpu|cuda`` to a device and a user-facing explanation."""
    if requested not in DEVICE_CHOICES:
        raise DeviceError(f"Unknown device {requested!r}; choose one of {DEVICE_CHOICES}.")
    devices = detect_devices()
    gpus = [d for d in devices if d.kind == "cuda"]
    if requested == "cuda" and not gpus:
        raise DeviceError("A GPU was requested but no usable NVIDIA GPU/CUDA runtime was found. "
                          "Choose 'auto' or 'cpu' to train on the CPU.")
    if requested in ("cuda", "auto") and gpus:
        gpu = max(gpus, key=lambda d: d.total_memory_mb or 0)
        return torch.device("cuda", gpu.index), gpu, f"Training on {gpu.label()}."
    note = "Training on CPU (requested)." if requested == "cpu" else \
        "No NVIDIA GPU detected - training on CPU. This works but is slower; keep models and images small."
    return torch.device("cpu"), devices[0], note
