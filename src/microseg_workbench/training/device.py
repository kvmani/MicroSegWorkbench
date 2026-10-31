"""Compute device discovery with a visible, explicit CPU fallback."""

from __future__ import annotations

import os
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
    free_memory_mb: float | None = None
    compute_capability: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        if self.kind == "cuda":
            free = f", {self.free_memory_mb:.0f} MB free" if self.free_memory_mb is not None else ""
            return f"GPU {self.index}: {self.name} ({self.total_memory_mb or 0:.0f} MB{free})"
        return f"CPU: {self.name} ({os.cpu_count()} logical cores)"


def _cpu_info() -> DeviceInfo:
    import platform

    return DeviceInfo(kind="cpu", name=platform.processor() or platform.machine() or "CPU")


def cuda_diagnosis() -> str:
    """Plain-language reason why CUDA is or is not usable."""
    if torch.version.cuda is None:
        return ("This PyTorch build has no CUDA support (CPU-only wheel). Install the CUDA build of PyTorch from "
                "the internal mirror to use an NVIDIA GPU.")
    try:
        if not torch.cuda.is_available():
            return (f"PyTorch supports CUDA {torch.version.cuda}, but no usable NVIDIA GPU/driver was found. "
                    "Check that an NVIDIA GPU is installed and its driver is up to date.")
        return f"CUDA {torch.version.cuda} available with {torch.cuda.device_count()} GPU(s)."
    except Exception as exc:  # driver problems surface here
        return f"CUDA initialisation failed: {exc}"


def detect_devices() -> list[DeviceInfo]:
    devices = [_cpu_info()]
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                    free_mb = free / 2**20
                except Exception:
                    free_mb = None
                devices.append(DeviceInfo("cuda", props.name, i, props.total_memory / 2**20, free_mb,
                                          f"{props.major}.{props.minor}"))
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


def is_out_of_memory(exc: BaseException) -> bool:
    """True for GPU or host out-of-memory errors raised during a training step."""
    oom_types = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                                  getattr(torch.cuda, "OutOfMemoryError", None)) if t is not None)
    if isinstance(exc, (MemoryError, *oom_types)):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def probe_batch_size(try_step, start: int, minimum: int = 1) -> tuple[int, list[dict]]:
    """Largest batch size <= ``start`` (halving) for which ``try_step(batch)`` does not run out of memory.

    ``try_step`` performs one forward/backward pass. Returns the batch size and the attempts made.
    Raises the last OOM error if even ``minimum`` does not fit.
    """
    batch, attempts = start, []
    while True:
        try:
            try_step(batch)
            attempts.append({"batch_size": batch, "ok": True})
            return batch, attempts
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            attempts.append({"batch_size": batch, "ok": False})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch <= minimum:
                raise
            batch = max(minimum, batch // 2)
