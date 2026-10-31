"""Background sampling of CPU, RAM and GPU usage during training.

Sources, in order of preference: NVML via ``pynvml`` (if installed), the
``nvidia-smi`` tool shipped with the NVIDIA driver, and PyTorch's own memory
counters. All figures are real measurements and carry ``simulated: False``.

For UI development on machines without a GPU, setting the environment variable
``MSW_SIMULATED_GPU_TELEMETRY=1`` adds a fake GPU reading flagged
``simulated: True``; the UI labels such values as SIMULATED. Training itself is
never simulated.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

SIMULATE_ENV = "MSW_SIMULATED_GPU_TELEMETRY"
NVIDIA_SMI_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=<NVIDIA_SMI_QUERY> --format=csv,noheader,nounits`` output."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue

        def num(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:  # "[N/A]", "[Not Supported]"
                return None

        gpus.append({
            "index": int(num(parts[0]) or 0), "name": parts[1], "utilization_percent": num(parts[2]),
            "memory_used_mb": num(parts[3]), "memory_total_mb": num(parts[4]), "temperature_c": num(parts[5]),
            "power_w": num(parts[6]), "source": "nvidia-smi", "simulated": False,
        })
    return gpus


def _query_nvml() -> list[dict[str, Any]] | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            name = pynvml.nvmlDeviceGetName(h)
            out.append({"index": i, "name": name.decode() if isinstance(name, bytes) else name,
                        "utilization_percent": float(util.gpu), "memory_used_mb": mem.used / 2**20,
                        "memory_total_mb": mem.total / 2**20,
                        "temperature_c": float(pynvml.nvmlDeviceGetTemperature(h, 0)), "power_w": None,
                        "source": "nvml", "simulated": False})
        return out
    except Exception:
        return None


def _query_nvidia_smi() -> list[dict[str, Any]] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        res = subprocess.run([exe, f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(res.stdout) if res.returncode == 0 else None


def _simulated_gpu(t: float) -> list[dict[str, Any]]:
    phase = (t % 20) / 20
    return [{"index": 0, "name": "SIMULATED GPU (no hardware)", "utilization_percent": round(60 + 30 * phase, 1),
             "memory_used_mb": round(2000 + 1500 * phase, 1), "memory_total_mb": 8192.0, "temperature_c": 55.0,
             "power_w": None, "source": "simulated", "simulated": True}]


def sample_system(torch_device: Any = None) -> dict[str, Any]:
    """One snapshot of process/system/GPU usage."""
    out: dict[str, Any] = {"simulated": False, "time": time.time()}
    try:
        import psutil

        proc = psutil.Process()
        vm = psutil.virtual_memory()
        out.update(process_rss_mb=round(proc.memory_info().rss / 2**20, 1),
                   system_cpu_percent=psutil.cpu_percent(interval=None),
                   system_memory_percent=vm.percent,
                   system_memory_total_mb=round(vm.total / 2**20))
    except ImportError:
        out["note"] = "psutil not installed; CPU/RAM not measured"
    gpus = _query_nvml() or _query_nvidia_smi()
    if gpus:
        out["gpus"] = gpus
    elif os.environ.get(SIMULATE_ENV) == "1":
        out["gpus"] = _simulated_gpu(time.time())
        out["simulated_gpu"] = True
    if torch_device is not None and getattr(torch_device, "type", "") == "cuda":
        import torch

        out["torch_gpu_allocated_mb"] = round(torch.cuda.memory_allocated(torch_device) / 2**20, 1)
        out["torch_gpu_peak_mb"] = round(torch.cuda.max_memory_allocated(torch_device) / 2**20, 1)
    return out


class ResourceSampler:
    """Samples in a daemon thread so slow queries (nvidia-smi) never stall training."""

    def __init__(self, interval_s: float = 2.0, torch_device: Any = None) -> None:
        self.interval_s = interval_s
        self.torch_device = torch_device
        self._latest: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="msw-resource-sampler", daemon=True)

    def start(self) -> "ResourceSampler":
        try:
            import psutil

            psutil.cpu_percent(interval=None)  # prime the CPU counter
        except ImportError:
            pass
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            snap = sample_system(self.torch_device)
            with self._lock:
                self._latest = snap
            self._stop.wait(self.interval_s)

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest) if self._latest else sample_system(self.torch_device)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
