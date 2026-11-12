"""Collect environment/version information for results files."""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def _run(cmd: list[str]) -> str | None:
    """Run a command, returning stripped stdout or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _torch_info() -> dict[str, Any]:
    """Torch/torchvision/CUDA/cuDNN/GPU versions; all None if torch is not installed."""
    info: dict[str, Any] = {k: None for k in
                            ("torch", "torchvision", "cuda", "cudnn", "cuda_available", "gpu_name")}
    try:
        import torch
    except ImportError:
        return info
    info.update(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    return info


def collect_env() -> dict[str, Any]:
    """Return python, platform, torch, CUDA, cuDNN, GPU, driver and pip-freeze information."""
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    return {
        "python": sys.version,
        "platform": platform.platform(),
        **_torch_info(),
        "driver_version": driver.splitlines()[0] if driver else None,
        "pip_freeze": freeze.splitlines() if freeze else None,
    }
