"""Device selection helpers for local model execution."""

from __future__ import annotations

from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover - handled at runtime when torch is unavailable
    torch = None
    HAS_TORCH = False


def resolve_device(preferred_device: Optional[str] = None) -> str:
    """Resolve a torch device string, preferring CUDA like the notebook setup."""
    if preferred_device:
        return preferred_device

    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def get_torch_device(preferred_device: Optional[str] = None):
    """Return a torch.device object or a plain string if torch is unavailable."""
    device_str = resolve_device(preferred_device)
    if HAS_TORCH:
        return torch.device(device_str)
    return device_str


def format_device_message(preferred_device: Optional[str] = None) -> str:
    """Return the notebook-style device log line."""
    return f"Using device: {resolve_device(preferred_device)}"