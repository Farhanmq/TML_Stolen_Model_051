"""Model loading utilities for CIFAR-100 ResNet-18 checkpoints."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18


# ── Architecture ──────────────────────────────────────────────────────────────

def cifar_resnet18(num_classes: int = 100) -> nn.Module:
    """CIFAR-style ResNet-18: 3×3 first conv, no maxpool, custom num_classes."""
    model = resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Checkpoint readers ────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "F64": torch.float64, "F32": torch.float32,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64,   "I32": torch.int32,
    "I16": torch.int16,   "I8":  torch.int8,
    "U8":  torch.uint8,   "BOOL": torch.bool,
}


def _load_safetensors(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        # Read the entire data region into memory; avoids mmap exported-buffer issues.
        data = f.read()
    state: dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = data[start:end]
        t = torch.frombuffer(bytearray(raw), dtype=_DTYPE_MAP[meta["dtype"]]).reshape(meta["shape"])
        state[name] = t.to(device=device).clone()
    return state


def load_model(path: Path, device: torch.device, num_classes: int = 100) -> nn.Module:
    """Load a safetensors or .pt checkpoint into a CIFAR ResNet-18."""
    model = cifar_resnet18(num_classes).to(device)
    path = Path(path)

    if path.suffix == ".safetensors":
        sd = _load_safetensors(path, device)
    else:
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break

    # Strip common prefixes (module., model.)
    cleaned = {}
    for k, v in sd.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def list_suspects(suspect_dir: Path) -> list[Path]:
    """Return sorted list of suspect checkpoint paths."""
    paths = sorted(suspect_dir.glob("suspect_*.safetensors"))
    if not paths:
        paths = sorted(suspect_dir.glob("*.safetensors")) + sorted(suspect_dir.glob("*.pt"))
    return paths


def suspect_id_from_path(path: Path) -> int:
    """Extract integer id from filename like suspect_042.safetensors → 42."""
    stem = path.stem  # e.g. "suspect_042"
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot extract id from {path.name}")
