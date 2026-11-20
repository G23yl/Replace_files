"""OpenWAM-compatible grouped action/state normalization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json

import numpy as np
import torch

from .action import ACTION_DIM, LAYOUT


def load_stats(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Action/state statistics file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read statistics JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Statistics JSON must contain an object: {path}")
    stats = payload.get("norm_stats", payload)
    if not isinstance(stats, Mapping):
        raise ValueError(f"Statistics file {path} must contain norm_stats object")
    return dict(stats)


def _apply_robot_stats(
    values: torch.Tensor,
    valid: torch.Tensor,
    stats: dict,
    *,
    kind: str,
    mode: str,
    path: str,
    inverse: bool,
) -> torch.Tensor:
    if values.shape != valid.shape or values.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected matching (...,{ACTION_DIM}) values/mask, got {values.shape}/{valid.shape}")
    if mode not in {"zscore", "minmax"}:
        raise ValueError(f"Unsupported robot normalization mode {mode!r}")
    result = torch.zeros_like(values)
    offset = 0
    for group, width in LAYOUT:
        key = f"{kind}.{group}"
        block = stats.get(key)
        block_mask = valid[..., offset:offset + width]
        if not bool(block_mask.any()):
            offset += width
            continue
        if not isinstance(block, Mapping):
            raise ValueError(f"Statistics file {path} is missing active group {key!r}")
        low_key, high_key = (("mean", "std") if mode == "zscore" else ("min", "max"))
        if low_key not in block or high_key not in block:
            raise ValueError(f"Statistics group {key!r} must contain {low_key}/{high_key}")
        center = torch.as_tensor(block[low_key], dtype=values.dtype).reshape(-1)
        scale = torch.as_tensor(block[high_key], dtype=values.dtype).reshape(-1)
        if center.numel() != width or scale.shape != center.shape:
            raise ValueError(f"Statistics group {key!r} must have width {width}, got {center.numel()}/{scale.numel()}")
        if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()):
            raise ValueError(f"Statistics group {key!r} contains non-finite values")
        block_values = values[..., offset:offset + width]
        if mode == "zscore":
            transformed = block_values * (scale + 1e-8) + center if inverse else (block_values - center) / (scale + 1e-8)
        else:
            span = scale - center
            if bool((span <= 0).any()):
                raise ValueError(f"Statistics group {key!r} has max <= min")
            transformed = (block_values + 1.0) * 0.5 * span + center if inverse else 2.0 * (block_values - center) / span - 1.0
        if not inverse:
            transformed = transformed.clamp(-5.0, 5.0)
        result[..., offset:offset + width] = torch.where(
            block_mask, transformed, torch.zeros_like(transformed)
        )
        offset += width
    return result


def normalize_robot(values: torch.Tensor, valid: torch.Tensor, stats: dict, *, kind: str, mode: str, path: str) -> torch.Tensor:
    """Normalize valid channels; invalid/padded channels stay zero."""
    return _apply_robot_stats(values, valid, stats, kind=kind, mode=mode, path=path, inverse=False)


def denormalize_robot(values: torch.Tensor, valid: torch.Tensor, stats: dict, *, kind: str, mode: str, path: str) -> torch.Tensor:
    """Invert z-score/min-max values; invalid channels stay zero.

    Normalization clips outputs to [-5, 5], so values that hit that clip cannot
    be recovered exactly; this returns the raw value represented by the boundary.
    """
    return _apply_robot_stats(values, valid, stats, kind=kind, mode=mode, path=path, inverse=True)
