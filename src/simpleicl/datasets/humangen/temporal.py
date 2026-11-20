"""Temporal alignment for frame-indexed human demonstration modalities."""

from __future__ import annotations

import numpy as np
import torch


def uniform_indices(source_length: int, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint-inclusive sample indices and validity for a fixed target length."""
    source_length = int(source_length)
    target_length = int(target_length)
    if source_length <= 0 or target_length <= 0:
        raise ValueError(f"source_length and target_length must be positive, got {source_length}, {target_length}")
    valid_length = min(source_length, target_length)
    if source_length >= target_length:
        indices = np.linspace(0, source_length - 1, target_length, dtype=np.int64)
        mask = np.ones(target_length, dtype=np.bool_)
    else:
        indices = np.concatenate([
            np.arange(source_length, dtype=np.int64),
            np.full(target_length - source_length, source_length - 1, dtype=np.int64),
        ])
        mask = np.arange(target_length) < valid_length
    return indices, mask


def align_sequence(value: torch.Tensor | np.ndarray, indices: np.ndarray, *, axis: int = 0):
    """Apply one shared integer time index to any modality sequence."""
    if value.shape[axis] <= 0:
        raise ValueError("Cannot align a sequence with an empty time axis")
    if isinstance(value, torch.Tensor):
        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(axis, index)
    return np.take(value, indices, axis=axis)


def align_human_modalities(video, modalities: dict[str, torch.Tensor], target_length: int):
    """Resample/pad all modalities to one 121-frame grid.

    Video defines the normalized temporal grid. Feature sequences with another
    frame count are nearest-neighbour mapped onto it; no feature interpolation
    is performed. Short video padding repeats the last aligned feature frame.
    """
    source_length = int(video.shape[0])
    indices, mask = uniform_indices(source_length, target_length)
    aligned = {}
    for name, value in modalities.items():
        if value.ndim == 0:
            raise ValueError(f"{name} must have a time axis")
        if value.shape[0] == source_length:
            modality_indices = indices
        else:
            # Map the sampled video timestamps to the modality's own frame grid.
            modality_indices = np.rint(
                indices.astype(np.float64) / max(source_length - 1, 1) * (value.shape[0] - 1)
            ).astype(np.int64)
        aligned[name] = align_sequence(value, modality_indices)
    return indices, mask, aligned
