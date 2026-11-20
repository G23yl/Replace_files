"""Fixed three-slot robot camera layouts with black padding for absent views."""

from __future__ import annotations

import torch


def t_layout_geometry(size: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return OpenWAM's top/left/right slot sizes and origins for [H,W]."""
    height, width = map(int, size)
    top_h = int(round(height * 2.0 / 3.0))
    bottom_h = height - top_h
    left_w = width // 2
    return (
        [(top_h, width), (bottom_h, left_w), (bottom_h, width - left_w)],
        [(0, 0), (0, top_h), (left_w, top_h)],
    )


def compose_three_views(views: list[torch.Tensor | None], size: tuple[int, int]) -> torch.Tensor:
    """Paste three already slot-resized views into OpenWAM's fixed T canvas."""
    if len(views) != 3:
        raise ValueError(f"Exactly 3 camera slots are required, got {len(views)}")
    height, width = map(int, size)
    reference = next((view for view in views if view is not None), None)
    if reference is None:
        raise ValueError("All configured robot camera views are missing")
    frames = reference.shape[0]
    canvas = torch.zeros((frames, 3, height, width), dtype=torch.float32)
    slot_sizes, origins = t_layout_geometry(size)
    for view, (slot_h, slot_w), (x, y) in zip(views, slot_sizes, origins):
        if view is None:
            continue
        if tuple(view.shape[-2:]) != (slot_h, slot_w):
            raise ValueError(f"Camera slot expected {(slot_h, slot_w)}, got {tuple(view.shape[-2:])}")
        canvas[:, :, y:y + slot_h, x:x + slot_w] = view.float()
    return canvas.permute(1, 0, 2, 3).contiguous()
