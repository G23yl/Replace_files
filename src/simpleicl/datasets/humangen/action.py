"""Adapt HumanGen source action/state transforms into a fixed 30D layout."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

LAYOUT = (("hand.position", 14), ("arm.position", 14), ("effector.position", 2))
ACTION_DIM = sum(width for _, width in LAYOUT)
_POSE_WIDTHS = {"xyzq": 7, "xyzqxyzq": 7, "xyze": 6, "xyzexyze": 6, "xyza": 6, "xyzaxyza": 6}


def _dense(value):
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if hasattr(value, "as_py"):
        value = value.as_py()
    array = np.asarray(value)
    if array.dtype == object:
        if array.ndim == 0:
            return _dense(array.item())
        return np.stack([_dense(item) for item in array])
    return array


def _field_values(frame, key):
    if key not in frame.columns:
        raise KeyError(f"feature {key!r} missing from parquet columns {list(frame.columns)}")
    return np.stack([_dense(value).reshape(-1) for value in frame[key].to_numpy()]).astype(np.float32)


def _origin_values(frame, origin):
    if isinstance(origin, str):
        return _field_values(frame, origin)
    pieces = []
    for item in origin:
        if not isinstance(item, Mapping) or len(item) != 1:
            raise ValueError(f"Invalid action_transform origin_keys entry: {item!r}")
        key, selection = next(iter(item.items()))
        values = _field_values(frame, key)
        pieces.append(values[:, int(selection.get("start", 0)):int(selection["end"])])
    return np.concatenate(pieces, axis=-1)


def _canonical_pose(value, pose_format):
    width = _POSE_WIDTHS.get(pose_format)
    if width is None:
        return np.asarray(value, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    parts = []
    for offset in range(0, value.shape[-1], width):
        pose = value[..., offset:offset + width]
        if pose_format in ("xyze", "xyzexyze"):
            quat = Rotation.from_euler("xyz", pose[..., 3:6]).as_quat()
        elif pose_format in ("xyza", "xyzaxyza"):
            quat = Rotation.from_rotvec(pose[..., 3:6]).as_quat()
        else:
            quat = Rotation.from_quat(pose[..., 3:7]).as_quat()
        quat *= np.where(quat[..., 3:4] >= 0, 1.0, -1.0)
        parts.append(np.concatenate((pose[..., :3], quat.astype(np.float32)), axis=-1))
    return np.concatenate(parts, axis=-1)


def _specs(entries):
    result = {}
    for entry in entries or []:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ValueError(f"Invalid action_transform entry: {entry!r}")
        key, spec = next(iter(entry.items()))
        result[key] = dict(spec)
    return result


def parse_transform(transform):
    return _specs(transform.get("states")), _specs(transform.get("actions")), _specs(transform.get("images"))


def transform_vector(frame, specs, kind, *, origin_offset=0, num_rows=None, action_specs=None):
    if num_rows is None:
        num_rows = len(frame) - origin_offset
    output = np.zeros((num_rows, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((num_rows, ACTION_DIM), dtype=bool)
    offset = 0
    for short_name, width in LAYOUT:
        spec = specs.get(f"{kind}.{short_name}")
        if spec is not None:
            shift = int(spec.get("shift", 0))
            if shift < 0:
                raise ValueError(f"Negative action/state shift for {kind}.{short_name}: {shift}")
            row_start = max(0, origin_offset - shift)
            values = _origin_values(frame.iloc[row_start:min(len(frame), row_start + num_rows)], spec["origin_keys"])
            fmt = spec.get("format")
            if fmt is None and kind.startswith("observation.state"):
                fmt = (action_specs or {}).get(f"action.{short_name}", {}).get("format")
            if fmt:
                values = _canonical_pose(values, fmt)
            if values.shape[-1] > width:
                raise ValueError(f"{kind}.{short_name} resolved to {values.shape[-1]} dims, maximum is {width}")
            n = min(num_rows, len(values))
            output[:n, offset:offset + values.shape[-1]] = values[:n]
            mask[:n, offset:offset + values.shape[-1]] = True
        offset += width
    return output, mask


def map_robotwin(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 16:
        raise ValueError(f"Robotwin expects 16D action/state, got {values.shape[-1]}")
    output = np.zeros((*values.shape[:-1], ACTION_DIM), dtype=np.float32)
    mask = np.zeros_like(output, dtype=bool)
    output[..., :7] = values[..., :7]
    output[..., 7:14] = values[..., 8:15]
    output[..., 28] = values[..., 7]
    output[..., 29] = values[..., 15]
    mask[..., :14] = True
    mask[..., 28:30] = True
    return output, mask
