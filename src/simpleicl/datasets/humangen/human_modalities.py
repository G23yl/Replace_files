"""Safe readers for human video, DINO features, hands and VAE latents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import av
import numpy as np
import torch

from .temporal import align_human_modalities, uniform_indices


def read_video(path: str | Path) -> torch.Tensor:
    """Decode RGB frames as uint8 [T,3,H,W]."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if not frames:
        raise ValueError(f"Human video has no decodable frames: {path}")
    return torch.stack(frames)


def prepare_human_video_for_vae(
    frames: torch.Tensor,
    *,
    target_frames: int = 121,
    video_size: tuple[int, int] = (256, 448),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same temporal/spatial preprocessing used by the HumanGen reader.

    Returns normalized ``[3,T,H,W]`` float video in [-1,1] and its valid-frame mask.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"Expected uint8 video [T,3,H,W], got {tuple(frames.shape)}")
    indices, frame_mask = uniform_indices(int(frames.shape[0]), int(target_frames))
    index = torch.as_tensor(indices, dtype=torch.long)
    selected = frames.index_select(0, index).float().div_(255.0)
    selected = torch.nn.functional.interpolate(
        selected, size=tuple(map(int, video_size)), mode="bilinear", align_corners=False
    )
    video = selected.permute(1, 0, 2, 3).contiguous().mul_(2.0).sub_(1.0)
    return video, torch.from_numpy(frame_mask)


def read_dino(path: str | Path, *, max_objects: int = 4) -> tuple[torch.Tensor, dict]:
    """Concatenate cls and patch_mean, then truncate/pad to a fixed object count."""
    if max_objects < 1:
        raise ValueError(f"max_objects must be positive, got {max_objects}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"cls", "patch_mean", "valid", "names", "ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"DINO file {path} misses keys {sorted(missing)}; keys={archive.files}")
        cls = np.asarray(archive["cls"])
        patch_mean = np.asarray(archive["patch_mean"])
        valid = np.asarray(archive["valid"], dtype=np.bool_)
        ids = np.asarray(archive["ids"]).tolist()
        names = np.asarray(archive["names"]).tolist()
    if cls.ndim != 3 or patch_mean.ndim != 3:
        raise ValueError(f"DINO cls/patch_mean must be [T,N,D], got {cls.shape}/{patch_mean.shape}")
    if cls.shape[:-1] != patch_mean.shape[:-1]:
        raise ValueError(f"DINO cls and patch_mean must have matching time/object dimensions, got {cls.shape}/{patch_mean.shape}")
    feature = np.concatenate([cls, patch_mean], axis=-1)
    if valid.ndim == 1:
        valid = np.broadcast_to(valid[:, None], feature.shape[:2])
    elif valid.ndim == 2 and valid.shape[1] == 1 and feature.shape[1] != 1:
        valid = np.broadcast_to(valid, feature.shape[:2])
    if valid.shape != feature.shape[:2]:
        raise ValueError(
            f"DINO valid mask must match [T,N]={feature.shape[:2]}, got {valid.shape}"
        )
    source_object_count = feature.shape[1]
    kept_object_count = min(source_object_count, max_objects)
    feature = feature[:, :kept_object_count]
    valid = valid[:, :kept_object_count]
    padded = np.zeros((feature.shape[0], max_objects, feature.shape[-1]), dtype=feature.dtype)
    padded[:, :kept_object_count] = feature
    padded_valid = np.zeros((feature.shape[0], max_objects), dtype=np.bool_)
    padded_valid[:, :kept_object_count] = valid
    return torch.from_numpy(np.ascontiguousarray(padded)).float(), {
        "key": "cls+patch_mean", "source_shape": tuple(feature.shape),
        "valid": torch.from_numpy(np.ascontiguousarray(padded_valid)),
        "object_count": kept_object_count, "source_object_count": source_object_count,
        "ids": ids, "names": names,
    }


def _safe_torch_load(path: str | Path):
    """Allow only NumPy scalar/dtype metadata used by the observed hands file."""
    try:
        from numpy._core.multiarray import scalar
        dtype_types = [np.dtypes.Float64DType, np.dtypes.Float32DType,
                       np.dtypes.Int64DType, np.dtypes.Int32DType,
                       np.dtypes.BoolDType]
        allowed = [(scalar, "numpy._core.multiarray.scalar"), (np.dtype, "numpy.dtype")]
        allowed.extend((dtype, f"numpy.dtypes.{dtype.__name__}") for dtype in dtype_types)
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        unsafe = []
        try:
            unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
        except Exception:
            pass
        raise RuntimeError(
            f"Cannot safely load {path} with weights_only=True; unsafe globals={unsafe}. "
            "Re-export payload using tensors and primitive metadata only."
        ) from exc


def read_hands(path: str | Path, *, key: str | None = None) -> tuple[torch.Tensor, dict]:
    """Extract two-hand 2D keypoints and validity from framewise pose predictions."""
    payload = _safe_torch_load(path)
    if isinstance(payload, Mapping) and payload.get("format") == "gem_pose_predictions_v1":
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"hands payload has no frame records: {path}")
        t = len(frames)
        points = torch.zeros((t, 2, 25, 2), dtype=torch.float32)
        valid = torch.zeros((t, 2, 25), dtype=torch.bool)
        image_sizes = []
        side_index = {"left": 0, "right": 1}
        for ti, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise ValueError(f"Invalid hand frame {ti} in {path}")
            image_size = frame.get("image_size")
            if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
                raise ValueError(f"Missing image_size [width,height] at frame {ti} in {path}")
            image_sizes.append(tuple(map(float, image_size)))
            for instance in frame.get("instances", []):
                side = str(instance.get("side", instance.get("track_id", ""))).lower()
                if side not in side_index:
                    continue
                si = side_index[side]
                coords = instance.get("keypoints_2d")
                point_valid = instance.get("projection_valid")
                if not isinstance(coords, torch.Tensor) or tuple(coords.shape) != (25, 2):
                    raise ValueError(f"Expected keypoints_2d [25,2] at frame {ti} in {path}")
                if not isinstance(point_valid, torch.Tensor) or tuple(point_valid.shape) != (25,):
                    raise ValueError(f"Expected projection_valid [25] at frame {ti} in {path}")
                points[ti, si] = coords.float()
                valid[ti, si] = point_valid.bool()
        sizes = torch.tensor(image_sizes, dtype=points.dtype)
        if not torch.all(sizes == sizes[0]):
            raise ValueError(f"hands image_size changes across frames in {path}; cannot normalize with one frame grid")
        width, height = sizes[0]
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid hands image_size {sizes[0].tolist()} in {path}")
        points[..., 0] /= width
        points[..., 1] /= height
        points = torch.cat([points, valid.to(points.dtype).unsqueeze(-1)], dim=-1)
        return points, {"valid": valid, "source_frame_count": t,
                        "format": payload.get("format"), "coordinate_system": "image_xy_0_1"}
    if isinstance(payload, torch.Tensor):
        value, meta = payload, {}
    elif isinstance(payload, Mapping):
        meta = {k: v for k, v in payload.items() if k != key} if key else {}
        if key is not None:
            if key not in payload:
                raise KeyError(f"hands key {key!r} missing from {path}; keys={list(payload)}")
            value = payload[key]
        else:
            candidates = [v for v in payload.values() if isinstance(v, torch.Tensor) and v.ndim >= 2]
            if len(candidates) != 1:
                raise ValueError(f"Cannot uniquely infer hands tensor in {path}; keys={list(payload)}")
            value = candidates[0]
    else:
        raise TypeError(f"Unsupported hands payload type {type(payload).__name__} in {path}")
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ValueError(f"Hands data must be tensor [T,...], got {type(value).__name__} in {path}")
    return value.float().contiguous(), meta


def load_human_latent(path: str | Path) -> tuple[torch.Tensor, dict]:
    """Load a full-video VAE latent without forcing it onto a 121-frame grid."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Human latent must be a dict: {path}")
    latent = payload.get("latent", payload.get("latents"))
    if not isinstance(latent, torch.Tensor):
        raise ValueError(f"Human latent payload has no tensor latent/latents: {path}")
    shape = {key: int(payload[key]) for key in
             ("latent_num_frames", "latent_height", "latent_width") if key in payload}
    if latent.ndim == 2 and len(shape) == 3:
        t, h, w = shape["latent_num_frames"], shape["latent_height"], shape["latent_width"]
        if latent.shape[0] != t * h * w:
            raise ValueError(f"Latent token count does not match metadata in {path}: {latent.shape}")
        latent = latent.reshape(t, h, w, -1).permute(3, 0, 1, 2)
    elif latent.ndim != 4:
        raise ValueError(f"Expected latent [C,T,H,W] or flattened tokens, got {tuple(latent.shape)} in {path}")
    return latent.contiguous(), {"shape": shape}


def load_video_modalities(video_path, dino_path, hands_path, *, target_frames: int, video_size=(256, 448), max_dino_objects: int = 4):
    """Load selected framewise modalities using one shared timestamp grid."""
    video = read_video(video_path)
    sequences = {}
    dino_meta, hands_meta = {}, {}
    if dino_path is not None:
        dino, dino_meta = read_dino(dino_path, max_objects=max_dino_objects)
        sequences["dino"] = dino
    if hands_path is not None:
        hands, hands_meta = read_hands(hands_path)
        sequences["hands"] = hands
    indices, mask, aligned = align_human_modalities(video, sequences, target_frames)
    dino_valid = dino_meta.get("valid")
    if dino_valid is not None:
        if dino_valid.shape[0] == video.shape[0]:
            valid_indices = indices
        else:
            valid_indices = np.rint(
                indices.astype(np.float64) / max(video.shape[0] - 1, 1) * (dino_valid.shape[0] - 1)
            ).astype(np.int64)
        dino_valid = dino_valid.index_select(0, torch.as_tensor(valid_indices, dtype=torch.long))
        frame_valid = torch.from_numpy(mask).reshape(
            (len(mask),) + (1,) * (dino_valid.ndim - 1)
        )
        dino_valid &= frame_valid
    hand_valid = hands_meta.get("valid")
    if hand_valid is not None:
        if hand_valid.shape[0] == video.shape[0]:
            valid_indices = indices
        else:
            valid_indices = np.rint(
                indices.astype(np.float64) / max(video.shape[0] - 1, 1) * (hand_valid.shape[0] - 1)
            ).astype(np.int64)
        hand_valid = hand_valid.index_select(0, torch.as_tensor(valid_indices, dtype=torch.long))
        hand_valid &= torch.from_numpy(mask)[:, None, None]
    sampled_video, _ = prepare_human_video_for_vae(
        video, target_frames=target_frames, video_size=video_size
    )
    result = {"human_video": sampled_video}
    if "dino" in aligned:
        result["human_dino"] = aligned["dino"]
    if "hands" in aligned:
        result["human_hands"] = aligned["hands"]
    if dino_valid is not None:
        result["human_dino_valid"] = dino_valid.bool()
    if hand_valid is not None:
        result["human_hands_valid"] = hand_valid.bool()
    return result, {"dino": {k: v for k, v in dino_meta.items() if k != "valid"},
                    "hands": {k: v for k, v in hands_meta.items() if k != "valid"},
                    "video_source_frames": int(video.shape[0])}


def load_human_features(dino_path, hands_path, *, target_frames: int, max_dino_objects: int = 4):
    """Load and align DINO/hands without decoding the original human video."""
    dino, dino_meta = read_dino(dino_path, max_objects=max_dino_objects)
    hands, hands_meta = read_hands(hands_path)
    reference = torch.empty((dino.shape[0], 1))
    indices, mask, aligned = align_human_modalities(
        reference, {"dino": dino, "hands": hands}, target_frames
    )
    result = {
        "human_dino": aligned["dino"],
        "human_hands": aligned["hands"],
    }
    dino_valid = dino_meta.get("valid")
    if dino_valid is not None:
        valid_indices = indices if dino_valid.shape[0] == dino.shape[0] else np.rint(
            indices.astype(np.float64) / max(dino.shape[0] - 1, 1) * (dino_valid.shape[0] - 1)
        ).astype(np.int64)
        dino_valid = dino_valid.index_select(
            0, torch.as_tensor(valid_indices, dtype=torch.long)
        ).bool()
        frame_valid = torch.from_numpy(mask).reshape(
            (len(mask),) + (1,) * (dino_valid.ndim - 1)
        )
        result["human_dino_valid"] = dino_valid & frame_valid
    hands_valid = hands_meta.get("valid")
    if hands_valid is not None:
        valid_indices = indices if hands_valid.shape[0] == dino.shape[0] else np.rint(
            indices.astype(np.float64) / max(dino.shape[0] - 1, 1) * (hands_valid.shape[0] - 1)
        ).astype(np.int64)
        result["human_hands_valid"] = hands_valid.index_select(
            0, torch.as_tensor(valid_indices, dtype=torch.long)
        ).bool() & torch.from_numpy(mask)[:, None, None]
    metadata = {
        "dino": {key: value for key, value in dino_meta.items() if key != "valid"},
        "hands": {key: value for key, value in hands_meta.items() if key != "valid"},
    }
    return result, metadata
