#!/usr/bin/env python3
"""Encode HumanGen demonstration videos into VAE latents.

Run from ``icl_model/``. Every input video is mirrored under
``../data/121_human_latents`` with the same relative source/run/sample path.
Only local Wan2.2 VAE weights are used; the script does not download weights.

Examples::

    python scripts/precompute_humangen_video_latents.py
    python scripts/precompute_humangen_video_latents.py +human_vae_models_per_gpu=2
    torchrun --standalone --nproc_per_node=4 scripts/precompute_humangen_video_latents.py
    torchrun --standalone --nproc_per_node=4 scripts/precompute_humangen_video_latents.py +human_vae_models_per_gpu=2

``human_vae_models_per_gpu`` defaults to 1. Increasing it loads multiple VAE
copies on each GPU and increases per-GPU memory use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simpleicl.datasets.humangen.human_modalities import prepare_human_video_for_vae
from simpleicl.models.wan22.helpers.loader import _load_registered_model
from simpleicl.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
DEFAULT_HUMAN_ROOT = Path("../data/human_data")
DEFAULT_OUTPUT_ROOT = Path("../data/121_human_latents")
DEFAULT_VAE_PATH = Path("checkpoints/wan22-5b/Wan2.2_VAE.pth")
DEFAULT_VIDEO_SIZE = (320, 448)
DEFAULT_NUM_FRAMES = 121


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _read_video(path: Path) -> torch.Tensor:
    """Decode full RGB video as uint8 [T,3,H,W] for shared preprocessing."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if not frames:
        raise ValueError(f"Human video has no decodable frames: {path}")
    return torch.stack(frames)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _discover_videos(human_root: Path) -> list[Path]:
    if not human_root.is_dir():
        raise FileNotFoundError(f"human_data root does not exist: {human_root}")
    return sorted(path for path in human_root.rglob("*.mp4") if path.is_file())


def _destination_for(video_path: Path, human_root: Path, output_root: Path) -> Path:
    try:
        relative = video_path.relative_to(human_root)
    except ValueError as exc:
        raise ValueError(f"Video {video_path} is outside human root {human_root}") from exc
    return (output_root / relative).with_suffix(".pth")


def _load_vae(vae_path: Path, device: str) -> torch.nn.Module:
    if not vae_path.is_file():
        raise FileNotFoundError(f"Local Wan VAE checkpoint not found: {vae_path}")
    return _load_registered_model(
        str(vae_path),
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()


def _encode_video(vae, video_path: Path, *, num_frames: int, video_size: tuple[int, int], device: str):
    raw = _read_video(video_path)
    video, frame_mask = prepare_human_video_for_vae(
        raw, target_frames=num_frames, video_size=video_size
    )
    # Wan VAE consumes [B,C,T,H,W], with RGB already normalized to [-1,1].
    video = video.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        latent = vae.encode(video, device=torch.device(device), tiled=False)
    if isinstance(latent, list):
        if len(latent) != 1:
            raise ValueError(f"Expected one latent for {video_path}, got {len(latent)}")
        latent = latent[0].unsqueeze(0)
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"Expected VAE output [1,C,T,H,W], got {tuple(latent.shape)}")
    latent = latent[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    payload = {
        "latent": latent,
        "latent_num_frames": int(latent.shape[1]),
        "latent_height": int(latent.shape[2]),
        "latent_width": int(latent.shape[3]),
        "source_num_frames": int(raw.shape[0]),
        "sampled_num_frames": int(num_frames),
        "valid_frame_count": int(frame_mask.sum().item()),
        "video_size": list(map(int, video_size)),
        "preprocessing": "uniform_endpoints_repeat_last_resize_crop_rgb_minus1_1",
    }
    return payload


def _encode_shard(
    *, rank: int, replica_index: int, replica_count: int, videos: list[Path],
    human_root: Path, output_root: Path, vae_path: Path, num_frames: int,
    video_size: tuple[int, int], device: str, overwrite: bool,
) -> tuple[int, int, int]:
    """Run one local VAE replica on a disjoint shard of this rank's videos."""
    vae = _load_vae(vae_path, device)
    stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None
    encoded = skipped = errors = 0
    for video_path in tqdm(
        videos[replica_index::replica_count],
        desc=f"rank {rank} VAE {replica_index + 1}/{replica_count}",
        unit="video",
        disable=replica_index != 0,
    ):
        destination = _destination_for(video_path, human_root, output_root)
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    payload = _encode_video(
                        vae, video_path, num_frames=num_frames,
                        video_size=video_size, device=device,
                    )
                stream.synchronize()
            else:
                payload = _encode_video(
                    vae, video_path, num_frames=num_frames,
                    video_size=video_size, device=device,
                )
            payload["source_video"] = str(video_path.relative_to(human_root))
            _atomic_torch_save(payload, destination)
            encoded += 1
        except Exception:
            errors += 1
            logger.exception("Rank %d VAE replica %d failed to encode %s", rank, replica_index, video_path)
    del vae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return encoded, skipped, errors


def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    is_distributed, rank, world_size, local_rank = _init_distributed()
    try:
        human_root = Path(str(cfg.get("human_data_root", DEFAULT_HUMAN_ROOT))).expanduser()
        if not human_root.is_absolute():
            human_root = (Path.cwd() / human_root).resolve()
        output_root = Path(str(cfg.get("human_latent_output_root", DEFAULT_OUTPUT_ROOT))).expanduser()
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()
        vae_path = Path(str(cfg.get("human_vae_path", DEFAULT_VAE_PATH))).expanduser()
        if not vae_path.is_absolute():
            vae_path = (Path.cwd() / vae_path).resolve()
        num_frames = int(cfg.get("human_latent_num_frames", DEFAULT_NUM_FRAMES))
        video_size = tuple(map(int, cfg.get("human_latent_video_size", DEFAULT_VIDEO_SIZE)))
        vae_models_per_gpu = int(cfg.get("human_vae_models_per_gpu", 1))
        overwrite_value = cfg.get("overwrite", False)
        if isinstance(overwrite_value, bool):
            overwrite = overwrite_value
        else:
            normalized_overwrite = str(overwrite_value).strip().lower()
            if normalized_overwrite not in {"1", "true", "yes", "y", "0", "false", "no", "n"}:
                raise ValueError(f"Invalid overwrite value: {overwrite_value!r}")
            overwrite = normalized_overwrite in {"1", "true", "yes", "y"}
        if num_frames < 1 or vae_models_per_gpu < 1:
            raise ValueError("human_latent_num_frames and human_vae_models_per_gpu must be positive")
        if not human_root.is_dir():
            raise FileNotFoundError(f"human_data root does not exist: {human_root}")
        if not vae_path.is_file():
            raise FileNotFoundError(f"Local Wan VAE checkpoint not found: {vae_path}")

        videos = _discover_videos(human_root)
        if not videos:
            raise ValueError(f"No .mp4 files found under {human_root}")
        local_videos = videos[rank::world_size]
        if is_distributed and not torch.cuda.is_available():
            raise RuntimeError("Distributed HumanGen VAE precompute requires CUDA on every rank")
        if not torch.cuda.is_available() and vae_models_per_gpu != 1:
            raise ValueError("human_vae_models_per_gpu > 1 requires CUDA")
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        replica_count = vae_models_per_gpu if device.startswith("cuda") else 1
        logger.info(
            "Rank %d/%d assigned %d videos; local VAE replicas=%d on %s",
            rank, world_size, len(local_videos), replica_count, device,
        )
        replica_results = []
        # Separate VAE replicas and CUDA streams keep each local shard independent;
        # increase this only when the GPU has enough memory for multiple VAE copies.
        with ThreadPoolExecutor(max_workers=replica_count) as pool:
            futures = [
                pool.submit(
                    _encode_shard,
                    rank=rank,
                    replica_index=replica_index,
                    replica_count=replica_count,
                    videos=local_videos,
                    human_root=human_root,
                    output_root=output_root,
                    vae_path=vae_path,
                    num_frames=num_frames,
                    video_size=video_size,
                    device=device,
                    overwrite=overwrite,
                )
                for replica_index in range(replica_count)
            ]
            for future in as_completed(futures):
                replica_results.append(future.result())
        encoded = sum(result[0] for result in replica_results)
        skipped = sum(result[1] for result in replica_results)
        errors = sum(result[2] for result in replica_results)

        counts_device = torch.device(device) if is_distributed else torch.device("cpu")
        counts = torch.tensor([encoded, skipped, errors], dtype=torch.long, device=counts_device)
        if is_distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.barrier()
        if rank == 0:
            logger.info(
                "Finished VAE latent precompute: encoded=%d skipped=%d errors=%d total_videos=%d",
                int(counts[0]), int(counts[1]), int(counts[2]), len(videos),
            )
        if int(counts[2]) > 0:
            raise RuntimeError(f"Failed to encode {int(counts[2])} human videos; see per-file errors above")
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
