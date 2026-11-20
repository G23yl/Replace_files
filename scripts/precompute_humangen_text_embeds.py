#!/usr/bin/env python3
"""Precompute Wan text-context caches for HumanGen episode prompts.

Run from ``icl_model/`` so Hydra resolves the same ``data=humangen`` config as
training. The script only uses local Wan files and never downloads checkpoints::

    python scripts/precompute_humangen_text_embeds.py data=humangen
    torchrun --standalone --nproc_per_node=4 scripts/precompute_humangen_text_embeds.py data=humangen
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import uuid
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simpleicl.models.wan22.helpers.loader import _load_registered_model
from simpleicl.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from simpleicl.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parents[1] / "checkpoints" / "wan22-5b")
CACHE_ENCODER_ID = "wan22ti2v5b"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
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


def _all_reduce_counts(values: list[int], device: str) -> list[int]:
    if not dist.is_initialized():
        return values
    reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
    counts = torch.tensor(values, dtype=torch.long, device=reduce_device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts.cpu().tolist()


def _atomic_torch_save(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _prompt_for_record(record: dict) -> str:
    action_configs = record["episode"].get("action_config") or []
    if isinstance(action_configs, Mapping):
        action_configs = [action_configs]
    action_text = next(
        (entry.get("action_text") for entry in action_configs
         if isinstance(entry, Mapping) and entry.get("action_text")),
        None,
    )
    task = str((record["episode"].get("tasks") or [record["root"].name])[0])
    prompt = str(action_text or task).strip()
    if not prompt:
        raise ValueError(
            f"Empty prompt in {record['root']} episode {record['episode_index']}"
        )
    return prompt


def _unique_prompts(dataset) -> list[str]:
    prompts = []
    seen = set()
    source_datasets = getattr(dataset, "datasets", [dataset])
    for source_dataset in source_datasets:
        for record in source_dataset.records:
            prompt = _prompt_for_record(record)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def _resolve_local_file(model_dir: Path, pattern: str) -> str:
    if pattern.endswith("/"):
        path = model_dir / pattern
        if not path.is_dir():
            raise FileNotFoundError(f"Expected local tokenizer directory: {path}")
        return str(path)
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No local file matches {model_dir / pattern}")
    if len(matches) == 1:
        return str(matches[0])
    return [str(path) for path in matches]


def _load_local_text_components(
    *, model_dir: Path, context_len: int, device: str, dtype: torch.dtype,
):
    # Resolve explicit local files so this precompute path cannot initiate downloads.
    text_path = _resolve_local_file(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = _resolve_local_file(model_dir, "google/umt5-xxl/")
    encoder = _load_registered_model(
        text_path, "wan_video_text_encoder", torch_dtype=dtype, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path, seq_len=int(context_len), clean="whitespace"
    )
    return encoder, tokenizer


def precompute(
    *,
    cache_dir: Path,
    prompts: list[str],
    model_dir: Path,
    context_len: int,
    batch_size: int,
    device: str,
    overwrite: bool,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[int, int, int]:
    pending = []
    skipped_existing = 0
    overwritten = 0
    for prompt in prompts[rank::world_size]:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        destination = cache_dir / f"{digest}.t5_len{context_len}.{CACHE_ENCODER_ID}.pt"
        if destination.exists() and not overwrite:
            skipped_existing += 1
            continue
        if destination.exists():
            overwritten += 1
        pending.append((prompt, destination))
    if not pending:
        return 0, overwritten, skipped_existing

    # The local checkpoint stores bf16 weights; preserve bf16 on CPU too to avoid doubling memory.
    encoder, tokenizer = _load_local_text_components(
        model_dir=model_dir,
        context_len=context_len,
        device=device,
        dtype=torch.bfloat16,
    )
    written = 0
    progress = f"Encoding HumanGen prompts (rank {rank}/{world_size})"
    with torch.inference_mode():
        for start in tqdm(range(0, len(pending), batch_size), desc=progress, unit="batch", disable=rank != 0):
            batch = pending[start:start + batch_size]
            text_batch = [prompt for prompt, _ in batch]
            ids, mask = tokenizer(text_batch, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context_batch = encoder(ids, mask)
            for index, (_, destination) in enumerate(batch):
                payload = {
                    "context": context_batch[index].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[index].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                }
                _atomic_torch_save(payload, destination)
                written += 1
    return written, overwritten, skipped_existing


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    is_distributed, rank, world_size, local_rank = _init_distributed()
    try:
        if cfg.get("data") is None:
            raise ValueError("A data config is required; run with data=humangen.")

        data_cfg = cfg.data.train
        if data_cfg.get("_target_") != "simpleicl.datasets.humangen.dataset.HumanGenMixtureDataset":
            raise ValueError("This script expects data=humangen.")
        dataset_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        dataset_cfg["enable_icl"] = False
        dataset_cfg["load_text_embedding_cache"] = False
        dataset_cfg["return_metadata"] = False
        dataset = instantiate(dataset_cfg)
        prompts = _unique_prompts(dataset)

        cache_dir = Path(str(data_cfg.text_embedding_cache_dir)).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = (Path.cwd() / cache_dir).resolve()
        model_dir = Path(str(cfg.get("human_text_model_dir", DEFAULT_MODEL_DIR))).expanduser()
        context_len = int(data_cfg.get("context_len", DEFAULT_CONTEXT_LEN))
        batch_size = int(cfg.get("human_text_batch_size", DEFAULT_BATCH_SIZE))
        if batch_size < 1:
            raise ValueError("human_text_batch_size must be >= 1")
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Local Wan model directory does not exist: {model_dir}")

        requested_device = cfg.get("human_text_device")
        if requested_device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "The Wan 5B text encoder cannot be loaded safely on CPU in this environment. "
                    "Run on a CUDA host, or explicitly set human_text_device=cpu if sufficient RAM is available."
                )
            device = f"cuda:{local_rank}" if is_distributed else "cuda"
        else:
            device = str(requested_device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"human_text_device={device} requested but CUDA is unavailable")
            if is_distributed and device.startswith("cuda"):
                device = f"cuda:{local_rank}"

        raw_overwrite = cfg.get("overwrite", False)
        overwrite = raw_overwrite if isinstance(raw_overwrite, bool) else str(raw_overwrite).strip().lower() in {"1", "true", "yes", "y"}
        if rank == 0:
            logger.info(
                "Precomputing %d unique prompts across %d rank(s) to %s using local model directory %s (device=%s, context_len=%d).",
                len(prompts), world_size, cache_dir, model_dir, device, context_len,
            )
        if not prompts:
            if rank == 0:
                logger.warning("No eligible prompts found in the selected HumanGen split.")
            return

        written, overwritten, skipped = precompute(
            cache_dir=cache_dir,
            prompts=prompts,
            model_dir=model_dir,
            context_len=context_len,
            batch_size=batch_size,
            device=device,
            overwrite=overwrite,
            rank=rank,
            world_size=world_size,
        )
        totals = _all_reduce_counts([written, overwritten, skipped], device)
        if rank == 0:
            logger.info(
                "Finished: written=%d overwritten=%d skipped_existing=%d unique_prompts=%d",
                totals[0], totals[1], totals[2], len(prompts),
            )
        if is_distributed:
            dist.barrier()
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
