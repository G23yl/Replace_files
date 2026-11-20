"""Independent LeRobot-v2.1 reader with paired human demonstration modalities."""

from __future__ import annotations

import bisect
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import fnmatch
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate

from .action import ACTION_DIM, map_robotwin, parse_transform, transform_vector
from .human_modalities import load_human_features, load_human_latent, load_video_modalities
from .normalization import load_stats, normalize_robot
from .manifest import ICLManifest
from .multiview import compose_three_views, t_layout_geometry

logger = logging.getLogger(__name__)
ROBOT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


@lru_cache(maxsize=16)
def _read_parquet_cached(path: Path) -> pd.DataFrame:
    """Cache complete episode tables per worker; callers must treat them as read-only."""
    return pd.read_parquet(path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_under(path: str | Path, base: Path) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else (base / result).resolve()


def _video_path(root: Path, info: dict, camera: str, episode: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk, file_index = divmod(episode, chunk_size)
    template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    relative = template.format(episode_chunk=chunk, episode_index=episode, chunk_index=chunk,
                               file_index=file_index, video_key=camera)
    path = root / relative
    if path.is_file():
        return path
    fallback = root / "videos" / camera / f"episode_{episode:06d}.mp4"
    return fallback if fallback.is_file() else path


def _read_robot_camera(
    path: Path | None, start: int, count: int, stride: int,
    size: tuple[int, int], slot_index: int, *, length: int,
):
    if path is None or not path.is_file():
        return None
    if length < 1:
        return None
    wanted = [min(start + i * stride, length - 1) for i in range(count)]
    wanted_set = set(wanted)
    decoded = {}
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in wanted_set:
                    decoded[frame_index] = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                if frame_index >= wanted[-1]:
                    break
    except av.error.FFmpegError as exc:
        raise RuntimeError(
            f"Robot video decode failed: path={path}, "
            f"start={start}, count={count}, stride={stride}, slot={slot_index}; cause={exc}"
        ) from exc
    missing = wanted_set - decoded.keys()
    if missing:
        raise RuntimeError(
            f"Robot video shorter than episode metadata: path={path}, "
            f"length={length}, missing_frames={sorted(missing)}, slot={slot_index}"
        )
    frames = [decoded[index] for index in wanted]
    video = torch.stack(frames).float().div_(127.5).sub_(1.0)
    slot_sizes, _ = t_layout_geometry(size)
    return F.interpolate(video, size=slot_sizes[slot_index], mode="bilinear", align_corners=False)


def _pad_repeat(array: torch.Tensor, target: int, *, dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    current = int(array.shape[dim])
    if current <= 0:
        raise ValueError("Cannot pad an empty sequence")
    if current > target:
        indices = torch.linspace(0, current - 1, target).round().long().to(array.device)
        array = array.index_select(dim, indices)
        mask = torch.ones(target, dtype=torch.bool)
    else:
        mask = torch.arange(target) < current
        if current < target:
            tail = array.select(dim, current - 1).unsqueeze(dim).expand(
                *[array.shape[i] if i != dim else target - current for i in range(array.ndim)]
            )
            array = torch.cat([array, tail], dim=dim)
    return array, mask


class HumanGenDataset(Dataset):
    """One source dataset. Robot video/action contract matches FastWAM build_inputs."""

    def __init__(
        self,
        name: str,
        robot_root: str,
        human_root: str,
        icl_config: str,
        *,
        project_root: str = ".",
        split: str = "train",
        num_frames: int = 33,
        global_sample_stride: int = 2,
        action_video_freq_ratio: int = 4,
        window_stride: int = 1,
        robot_video_size: tuple[int, int] = (384, 448),
        human_video_size: tuple[int, int] | None = (256, 448),
        camera_keys: list[str | None] | None = None,
        human_input_mode: str = "video",
        human_num_frames: int = 121,
        max_dino_objects: int = 4,
        enable_icl: bool = True,
        stats_paths: dict[str, str] | None = None,
        human_video_suffix: str = "generated_video_kling-v3.mp4",
        dino_suffix: str = "generated_video_kling-v3_dinov2.npz",
        hands_suffix: str = "generated_video_kling-v3_hands.pt",
        latent_root: str = "../data/human_latents",
        load_text_embedding_cache: bool = False,
        text_embedding_cache_dir: str = "cache/humangen/text_embeds",
        context_len: int = 128,
        normalization: str = "none",
        return_metadata: bool = False,
        excluded_task_names: list[str] | None = None,
        max_episodes: int | None = None,
    ):
        self.name = str(name)
        config_root = Path(project_root).expanduser().resolve()
        self.project_root = config_root
        self.robot_root = _resolve_under(robot_root, config_root)
        self.human_root = _resolve_under(human_root, config_root)
        self.latent_root = _resolve_under(latent_root, config_root) / self.name
        self.text_cache = _resolve_under(text_embedding_cache_dir, config_root)
        self.split = str(split)
        self.num_frames = int(num_frames)
        self.global_sample_stride = int(global_sample_stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.window_stride = int(window_stride)
        self.robot_video_size = tuple(map(int, robot_video_size))
        self.human_video_size = tuple(map(int, human_video_size)) if human_video_size is not None else None
        self._camera_keys_configured = camera_keys is not None
        configured_cameras = camera_keys if camera_keys is not None else [None, None, None]
        if len(configured_cameras) > 3:
            raise ValueError("camera_keys accepts at most three camera slots")
        self.camera_keys = list(configured_cameras) + [None] * (3 - len(configured_cameras))
        self.human_input_mode = str(human_input_mode)
        self.human_num_frames = int(human_num_frames)
        self.max_dino_objects = int(max_dino_objects)
        self.enable_icl = bool(enable_icl)
        self.normalization = str(normalization)
        self.return_metadata = bool(return_metadata)
        self.stats_paths = {str(key): _resolve_under(path, config_root) for key, path in (stats_paths or {}).items()}
        self.human_video_suffix, self.dino_suffix, self.hands_suffix = human_video_suffix, dino_suffix, hands_suffix
        self.load_text_embedding_cache = bool(load_text_embedding_cache)
        self.text_encoder_id = "wan22ti2v5b"
        self.context_len = int(context_len)
        self.robotwin_30d = self.name == "robotwin"
        self.excluded_task_names = {str(value) for value in (excluded_task_names or [])}
        self.stats_by_root = {}
        self.stats_path_by_root = {}
        if self.human_input_mode not in {"video", "latent"}:
            raise ValueError("human_input_mode must be 'video' or 'latent'")
        if self.global_sample_stride < 1:
            raise ValueError("global_sample_stride must be >= 1")
        if self.action_video_freq_ratio < 1:
            raise ValueError("action_video_freq_ratio must be >= 1")
        if self.human_num_frames < 1:
            raise ValueError("human_num_frames must be positive")
        if self.max_dino_objects < 1:
            raise ValueError("max_dino_objects must be positive")
        if not self.robot_root.is_dir():
            raise FileNotFoundError(f"Robot dataset root does not exist: {self.robot_root}")
        self.icl_config_path = _resolve_under(icl_config, self.project_root) if self.enable_icl else None
        self.manifest = (
            ICLManifest(self.icl_config_path, human_root=self.human_root)
            if self.enable_icl else None
        )
        self.records: list[dict[str, Any]] = []
        for meta_path in sorted(self.robot_root.rglob("meta/info.json")):
            root = meta_path.parent.parent
            info = json.loads(meta_path.read_text(encoding="utf-8"))
            split_spec = info.get("splits", {}).get(self.split)
            transform_path = next((candidate for candidate in (root / "meta/action_transform.yaml", root.parent / "meta/action_transform.yaml") if candidate.is_file()), None)
            if transform_path is None and not self.robotwin_30d:
                raise FileNotFoundError(f"Missing meta/action_transform.yaml for {root}")
            if transform_path is None:
                state_specs, action_specs, image_specs = {}, {}, {}
            else:
                import yaml
                transform = yaml.safe_load(transform_path.read_text(encoding="utf-8")) or {}
                state_specs, action_specs, image_specs = parse_transform(transform)
            source_camera_keys = list(self.camera_keys)
            if image_specs:
                inferred = []
                for spec in image_specs.values():
                    origin = spec.get("origin_keys")
                    if isinstance(origin, str):
                        inferred.append(origin)
                    elif origin:
                        inferred.extend(next(iter(item)) for item in origin)
                unique_cameras = list(dict.fromkeys(inferred))
                source_camera_keys = (unique_cameras[:3] + [None] * 3)[:3]
            elif not any(source_camera_keys):
                video_features = [key for key, feature in info.get("features", {}).items()
                                  if feature.get("dtype") == "video"]
                source_camera_keys = (video_features[:3] + [None] * 3)[:3]
            episodes = _read_jsonl(root / "meta/episodes.jsonl")
            for episode in episodes:
                episode_index = int(episode.get("episode_index", -1))
                length = int(episode.get("length", 0))
                task_names = [str(task) for task in episode.get("tasks", [])]
                task_root_name = root.name
                if any(task_root_name == excluded or task_root_name.startswith(excluded + "-") for excluded in self.excluded_task_names):
                    continue
                if episode_index < 0 or length <= 0 or not episode.get("action_config"):
                    continue
                if split_spec and not self._in_split(split_spec, episode_index):
                    continue
                parquet_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
                chunk, file_index = divmod(episode_index, int(info.get("chunks_size", 1000)))
                parquet = root / parquet_template.format(episode_chunk=chunk, episode_index=episode_index,
                                                        chunk_index=chunk, file_index=file_index, video_key="")
                if not parquet.is_file():
                    continue
                camera_paths = [
                    _video_path(root, info, camera, episode_index) if camera else None
                    for camera in source_camera_keys
                ]
                if not any(path is not None and path.is_file() for path in camera_paths):
                    continue
                primary_camera = source_camera_keys[0]
                primary_robot_path = (
                    _video_path(root, info, primary_camera, episode_index)
                    if primary_camera is not None else next(path for path in camera_paths if path is not None)
                )
                pair = None
                if self.enable_icl:
                    pair = self.manifest.lookup(primary_robot_path.relative_to(self.robot_root).as_posix())
                    if pair is None:
                        continue
                    pair["video_path"] = Path(pair["video_path"])
                    sample_dir = pair["video_path"].parent
                    pair["dino_path"] = sample_dir / self.dino_suffix
                    pair["hands_path"] = sample_dir / self.hands_suffix
                    pair["latent_path"] = self._latent_path(pair)
                    required_human_files = [pair["dino_path"], pair["hands_path"]]
                    if self.human_input_mode == "latent":
                        required_human_files.append(pair["latent_path"])
                    else:
                        required_human_files.append(pair["video_path"])
                    if any(not path.is_file() for path in required_human_files):
                        continue
                n_windows = max(1, ((length + self.global_sample_stride - 1) // self.global_sample_stride + self.window_stride - 1) // self.window_stride)
                self.records.append({"root": root, "info": info, "episode": episode,
                                     "episode_index": episode_index, "length": length,
                                     "parquet": parquet, "robot_video": next(path for path in camera_paths if path is not None and path.is_file()),
                                     "robot_camera_paths": camera_paths,
                                     "action_specs": action_specs, "state_specs": state_specs,
                                     "pair": pair, "n_windows": n_windows})
        if max_episodes is not None:
            self.records = self.records[: int(max_episodes)]
        self.offsets = [0]
        for record in self.records:
            self.offsets.append(self.offsets[-1] + record["n_windows"])
        if not self.records:
            raise ValueError(f"No usable episodes under {self.robot_root} (split={self.split}, enable_icl={self.enable_icl})")
        for record in self.records:
            if self.normalization == "none":
                continue
            root_name = record["root"].name
            matches = [(pattern, Path(str(path).replace("{root}", root_name)))
                       for pattern, path in self.stats_paths.items()
                       if pattern != "__source__" and fnmatch.fnmatchcase(root_name, pattern)]
            if len(matches) > 1:
                raise ValueError(f"Multiple stats_paths match task root {root_name}: {[key for key, _ in matches]}")
            if matches:
                pattern, path = matches[0]
            elif "__source__" in self.stats_paths:
                path = self.stats_paths["__source__"]
            elif self.normalization != "none":
                raise KeyError(f"No stats_paths entry matches task root {root_name!r}")
            else:
                continue
            self.stats_by_root[root_name] = load_stats(path)
            self.stats_path_by_root[root_name] = path

    @staticmethod
    def _column_array(frame, key):
        if key not in frame.columns:
            raise KeyError(f"Missing parquet column {key!r}; available={list(frame.columns)}")
        rows = []
        for value in frame[key].to_numpy():
            if hasattr(value, "as_py"):
                value = value.as_py()
            rows.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.stack(rows).astype(np.float32)

    @staticmethod
    def _in_split(spec, episode: int) -> bool:
        if isinstance(spec, str):
            for item in spec.split(","):
                if ":" in item:
                    start, end = item.split(":", 1)
                    if int(start or 0) <= episode < int(end or 2**63 - 1):
                        return True
                elif int(item) == episode:
                    return True
            return False
        if isinstance(spec, list):
            return episode in spec
        return True

    def _latent_path(self, pair: dict) -> Path:
        # Latent file naming follows the source video stem unless overridden by ICL metadata.
        raw = pair["pair"].get("human_latent_path")
        if raw:
            return _resolve_under(raw, self.project_root)
        rel = pair["video_path"].relative_to(self.human_root)
        return self.latent_root / rel.parent / (rel.stem + ".pth")

    def __len__(self):
        return self.offsets[-1]

    def _locate(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rec_index = bisect.bisect_right(self.offsets, index) - 1
        rec = self.records[rec_index]
        local = index - self.offsets[rec_index]
        return rec, local * self.window_stride

    def _text_context_from_cache(self, prompt: str):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_cache / f"{digest}.t5_len{self.context_len}.{self.text_encoder_id}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing prompt embedding cache {cache_path}; precompute embeddings first")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=False)
        context = payload["context"].clone().float()
        mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Text context must be [{self.context_len},D], got {tuple(context.shape)}")
        if mask.shape != (self.context_len,):
            raise ValueError(f"Text mask must be [{self.context_len}], got {tuple(mask.shape)}")
        context[~mask] = 0.0
        return context, torch.ones_like(mask)

    def _normalize_robot(self, values: torch.Tensor, mask: torch.Tensor, kind: str, record: dict) -> torch.Tensor:
        stats = self.stats_by_root.get(record["root"].name)
        if stats is None:
            raise KeyError(f"No stats_paths entry configured for task root {record['root'].name!r}")
        return normalize_robot(
            values, mask, stats, kind=kind, mode=self.normalization,
            path=str(self.stats_path_by_root[record["root"].name]),
        )

    def __getitem__(self, index):
        rec, start = self._locate(index)
        source_start = start * self.global_sample_stride
        requested_source_frames = (self.num_frames - 1) * self.global_sample_stride + 1
        source_end = min(source_start + requested_source_frames, rec["length"])
        table = _read_parquet_cached(rec["parquet"]).iloc[source_start:source_end]
        if table.empty:
            raise ValueError(f"Empty parquet window: {rec['parquet']} [{source_start}:{source_end}]")
        record = rec
        if self.robotwin_30d:
            actions_raw, action_mask_raw = map_robotwin(self._column_array(table, "action"))
            states_raw, state_mask_raw = map_robotwin(self._column_array(table, "observation.state"))
        else:
            actions_raw, action_mask_raw = transform_vector(
                table, rec["action_specs"], "action", num_rows=len(table)
            )
            states_raw, state_mask_raw = transform_vector(
                table, rec["state_specs"], "observation.state", num_rows=len(table),
                action_specs=rec["action_specs"],
            )
        video_positions = list(range(0, self.num_frames, self.action_video_freq_ratio))
        camera_views = [
            _read_robot_camera(
                path, source_start, len(video_positions),
                self.global_sample_stride * self.action_video_freq_ratio, self.robot_video_size, slot_index,
                length=rec["length"],
            )
            for slot_index, path in enumerate(rec["robot_camera_paths"])
        ]
        video = compose_three_views(camera_views, self.robot_video_size)
        # Dataset indices represent globally-strided observations. Align 32
        # action transitions and 33 proprio observations to 9 sampled video frames.
        states_sampled = states_raw[::self.global_sample_stride][:self.num_frames]
        actions_sampled = actions_raw[::self.global_sample_stride][:self.num_frames - 1]
        action_masks_sampled = action_mask_raw[::self.global_sample_stride][:self.num_frames - 1]
        state_masks_sampled = state_mask_raw[::self.global_sample_stride][:self.num_frames - 1]
        action = torch.from_numpy(actions_sampled).float()
        proprio = torch.from_numpy(states_sampled[: self.num_frames - 1]).float()
        action, action_pad_mask = _pad_repeat(action, self.num_frames - 1)
        proprio, proprio_pad_mask = _pad_repeat(proprio, self.num_frames - 1)
        action_mask = torch.zeros_like(action, dtype=torch.bool)
        proprio_mask = torch.zeros_like(proprio, dtype=torch.bool)
        action_mask[:len(action_masks_sampled)] = torch.from_numpy(action_masks_sampled)
        proprio_mask[:len(state_masks_sampled)] = torch.from_numpy(state_masks_sampled)
        action_mask &= action_pad_mask[:, None]
        proprio_mask &= proprio_pad_mask[:, None]
        if self.normalization != "none":
            action = self._normalize_robot(action, action_mask, "action", rec)
            proprio = self._normalize_robot(proprio, proprio_mask, "observation.state", rec)
        action[~action_mask] = 0.0
        proprio[~proprio_mask] = 0.0
        image_mask = torch.arange(len(video_positions)) * self.action_video_freq_ratio < math.ceil((source_end - source_start) / self.global_sample_stride)
        task = str((rec["episode"].get("tasks") or [rec["root"].name])[0])
        action_configs = rec["episode"].get("action_config") or []
        action_text = next((item.get("action_text") for item in action_configs
                            if isinstance(item, Mapping) and item.get("action_text")), None)
        prompt = str(action_text or task).strip()
        if not prompt:
            raise ValueError(f"Episode {rec['episode_index']} has empty action_text and task")
        # Apply classifier-free text conditioning only to ICL samples; the empty
        # prompt has its own cached embedding.
        if self.enable_icl and np.random.random() < 0.4:
            prompt = ""
        if self.load_text_embedding_cache:
            context, context_mask = self._text_context_from_cache(prompt)
        else:
            context, context_mask = None, None
        sample = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": ~image_mask,
            "action_is_pad": ~action_mask,
            "proprio_is_pad": ~proprio_mask,
        }
        if not self.enable_icl:
            if self.return_metadata:
                sample["metadata"] = {"dataset_name": self.name, "episode_index": rec["episode_index"]}
            return sample
        pair = rec["pair"]
        metadata = None
        if self.return_metadata:
            metadata = {
                "dataset_name": self.name,
                "task_name": rec["root"].name,
                "episode_index": rec["episode_index"],
                "robot_video_path": str(rec["robot_video"]),
                "human_video_path": str(pair["video_path"]),
                "human_sample_id": pair["sample_id"],
                "human_dino_path": str(pair["dino_path"]),
                "human_hands_path": str(pair["hands_path"]),
            }
        if self.human_input_mode == "video":
            human, human_meta = load_video_modalities(
                pair["video_path"], pair["dino_path"], pair["hands_path"],
                target_frames=self.human_num_frames, video_size=self.human_video_size,
                max_dino_objects=self.max_dino_objects,
            )
            if human["human_hands"].shape != (self.human_num_frames, 2, 25, 3):
                raise ValueError(f"hands temporal/feature shape mismatch for sample index {index}: {human['human_hands'].shape}")
            if (human["human_dino"].ndim != 3
                    or human["human_dino"].shape[0] != self.human_num_frames
                    or human["human_dino"].shape[-1] != 2048):
                raise ValueError(f"DINO temporal/feature shape mismatch for sample index {index}: {human['human_dino'].shape}")
            sample.update(human)
            if metadata is not None:
                metadata["human_meta"] = human_meta
        else:
            latent, _ = load_human_latent(pair["latent_path"])
            human, human_meta = load_human_features(
                pair["dino_path"], pair["hands_path"], target_frames=self.human_num_frames,
                max_dino_objects=self.max_dino_objects,
            )
            if human["human_hands"].shape != (self.human_num_frames, 2, 25, 3):
                raise ValueError(f"hands temporal/feature shape mismatch for sample index {index}: {human['human_hands'].shape}")
            if (human["human_dino"].ndim != 3
                    or human["human_dino"].shape[0] != self.human_num_frames
                    or human["human_dino"].shape[-1] != 2048):
                raise ValueError(f"DINO temporal/feature shape mismatch for sample index {index}: {human['human_dino'].shape}")
            human["human_latents"] = latent
            sample.update(human)
            if metadata is not None:
                metadata["human_latent_path"] = str(pair["latent_path"])
                metadata["human_meta"] = human_meta
        if metadata is not None:
            sample["metadata"] = metadata
        return sample

    def _get_worker_stats(self):
        return {"source": self.name, "episodes": len(self.records), "windows": len(self),
                "mapped_pairs": sum(rec["pair"] is not None for rec in self.records)}


class HumanGenMixtureDataset(Dataset):
    """Concatenate source readers; keep each source boundary and metadata."""

    def __init__(self, sources: list[str] | list[dict], datasets: dict[str, dict], *, project_root="../", **shared):
        specs = []
        for source in sources:
            if isinstance(source, str):
                if source not in datasets:
                    raise KeyError(f"Unknown HumanGen source {source!r}; configured={sorted(datasets)}")
                spec = dict(datasets[source])
                spec.setdefault("name", source)
            else:
                spec = dict(source)
            options = dict(shared)
            options.update(spec)
            options.setdefault("project_root", project_root)
            specs.append(options)
        # Source scans are independent I/O-heavy work; map preserves configured order.
        with ThreadPoolExecutor(max_workers=max(1, len(specs))) as pool:
            self.datasets = list(pool.map(lambda options: HumanGenDataset(**options), specs))
        self.offsets = [0]
        for dataset in self.datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))
        self.stats = [dataset._get_worker_stats() for dataset in self.datasets]

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = bisect.bisect_right(self.offsets, index) - 1
        return self.datasets[source_index][index - self.offsets[source_index]]

    @staticmethod
    def collate_fn(batch):
        # Human modalities are top-level fields; keep optional context/metadata handling explicit.
        contexts = [sample.get("context") for sample in batch]
        context_masks = [sample.get("context_mask") for sample in batch]
        has_metadata_key = any("metadata" in sample for sample in batch)
        collated = default_collate([
            {key: value for key, value in sample.items()
             if key not in {"metadata", "context", "context_mask"}}
            for sample in batch
        ])
        collated["context"] = default_collate(contexts) if all(value is not None for value in contexts) else None
        collated["context_mask"] = default_collate(context_masks) if all(value is not None for value in context_masks) else None
        if has_metadata_key:
            collated["metadata"] = [sample.get("metadata") for sample in batch]
        return collated

    @classmethod
    def from_config(cls, cfg):
        return cls(**dict(cfg))
