from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    unpack_robot_state,
)

POLICY_DIR = Path(__file__).resolve().parent
IMPORTABLE_ROOT = Path(__file__).resolve().parents[2]


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_path(value: Any, base: Path) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _task_name(value: Any) -> str:
    name = str(value or "").strip().lower()
    for suffix in ("_task", "_variation"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _load_human_video_map(path: Path) -> dict[str, list[str]]:
    spec = importlib.util.spec_from_file_location("skeleicl_human_video_map", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import human video map: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping = getattr(module, "ROBOTWIN_ICL_HUMAN_VIDEOS", None)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} must define ROBOTWIN_ICL_HUMAN_VIDEOS")
    return {str(key): [str(item) for item in values] for key, values in mapping.items()}


def _standardize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    width, height = size
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def _build_robotwin_image(vision: dict) -> np.ndarray:
    head_key = "cam_head" if "cam_head" in vision else "cam_high"
    head = _standardize_rgb(vision[head_key]["color"], (320, 256))
    left = _standardize_rgb(vision["cam_left_wrist"]["color"], (160, 128))
    right = _standardize_rgb(vision["cam_right_wrist"]["color"], (160, 128))
    return np.ascontiguousarray(
        np.concatenate((head, np.concatenate((left, right), axis=1)), axis=0)
    )


def _instruction(obs: dict, fallback: str) -> str:
    value = obs.get("instruction", obs.get("instructions", fallback))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value or "").strip()
    return text or fallback


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.upstream_model_task_config = str(
            self.model_cfg.get("upstream_model_task_config") or "skeleicl_humangen"
        )
        self.action_type = str(self.model_cfg["action_type"])
        if self.action_type != "ee":
            raise ValueError("SkeleICL Robotwin checkpoints support action_type='ee' only")
        self.env_cfg_type = str(self.model_cfg["env_cfg_type"])
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        if len(self.robot_action_dim_info["arm_dim"]) != 2 or self.robot_action_dim_info["ee_dim"] != [1, 1]:
            raise ValueError("SkeleICL Robotwin requires a dual-arm robot with one gripper value per arm")
        self.model_root = _resolve_path(self.model_cfg.get("model_root"), POLICY_DIR)
        if self.model_root is None or not self.model_root.is_dir():
            raise FileNotFoundError(f"Upstream model_root does not exist: {self.model_root}")
        self.checkpoints_dir = POLICY_DIR / "checkpoints"
        self.run_root = resolve_checkpoint_root(
            self.model_cfg,
            self.checkpoints_dir,
            policy_dir=POLICY_DIR,
        )
        if self.run_root.is_file():
            self.run_root = self.run_root.parent
        self.checkpoint_path = _resolve_path(self.model_cfg.get("checkpoint_path"), self.run_root)
        if self.checkpoint_path is None:
            weights = self.run_root / "checkpoints" / "weights"
            candidates = sorted(weights.glob("step_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No step_*.pt checkpoint under {weights}")
            self.checkpoint_path = candidates[-1]
        if self.checkpoint_path.is_dir():
            candidates = sorted((self.checkpoint_path / "checkpoints" / "weights").glob("step_*.pt"))
            if not candidates:
                candidates = sorted(self.checkpoint_path.glob("step_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No step_*.pt checkpoint in {self.checkpoint_path}")
            self.checkpoint_path = candidates[-1]
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        stats_value = self.model_cfg.get("dataset_stats_path")
        self.stats_path = _resolve_path(stats_value, self.run_root) or (self.run_root / "dataset_stats.json")
        if not self.stats_path.is_file():
            raise FileNotFoundError(f"HumanGen action statistics do not exist: {self.stats_path}")
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        self.stats = self.stats.get("norm_stats", self.stats)
        self.normalization = str(self.model_cfg.get("normalization", "zscore"))
        self.valid_layout = self._get_valid_layout(self.action_type)

        self.device = str(self.model_cfg.get("device", "cuda"))
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SkeleICL requires CUDA for Wan2.2 inference")
        precision = str(self.model_cfg.get("mixed_precision", "bf16")).lower()
        self.torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}.get(precision)
        if self.torch_dtype is None:
            raise ValueError("mixed_precision must be bf16, fp16, or no")
        self.seed_inference = int(self.model_cfg.get("seed_inference", 0))
        self.num_inference_steps = int(self.model_cfg.get("num_inference_steps", 10))
        self.action_horizon = int(self.model_cfg.get("action_horizon", 32))
        self.replan_steps = int(self.model_cfg.get("replan_steps", 24))
        if self.action_horizon <= 0 or not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                "Expected 1 <= replan_steps <= action_horizon, got "
                f"replan_steps={self.replan_steps}, action_horizon={self.action_horizon}"
            )
        self.rand_device = str(self.model_cfg.get("rand_device", "cpu"))
        self.tiled = _is_true(self.model_cfg.get("tiled", False))
        self.default_instruction = "follow the instruction"
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.batch_obs = {}
        self.batch_instruction = {}
        self.video_ctx_latents = None
        self.human_modalities = None
        self.video_map = {}

        self._load_upstream_model()
        self.has_video_ctx_condition = bool(
            getattr(getattr(self.model, "video_expert", None), "has_video_ctx_condition", False)
        )
        self.video_ctx_type = str(
            getattr(getattr(self.model, "video_expert", None), "video_ctx_type", "simpleicl")
        )
        if self.has_video_ctx_condition:
            map_path = _resolve_path(
                self.model_cfg.get("human_video_map_path"),
                self.model_root,
            ) or (self.model_root / "scripts" / "robotwin_icl_human_videos.py")
            if not map_path.is_file():
                raise FileNotFoundError(f"Human video mapping does not exist: {map_path}")
            self.video_map = _load_human_video_map(map_path)
            if self.video_ctx_type == "simpleicl":
                self.video_ctx_latents = self._load_video_ctx_latents()
            elif self.video_ctx_type == "skeleicl":
                self.human_modalities = self._load_human_modalities()
            else:
                raise ValueError(f"Unsupported video_ctx_type={self.video_ctx_type!r}")

        print(
            "[SkeleICL] Initialization complete\\n"
            f"  model_task_config: {self.upstream_model_task_config}\\n"
            f"  checkpoint: {self.checkpoint_path}\\n"
            f"  stats: {self.stats_path}\\n"
            f"  action_horizon: {self.action_horizon}\\n"
            f"  replan_steps: {self.replan_steps}\\n"
            f"  action_dim: {self.action_dim}\\n"
            f"  device: {self.device}\\n"
            f"  video_ctx_type: {self.video_ctx_type if self.has_video_ctx_condition else 'disabled'}",
            flush=True,
        )

    def _load_upstream_model(self):
        root = self.model_root
        for path in (str(root / "src"), str(root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import hydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config_dir_value = self.model_cfg.get("upstream_config_dir")
        config_dir = _resolve_path(config_dir_value, root) or (root / "configs")
        if not config_dir.is_dir():
            raise FileNotFoundError(f"Upstream Hydra config directory does not exist: {config_dir}")
        task_name = self.upstream_model_task_config
        overrides = list(self.model_cfg.get("upstream_config_overrides") or [])
        overrides.append(f"task={task_name}")
        with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = hydra.compose(config_name="sim_robotwin", overrides=overrides)

        model_cfg = cfg.model
        if model_cfg.get("_target_") is None and model_cfg.get("__target__") is not None:
            model_cfg["_target_"] = model_cfg.pop("__target__")
        if not model_cfg.get("_target_"):
            raise ValueError(
                f"Hydra-composed model config for task {task_name!r} must declare _target_"
            )
        print(
            "[SkeleICL] Composed upstream model config:\n"
            + OmegaConf.to_yaml(model_cfg, resolve=True),
            flush=True,
        )
        self.model = instantiate(model_cfg, device=self.device, model_dtype=self.torch_dtype)
        self.model.load_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        if torch.device(self.device).type == "cuda":
            # Upstream loads checkpoint tensors onto CUDA before copying them
            # into the model. Return the freed loading buffers to the driver so
            # the simulator/renderer in another process can use that memory.
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def _mapped_human_video_path(self) -> Path:
        key = _task_name(self.model_cfg.get("task_name"))
        choices = self.video_map.get(key)
        if not choices:
            raise KeyError(f"No HumanGen video mapping for task {key!r}; available: {sorted(self.video_map)}")
        video_path = Path(choices[0]).expanduser()
        if not video_path.is_absolute():
            video_path = (self.model_root / video_path).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"HumanGen video does not exist: {video_path}")
        return video_path

    def _load_video_ctx_latents(self):
        video_path = self._mapped_human_video_path()
        print(f"[SimpleICL] Use human video: {video_path}")
        import av
        from simpleicl.datasets.humangen.human_modalities import prepare_human_video_for_vae

        frames = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(video=0):
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
        if not frames:
            raise ValueError(f"HumanGen video has no decodable frames: {video_path}")
        target_frames = int(self.model_cfg.get("human_video_num_frames", 121))
        size = tuple(map(int, self.model_cfg.get("human_video_size", [320, 448])))
        video, _ = prepare_human_video_for_vae(torch.stack(frames), target_frames=target_frames, video_size=size)
        latent = self.model._encode_cond_latents(video.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype), tiled=self.tiled)
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        return latent

    def _load_human_modalities(self) -> dict[str, torch.Tensor]:
        video_path = self._mapped_human_video_path()
        sample_dir = video_path.parent
        dino_path = sample_dir / str(self.model_cfg.get("human_dino_suffix", "generated_video_kling-v3_dinov2.npz"))
        hands_path = sample_dir / str(self.model_cfg.get("human_hands_suffix", "generated_video_kling-v3_hands.pt"))
        if not dino_path.is_file():
            raise FileNotFoundError(f"HumanGen DINO features do not exist: {dino_path}")
        if not hands_path.is_file():
            raise FileNotFoundError(f"HumanGen hand features do not exist: {hands_path}")

        print(f"[SkeleICL] Use dino path: {dino_path}, use hands path: {hands_path}")

        from simpleicl.datasets.humangen.human_modalities import load_human_features

        modalities, _ = load_human_features(
            dino_path,
            hands_path,
            target_frames=int(self.model_cfg.get("human_video_num_frames", 121)),
            max_dino_objects=int(
                getattr(self.model.video_expert.video_ctx_qformer, "num_objects", 4)
            ),
        )
        expected = {"human_hands", "human_hands_valid", "human_dino", "human_dino_valid"}
        missing = expected - modalities.keys()
        if missing:
            raise ValueError(f"SkeleICL HumanGen features are missing: {sorted(missing)}")
        return {
            key: value.unsqueeze(0).to(
                device=self.device,
                dtype=torch.bool if key.endswith("valid") else self.torch_dtype,
            )
            for key, value in modalities.items()
        }

    @staticmethod
    def _get_valid_layout(action_type: str) -> dict[str, tuple[int, int]]:
        """Return active 30D HumanGen layout groups for the robot action type."""
        if action_type == "ee":
            return {"hand.position": (0, 14), "effector.position": (28, 2)}
        if action_type == "joint":
            return {"arm.position": (14, 14), "effector.position": (28, 2)}
        raise ValueError(f"Unsupported action_type={action_type!r}")

    def _apply_stats(
        self, values: np.ndarray, *, kind: str, inverse: bool
    ) -> np.ndarray:
        """Normalize or denormalize active layout groups, leaving inactive groups zero."""
        result = np.zeros_like(np.asarray(values, dtype=np.float32))
        if result.shape[-1] != 30:
            raise ValueError(f"Expected last dimension 30, got {result.shape}")
        for group, (offset, width) in self.valid_layout.items():
            block_values = np.asarray(values, dtype=np.float32)[..., offset:offset + width]
            if self.normalization == "none":
                transformed = block_values
            else:
                block = self.stats.get(f"{kind}.{group}")
                if block is None:
                    raise KeyError(f"Stats are missing active group {kind}.{group}")
            if self.normalization == "zscore":
                center = np.asarray(block["mean"], dtype=np.float32)
                scale = np.asarray(block["std"], dtype=np.float32)
                transformed = block_values * (scale + 1e-8) + center if inverse else (block_values - center) / (scale + 1e-8)
            elif self.normalization == "minmax":
                low = np.asarray(block["min"], dtype=np.float32)
                high = np.asarray(block["max"], dtype=np.float32)
                if np.any(high <= low):
                    raise ValueError(f"Stats group {kind}.{group} has max <= min")
                transformed = (block_values + 1.0) * 0.5 * (high - low) + low if inverse else 2.0 * (block_values - low) / (high - low) - 1.0
            elif self.normalization == "none":
                transformed = block_values
            else:
                raise ValueError(f"Unsupported normalization mode {self.normalization!r}")
            if not inverse and self.normalization != "none":
                transformed = np.clip(transformed, -5.0, 5.0)
            result[..., offset:offset + width] = transformed
        return result

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        return self._apply_stats(state, kind="observation.state", inverse=False)

    def _encode_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image = _build_robotwin_image(obs["vision"])
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).mul_(2.0).sub_(1.0)
        image_tensor = image_tensor.unsqueeze(0)
        state_dict = obs["state"]
        left_pose = np.asarray(state_dict["left_ee_pose"], dtype=np.float32).reshape(-1)
        right_pose = np.asarray(state_dict["right_ee_pose"], dtype=np.float32).reshape(-1)
        left_gripper = np.asarray(state_dict["left_ee_joint_state"], dtype=np.float32).reshape(-1)
        right_gripper = np.asarray(state_dict["right_ee_joint_state"], dtype=np.float32).reshape(-1)
        if (left_pose.size, right_pose.size, left_gripper.size, right_gripper.size) != (7, 7, 1, 1):
            raise ValueError("Expected dual-arm state with 7D EE poses and one gripper value per arm")
        state = np.zeros(30, dtype=np.float32)
        state[:7] = left_pose
        state[7:14] = right_pose
        state[28:30] = np.concatenate((left_gripper, right_gripper))
        state = self._normalize_state(state)
        return image_tensor, torch.from_numpy(state)

    def update_obs(self, obs):
        self.last_obs = self._encode_obs(obs)
        self.last_instruction = _instruction(obs, self.default_instruction)

    def update_obs_batch(self, obs_list):
        self.batch_obs = {}
        self.batch_instruction = {}
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            self.batch_obs[env_idx] = self._encode_obs(obs)
            self.batch_instruction[env_idx] = _instruction(obs, self.default_instruction)

    def _denormalize(self, actions: np.ndarray) -> np.ndarray:
        return self._apply_stats(actions, kind="action", inverse=True)

    def _infer(self, state: tuple[torch.Tensor, torch.Tensor], instruction: str) -> list[dict]:
        if state is None:
            raise ValueError("Call update_obs() before get_action()")
        image, proprio = state
        prompt = instruction
        result = self.model.infer_action(
            prompt=prompt,
            input_image=image.to(device=self.device, dtype=self.torch_dtype),
            action_horizon=self.action_horizon,
            proprio=proprio.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype),
            num_inference_steps=self.num_inference_steps,
            seed=self.seed_inference,
            rand_device=self.rand_device,
            tiled=self.tiled,
            video_ctx_latents=self.video_ctx_latents,
            human_modalities=self.human_modalities,
        )
        action = self._denormalize(result["action"].detach().float().cpu().numpy())
        if action.ndim != 2 or action.shape[-1] != 30:
            raise ValueError(f"Expected HumanGen action with shape [T, 30], got {action.shape}")
        action = action[:self.replan_steps, :]
        robot_action = np.concatenate((action[:, :7], action[:, 28:29], action[:, 7:14], action[:, 29:30]), axis=-1)
        return unpack_robot_state(robot_action, "ee", self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        return self._infer(self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list=None):
        indices = list(self.batch_obs) if env_idx_list is None else [int(i) for i in env_idx_list]
        return [self._infer(self.batch_obs[i], self.batch_instruction[i]) for i in indices]

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.batch_obs.clear()
        self.batch_instruction.clear()
        # FastWAM inference is stateless: its video KV cache is local to
        # infer_action(). Episode state belongs to this adapter, not the model.
