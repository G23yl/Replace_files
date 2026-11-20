# SkeleICL

**Contributor:** SimpleICL integration | **Original code:** local upstream at `/mnt/workspace/tusifan/wm/guood/simple-icl/icl_model`

SkeleICL selects its upstream task config through `upstream_model_task_config` in `deploy.yml` (default: `skeleicl_humangen`). The composed model's `video_expert.has_video_ctx_condition` and `video_ctx_type` settings determine whether/how human conditions are loaded. For `video_ctx_type: skeleicl`, the adapter resolves the task's HumanGen video through `human_video_map_path` and loads the configured DINO and hands files from that video's directory. The adapter instantiates the model using the composed `model._target_`; set `upstream_config_dir` or `upstream_config_overrides` when using another config composition. The HumanGen task uses the 30D Robotwin action representation.

The adapter supports simulation `action_type=ee`; it does not restrict `env_cfg_type` to the real-robot `franka` config. The selected environment must describe a dual-arm robot with one gripper channel per arm, and its observations must provide 7D EE poses. The upstream model's HumanGen representation maps those EE poses and grippers into a 30D vector and back.

## Installation

The policy server environment needs the upstream `icl_model` checkout, its Python dependencies, XPolicyLab dependencies, and Wan2.2 model weights. When using an existing pip environment, install directly into that environment:

```bash
python -m pip install -e /path/to/icl_model
python -m pip install -e /path/to/RoboTwin/XPolicyLab
```

The optional `install.sh` creates a dedicated conda environment; it is not used by evaluation. Server and client launch scripts use the active Python interpreter and accept the XPolicyLab environment-label arguments only for CLI compatibility. Install or make Wan2.2 base weights available under `<icl_model>/checkpoints/` and keep `DIFFSYNTH_MODEL_BASE_PATH` pointing there. The adapter does not download large model weights automatically.

## Data Processing

SkeleICL consumes HumanGen directly through `icl_model/configs/data/humangen.yaml`; `process_data.sh` does not convert or reshape data. This is not a LeRobot dataset version supported by the XPolicyLab official v2.1/v3.0 converters: the HumanGen dataset uses its own Robotwin manifest, 30D action mapping, grouped action statistics, and paired human-video layout. Prepare the HumanGen Robotwin robot data, ICL manifest and statistics at the paths in the upstream config. For SimpleICL, training requires the paired video/DINO/hands or latent artifacts expected by `human_input_mode`.

The SimpleICL evaluation video map defaults to `icl_model/scripts/robotwin_icl_human_videos.py`. The known task keys are `place_object_scale`, `stamp_seal`, `open_microwave`, `move_stapler_pad`, `stack_blocks_three`, `place_bread_basket`, and `place_empty_cup`. Set `human_video_map_path` in `deploy.yml` when using a mapping file other than the upstream default. Task evaluation names are matched after removing `_task` / `_variation` suffixes.

```bash
bash process_data.sh RoboTwin my_run <env_cfg_type> ee simpleicl
```

## Training

Training delegates to the upstream HumanGen task configs and preflight. `model_type` is the final optional argument:

```bash
bash train.sh RoboTwin my_run <env_cfg_type> ee 0 0,1,2,3 simpleicl
bash train.sh RoboTwin my_baseline <env_cfg_type> ee 0 0,1,2,3 fastwam
```

Upstream `simpleicl_humangen` enables ICL and latent-mode human conditions; `fastwam_humangen` disables ICL. Dataset roots and caches remain configured in `icl_model/configs/data/humangen.yaml`. Checkpoints are written under `checkpoints/RoboTwin-<ckpt_name>-<env_cfg_type>-ee-<seed>/`, with upstream `config.yaml`, `dataset_stats.json`, and `checkpoints/weights/step_*.pt` artifacts. At evaluation, the model architecture/task config is composed from `upstream_model_task_config`, `upstream_config_dir`, and `upstream_config_overrides`; the saved run `config.yaml` is not read by the adapter. HumanGen text conditioning uses the raw `action_text` from the episode (falling back to its task name), matching `HumanGenDataset` and `precompute_humangen_text_embeds.py`; no robot-video description prefix is added.

## Evaluation

Set the default `upstream_model_task_config` in `deploy.yml` to the task config used by the checkpoint, or override it per launch with `SKELEICL_UPSTREAM_MODEL_TASK_CONFIG`. This allows separate model-task runs without editing the shared config:

```bash
SKELEICL_UPSTREAM_MODEL_TASK_CONFIG=fastwam_humangen bash eval.sh RoboTwin place_object_scale <fastwam_checkpoint> <env_cfg_type> ee 0 0 0 skeleicl base
```

Then run:

```bash
bash eval.sh RoboTwin place_object_scale RoboTwin-my_run-<env_cfg_type>-ee-0 <env_cfg_type> ee 0 0 0 skeleicl base
```

`EVAL_ENV_TYPE=debug` uses the standard XPolicyLab debug environment. SimpleICL's mapped HumanGen video is encoded with the upstream condition VAE once at policy startup. The robot observation must provide RGB `cam_high`, `cam_left_wrist`, and `cam_right_wrist` images; the adapter resizes and T-concatenates them like FastWAM (head above the two wrist views). It also requires dual-arm 7D EE poses plus one gripper value per arm. `action_horizon` controls the predicted chunk length; `replan_steps` controls the returned prefix and how many environment steps execute before the next observation and inference. It must be between 1 and `action_horizon` (default 24).

## Limitations

- Wan2.2 inference requires a CUDA-capable GPU and the upstream base model weights.
- Only dual-arm EE checkpoints with 7D poses and one gripper channel per arm are supported; `env_cfg_type` is resolved through XPolicyLab's shared robot metadata.
- SimpleICL evaluation uses one configured HumanGen clip per task, as listed in the upstream video mapping.
- Batched environments currently run independent model inference calls; GPU throughput is not batched.
- Simulator and debug-mode evaluation have not been validated as part of this adapter change.
