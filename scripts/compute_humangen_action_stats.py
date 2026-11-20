#!/usr/bin/env python3
"""Compute HumanGen 30D absolute action/state normalization statistics.

Run from ``icl_model/``. The output JSON is compatible with the HumanGen
normalization loader and can be written once per action-transform owner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import yaml

from simpleicl.datasets.humangen.action import (
    ACTION_DIM,
    LAYOUT,
    map_robotwin,
    parse_transform,
    transform_vector,
)


class _Accumulator:
    """Online moments and bounded reservoir for approximate quantiles."""

    def __init__(self, cap: int, seed: int):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.sample = np.empty(cap, dtype=np.float64)
        self.sample_count = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size:
            return
        batch_count = len(values)
        batch_mean = float(values.mean())
        batch_m2 = float(np.square(values - batch_mean).sum())
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        start = self.count
        self.count = total

        # Vectorized reservoir replacement avoids a Python loop per data point.
        fill_count = min(batch_count, max(0, self.cap - start))
        if fill_count:
            self.sample[start:start + fill_count] = values[:fill_count]
            self.sample_count += fill_count
        remaining = values[fill_count:]
        if remaining.size:
            first_seen = start + fill_count
            seen = np.arange(first_seen + 1, total + 1, dtype=np.int64)
            indices = self.rng.integers(0, seen)
            replace = indices < self.cap
            self.sample[indices[replace]] = remaining[replace]

    def finalize(self) -> dict:
        sample = self.sample[: self.sample_count]
        q01, q99 = np.quantile(sample, [0.01, 0.99])
        return {
            "mean": self.mean,
            "std": float(np.sqrt(self.m2 / self.count)),
            "min": self.minimum,
            "max": self.maximum,
            "q01": float(q01),
            "q99": float(q99),
        }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _in_split(spec, episode: int) -> bool:
    if spec is None:
        return True
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


def _is_robotwin(root: Path) -> bool:
    return root.name == "robotwin" or root.parent.name == "robotwin_data" or "robotwin" in root.name


def _transform_for(root: Path, robotwin: bool) -> dict:
    candidates = (root / "meta/action_transform.yaml", root.parent / "meta/action_transform.yaml")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        if robotwin:
            return {}
        raise FileNotFoundError(f"Missing meta/action_transform.yaml for {root}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parquet_path(root: Path, info: dict, episode: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk, file_index = divmod(episode, chunk_size)
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    relative = template.format(
        episode_chunk=chunk,
        episode_index=episode,
        chunk_index=chunk,
        file_index=file_index,
        video_key="",
    )
    return root / relative


def _robotwin_column(frame: pd.DataFrame, key: str) -> np.ndarray:
    if key not in frame.columns:
        raise KeyError(f"Missing parquet feature {key!r}; available={list(frame.columns)}")
    rows = []
    for cell in frame[key].to_numpy():
        if hasattr(cell, "as_py"):
            cell = cell.as_py()
        rows.append(np.asarray(cell, dtype=np.float32).reshape(-1))
    return np.stack(rows)


def _mapped_rows(frame, state_specs, action_specs, robotwin):
    if robotwin:
        action, action_mask = map_robotwin(_robotwin_column(frame, "action"))
        state, state_mask = map_robotwin(_robotwin_column(frame, "observation.state"))
        return action, action_mask, state, state_mask

    # Mirror the reader's state shift while providing the preceding-row sentinel.
    state_frame = pd.concat([frame.iloc[[0]], frame], ignore_index=True)
    state, state_mask = transform_vector(
        state_frame,
        state_specs,
        "observation.state",
        origin_offset=1,
        num_rows=len(frame),
        action_specs=action_specs,
    )
    action, action_mask = transform_vector(frame, action_specs, "action", num_rows=len(frame))
    return action, action_mask, state, state_mask


def _new_accumulators(cap: int):
    return {
        kind: [_Accumulator(cap, seed=offset + index) for index in range(ACTION_DIM)]
        for offset, kind in ((0, "action"), (ACTION_DIM, "observation.state"))
    }


def _update(accumulators, kind: str, values: np.ndarray, mask: np.ndarray) -> None:
    for dim, accumulator in enumerate(accumulators[kind]):
        accumulator.update(values[mask[:, dim], dim])


def _finalize(accumulators, kind: str) -> dict:
    blocks = {}
    offset = 0
    for group, width in LAYOUT:
        active = [dim for dim in range(offset, offset + width) if accumulators[kind][dim].count]
        if active:
            finalized = {dim: accumulators[kind][dim].finalize() for dim in active}
            blocks[group] = {
                key: [finalized[dim][key] for dim in active]
                for key in ("mean", "std", "min", "max", "q01", "q99")
            }
        offset += width
    return blocks


def _transform_path(root: Path) -> Path | None:
    for candidate in (root / "meta/action_transform.yaml", root.parent / "meta/action_transform.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _iter_parquet_roots(dataset_dir: Path, source: str) -> list[Path]:
    roots = sorted(path.parent.parent for path in dataset_dir.rglob("meta/info.json"))
    return roots


def compute_stats(dataset_dir: Path, source: str, split: str, max_roots: int | None,
                  max_rows: int | None, reservoir_cap: int, roots: list[Path] | None = None) -> dict:
    if roots is None:
        roots = _iter_parquet_roots(dataset_dir, source)
    if max_roots is not None:
        roots = roots[:max_roots]
    accumulators = _new_accumulators(reservoir_cap)
    root_count = episode_count = action_rows = state_rows = 0

    for root in roots:
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        robotwin = _is_robotwin(root)
        state_specs, action_specs, _ = parse_transform(_transform_for(root, robotwin))
        split_spec = info.get("splits", {}).get(split)
        for episode in _read_jsonl(root / "meta/episodes.jsonl"):
            episode_index = int(episode.get("episode_index", -1))
            if episode_index < 0 or int(episode.get("length", 0)) <= 0:
                continue
            if not episode.get("action_config") or not _in_split(split_spec, episode_index):
                continue
            path = _parquet_path(root, info, episode_index)
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            if len(frame) < 2:
                continue
            action, action_mask, state, state_mask = _mapped_rows(
                frame, state_specs, action_specs, robotwin
            )
            # The last observation has no corresponding action target.
            action, action_mask = action[:-1], action_mask[:-1]
            state, state_mask = state[:-1], state_mask[:-1]
            if max_rows is not None:
                remaining = max_rows - action_rows
                if remaining <= 0:
                    break
                action, action_mask = action[:remaining], action_mask[:remaining]
                state, state_mask = state[:len(action)], state_mask[:len(action)]
            _update(accumulators, "action", action, action_mask)
            _update(accumulators, "observation.state", state, state_mask)
            action_rows += len(action)
            state_rows += len(state)
            episode_count += 1
        root_count += 1
        if max_rows is not None and action_rows >= max_rows:
            break

    if not action_rows or not state_rows:
        raise ValueError(f"{source}: no valid action/state rows found")
    return {
        "source": source,
        "split": split,
        "norm_stats": {
            **{f"action.{group}": block for group, block in _finalize(accumulators, "action").items()},
            **{f"observation.state.{group}": block
               for group, block in _finalize(accumulators, "observation.state").items()},
        },
        "num_task_roots": root_count,
        "num_episodes": episode_count,
        "num_action_rows": action_rows,
        "num_state_rows": state_rows,
    }


def _stats_groups(dataset_dir: Path, source: str):
    roots = _iter_parquet_roots(dataset_dir, source)
    if source in {"agibot", "robotwin"}:
        return [(source, dataset_dir, roots)]
    groups = {}
    for root in roots:
        transform_path = _transform_path(root)
        owner = transform_path.parent.parent if transform_path else root
        groups.setdefault(owner, []).append(root)
    return [(f"{source}__{owner.name}", owner, group_roots)
            for owner, group_roots in sorted(groups.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", help="Write one JSON for the given dataset root")
    parser.add_argument("--output-dir", help="Write one JSON per transform owner")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-roots", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--reservoir-cap", type=int, default=200_000)
    args = parser.parse_args()
    if bool(args.output) == bool(args.output_dir):
        parser.error("specify exactly one of --output or --output-dir")
    if args.reservoir_cap < 1:
        parser.error("--reservoir-cap must be positive")
    if args.max_roots is not None and args.max_roots < 1:
        parser.error("--max-roots must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if args.output:
        jobs = [(args.source, dataset_dir, _iter_parquet_roots(dataset_dir, args.source), Path(args.output))]
    else:
        jobs = [
            (name, owner, roots, Path(args.output_dir) / f"{name}.json")
            for name, owner, roots in _stats_groups(dataset_dir, args.source)
        ]
    for name, root, roots, output in jobs:
        result = compute_stats(
            root,
            args.source,
            args.split,
            args.max_roots,
            args.max_rows,
            args.reservoir_cap,
            roots=roots,
        )
        result["stats_group"] = name
        result["stats_owner"] = str(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}: action_rows={result['num_action_rows']} state_rows={result['num_state_rows']}")


if __name__ == "__main__":
    main()
