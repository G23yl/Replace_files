"""Resolve robot video records to human samples from ICL_config JSON."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _norm_path(value: str) -> str:
    text = str(value or "").replace("\\", "/").strip("/")
    text = re.sub(r"^icl_raw_data/", "", text)
    return text


def _human_video_path(raw: str, human_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute() and path.is_file():
        return path
    normalized = _norm_path(raw)
    marker = "human_data/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = human_root.parent / relative
        if candidate.is_file():
            return candidate
    candidate = human_root / Path(normalized).name
    if candidate.is_file():
        return candidate
    # Common manifest path form: human_data/<source>/<run>/samples/<sample>/file.
    parts = Path(normalized).parts
    for i, part in enumerate(parts):
        if part == human_root.name:
            candidate = human_root.parent.joinpath(*parts[i:])
            if candidate.is_file():
                return candidate
    candidate = human_root / Path(*parts[-4:]) if len(parts) >= 4 else human_root / Path(*parts)
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Unable to resolve human_video_path={raw!r}; tried under {human_root}")


class ICLManifest:
    """Index pair records by normalized robot video path."""

    def __init__(self, path: str | Path, *, human_root: str | Path):
        self.path = Path(path).expanduser()
        self.human_root = Path(human_root).expanduser()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        samples = payload.get("samples", payload.get("pairs", [])) if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise ValueError(f"ICL config {self.path} must contain a samples list")
        self.by_robot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in samples:
            if not isinstance(item, dict):
                continue
            status = item.get("status_info") if isinstance(item.get("status_info"), dict) else {}
            robot_paths = {
                _norm_path(item.get(key, ""))
                for key in ("video_rel_path", "original_video_rel_path", "source_video_rel_path")
            }
            robot_paths.update(
                _norm_path(status.get(key, ""))
                for key in ("video_rel_path", "original_video_rel_path")
            )
            robot_paths.discard("")
            human_raw = item.get("human_video_path") or status.get("human_video_path")
            if not human_raw:
                continue
            for robot_key in robot_paths:
                self.by_robot[robot_key].append(dict(item, _human_video_raw=str(human_raw)))

    def lookup(self, robot_video_rel_path: str) -> dict[str, Any] | None:
        key = _norm_path(robot_video_rel_path)
        matches = self.by_robot.get(key, [])
        if not matches:
            # Source paths can include a dataset prefix absent from info.json paths.
            suffix = [items for path, items in self.by_robot.items() if path.endswith(key) or key.endswith(path)]
            matches = [item for group in suffix for item in group]
        unique = {
            str(item.get("sample_id", item.get("sample", item.get("_human_video_raw")))): item
            for item in matches
        }
        matches = list(unique.values())
        if not matches:
            return None
        # ICL configs may intentionally map one robot clip to several generations;
        # preserve JSON order and consistently use the first candidate.
        item = matches[0]
        try:
            video = _human_video_path(item["_human_video_raw"], self.human_root)
        except FileNotFoundError:
            # Keep the candidate mapping so dataset policy can report or tolerate missing modalities.
            raw = item["_human_video_raw"]
            normalized = _norm_path(raw)
            if "human_data/" in normalized:
                candidate = self.human_root.parent / normalized.split("human_data/", 1)[1]
            else:
                candidate = self.human_root / Path(normalized).name
            video = candidate
        return {
            "pair": item,
            "video_path": video,
            "sample_id": str(item.get("sample_id", item.get("sample", ""))),
            "run": str(item.get("run", "")),
            "sample": str(item.get("sample", "")),
        }
