#!/usr/bin/env python3
"""Generate the HumanGen text-embedding cache for an empty prompt.

Run from ``icl_model/`` or from any directory. The default output is placed
next to this script, so it creates one cache file under ``scripts/``.
Weights and tokenizer files are resolved locally; no files are downloaded.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simpleicl.models.wan22.helpers.loader import _load_registered_model
from simpleicl.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer


CACHE_ENCODER_ID = "wan22ti2v5b"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "wan22-5b"


def _resolve_local_file(model_dir: Path, pattern: str) -> str:
    if pattern.endswith("/"):
        path = model_dir / pattern
        if not path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory does not exist: {path}")
        return str(path)
    matches = sorted(model_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No local file matches: {model_dir / pattern}")
    return str(matches[0])


def _atomic_torch_save(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        torch.save(payload, temporary)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.context_len < 1:
        parser.error("--context-len must be positive")
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local Wan model directory does not exist: {model_dir}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")

    import hashlib

    prompt = ""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    output_dir = args.output.expanduser().resolve()
    output_path = output_dir / f"{digest}.t5_len{args.context_len}.{CACHE_ENCODER_ID}.pt"
    if output_path.exists() and not args.overwrite:
        print(f"Cache already exists, skipped: {output_path}")
        return

    text_path = _resolve_local_file(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = _resolve_local_file(model_dir, "google/umt5-xxl/")
    encoder = _load_registered_model(
        text_path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path,
        seq_len=args.context_len,
        clean="whitespace",
    )

    with torch.inference_mode():
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        context = encoder(ids.to(device), mask.to(device=device, dtype=torch.bool))[0]

    payload = {
        "context": context.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "mask": mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous(),
    }
    _atomic_torch_save(payload, output_path)
    print(f"Wrote empty-prompt cache: {output_path}")
    print(f"context shape={tuple(payload['context'].shape)}, mask shape={tuple(payload['mask'].shape)}")


if __name__ == "__main__":
    main()
