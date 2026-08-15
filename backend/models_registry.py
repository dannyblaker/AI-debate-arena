"""Model registry: curated GGUF models on HuggingFace, RAM-aware quantization
selection, and download management.

A single model file is chosen at debate time: we list the .gguf files in the
chosen repo (with sizes) via the HuggingFace API and pick the highest-quality
quantization that fits comfortably in the memory available to this process.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil
import requests

from .config import MODELS_DIR

MODELS = [
    {
        "id": "dolphin3-llama3.1-8b",
        "name": "Dolphin 3.0 (Llama 3.1 8B)",
        "repo": "bartowski/Dolphin3.0-Llama3.1-8B-GGUF",
        "params": "8B",
        "uncensored": True,
        "default": True,
        "description": "Community fine-tune with no built-in guardrails; fully steerable via system prompt. Best all-round choice.",
    },
    {
        "id": "mistral-7b-instruct-v0.3",
        "name": "Mistral 7B Instruct v0.3",
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "params": "7B",
        "uncensored": False,
        "default": False,
        "description": "Strong general model with relatively light alignment.",
    },
    {
        "id": "qwen2.5-3b-instruct",
        "name": "Qwen 2.5 3B Instruct",
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "params": "3B",
        "uncensored": False,
        "default": False,
        "description": "Small and fast; good for machines with limited RAM.",
    },
    {
        "id": "qwen2.5-1.5b-instruct",
        "name": "Qwen 2.5 1.5B Instruct",
        "repo": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "params": "1.5B",
        "uncensored": False,
        "default": False,
        "description": "Tiny fallback for very low-memory machines. Expect weaker debating.",
    },
]

# Highest quality first. Q6/Q8 are deliberately excluded: on CPU they are much
# slower than Q5/Q4 for a negligible quality gain in this use case.
QUANT_PREFERENCE = ["Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_0",
                    "Q3_K_L", "Q3_K_M", "IQ3_M", "Q3_K_S", "Q2_K"]

# Rough runtime overhead on top of the weights file: KV cache + compute
# buffers + everything else in the process.
RUNTIME_OVERHEAD_BYTES = int(1.8 * 1024**3)


class StopRequested(Exception):
    pass


@dataclass
class QuantChoice:
    filename: str
    size_bytes: int
    quant: str


def get_model(model_id: str) -> dict:
    for m in MODELS:
        if m["id"] == model_id:
            return m
    raise ValueError(f"Unknown model id: {model_id}")


def _repo_dir(repo: str) -> Path:
    return MODELS_DIR / repo.replace("/", "__")


def find_downloaded_file(model: dict) -> Path | None:
    d = _repo_dir(model["repo"])
    if not d.is_dir():
        return None
    ggufs = sorted(d.glob("*.gguf"))
    return ggufs[0] if ggufs else None


def available_ram_bytes() -> int:
    """Available RAM, respecting a container memory limit when one is set."""
    avail = psutil.virtual_memory().available
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw.isdigit():
            avail = min(avail, int(raw))
    except OSError:
        pass
    return avail


def list_models() -> list[dict]:
    out = []
    for m in MODELS:
        f = find_downloaded_file(m)
        out.append({
            **m,
            "downloaded": f is not None,
            "downloaded_file": f.name if f else None,
            "downloaded_gb": round(f.stat().st_size / 1024**3, 2) if f else None,
        })
    return out


def _repo_gguf_files(repo: str) -> dict[str, int]:
    """Map of single-file .gguf filenames -> size in bytes for a HF repo."""
    r = requests.get(
        f"https://huggingface.co/api/models/{repo}",
        params={"blobs": "true"},
        timeout=30,
    )
    r.raise_for_status()
    files: dict[str, int] = {}
    for sib in r.json().get("siblings", []):
        name = sib.get("rfilename", "")
        size = sib.get("size")
        # Skip non-gguf and multi-part files (e.g. "...-00001-of-00002.gguf").
        if not name.lower().endswith(".gguf") or re.search(r"-of-\d+\.gguf$", name):
            continue
        if size:
            files[name] = size
    return files


def pick_quant(model: dict, budget_bytes: int) -> QuantChoice:
    files = _repo_gguf_files(model["repo"])
    if not files:
        raise RuntimeError(f"No GGUF files found in {model['repo']}")
    considered = []
    for quant in QUANT_PREFERENCE:
        for name, size in files.items():
            if quant.lower() in name.lower():
                considered.append((quant, name, size))
                if size + RUNTIME_OVERHEAD_BYTES <= budget_bytes:
                    return QuantChoice(filename=name, size_bytes=size, quant=quant)
                break  # try the next (smaller) quant
    smallest = min((s for _, _, s in considered), default=min(files.values()))
    need = (smallest + RUNTIME_OVERHEAD_BYTES) / 1024**3
    raise RuntimeError(
        f"Not enough memory for {model['name']}: the smallest quantization needs "
        f"~{need:.1f} GB but only {budget_bytes / 1024**3:.1f} GB is available. "
        f"Try a smaller model."
    )


def ensure_model_file(
    model: dict,
    progress: Callable[[int, int, str], None],
    stop: threading.Event,
) -> tuple[Path, str]:
    """Return (path, description) for the model's GGUF file, downloading it
    if necessary. `progress(done_bytes, total_bytes, filename)` is called
    periodically during download."""
    existing = find_downloaded_file(model)
    if existing:
        return existing, f"already downloaded ({existing.name})"

    budget = available_ram_bytes()
    choice = pick_quant(model, budget)
    dest_dir = _repo_dir(model["repo"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / choice.filename
    part = dest.with_suffix(dest.suffix + ".part")

    url = f"https://huggingface.co/{model['repo']}/resolve/main/{choice.filename}"
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or choice.size_bytes
        done = 0
        last_report = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if stop.is_set():
                    f.close()
                    part.unlink(missing_ok=True)
                    raise StopRequested()
                f.write(chunk)
                done += len(chunk)
                if done - last_report >= 16 * 1024 * 1024 or done == total:
                    last_report = done
                    progress(done, total, choice.filename)
    part.rename(dest)
    return dest, f"downloaded {choice.quant} ({total / 1024**3:.1f} GB)"
