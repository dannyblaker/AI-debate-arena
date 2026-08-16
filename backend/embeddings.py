"""Semantic embeddings for hybrid retrieval, via a tiny GGUF model.

The embedding model (~35 MB, BGE-small) runs on llama.cpp like everything
else — no extra ML runtime. Everything here is best-effort: if the model
cannot be downloaded (offline machine) or fails to load, load_embedder()
returns None and retrieval falls back to keyword-only BM25.
"""
from __future__ import annotations

import threading
from typing import Callable

from .config import EMBED_MODEL_FILE, EMBED_MODEL_REPO
from .models_registry import StopRequested, download_file

# BGE retrieval instruction: prepended to queries, never to documents.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# BGE-small's maximum sequence length is 512 tokens; chunks are ~300 tokens.
# The char cut is a safety net for pathologically dense text.
_N_CTX = 512
_MAX_EMBED_CHARS = 1600
_BATCH = 16


class Embedder:
    def __init__(self, path):
        from llama_cpp import Llama
        self.llama = Llama(
            model_path=str(path),
            embedding=True,
            n_ctx=_N_CTX,
            n_batch=_N_CTX,
            n_ubatch=_N_CTX,
            n_gpu_layers=0,  # tiny model — keep VRAM for the debater
            verbose=False,
        )
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]):
        """L2-normalized float32 vectors, one row per input text."""
        import numpy as np
        rows = []
        with self._lock:
            for i in range(0, len(texts), _BATCH):
                batch = [t[:_MAX_EMBED_CHARS] for t in texts[i:i + _BATCH]]
                result = self.llama.create_embedding(batch)
                rows += [d["embedding"] for d in result["data"]]
        vecs = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_docs(self, texts: list[str]):
        return self._embed(texts)

    def embed_query(self, text: str):
        return self._embed([QUERY_PREFIX + text])[0]


def load_embedder(progress: Callable[[int, int, str], None],
                  stop: threading.Event) -> Embedder | None:
    """Download (if needed) and load the embedding model; None on failure."""
    try:
        path = download_file(EMBED_MODEL_REPO, EMBED_MODEL_FILE, 0,
                             progress, stop)
        return Embedder(path)
    except StopRequested:
        raise
    except Exception:
        return None
