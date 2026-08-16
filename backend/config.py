import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT_DIR / "models"))

# Run the whole pipeline with a canned fake LLM (no model download, instant
# responses). Useful for developing the UI and testing the pipeline.
FAKE_LLM = os.environ.get("FAKE_LLM", "").lower() in ("1", "true", "yes")

# Context window for the debater/judge model. 16k comfortably fits the full
# transcript at maximum rounds plus generous research excerpts; raise it
# further if you have the memory (KV cache grows ~128 KB per token for an
# 8B model).
N_CTX = int(os.environ.get("N_CTX", "16384"))
N_THREADS = int(os.environ.get("N_THREADS", "0")) or None  # None -> llama.cpp default

# Model layers to offload to the GPU (-1 = all of them). Only has an effect
# with a CUDA build of llama-cpp-python (see Dockerfile.gpu); the default
# CPU build silently ignores it.
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))
MAX_WEB_SOURCES = int(os.environ.get("MAX_WEB_SOURCES", "8"))
MAX_WIKI_SOURCES = int(os.environ.get("MAX_WIKI_SOURCES", "2"))

# Retrieval: before each turn the debater LLM writes RAG_QUERIES search
# queries of its own; the top RAG_TOP_K passages across all queries are
# stitched with RAG_NEIGHBORS adjacent chunks of surrounding context, up to
# a total budget of RAG_MAX_CHARS characters per turn.
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
RAG_MAX_CHARS = int(os.environ.get("RAG_MAX_CHARS", "10000"))
RAG_NEIGHBORS = int(os.environ.get("RAG_NEIGHBORS", "1"))
RAG_QUERIES = int(os.environ.get("RAG_QUERIES", "3"))

# Hybrid retrieval: a small GGUF embedding model (llama.cpp) adds semantic
# search on top of BM25 keyword search. Set EMBEDDINGS=0 for keyword-only.
USE_EMBEDDINGS = os.environ.get("EMBEDDINGS", "1").lower() not in ("0", "false", "no")
EMBED_MODEL_REPO = os.environ.get(
    "EMBED_MODEL_REPO", "CompendiumLabs/bge-small-en-v1.5-gguf")
EMBED_MODEL_FILE = os.environ.get(
    "EMBED_MODEL_FILE", "bge-small-en-v1.5-q8_0.gguf")
