import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT_DIR / "models"))

# Run the whole pipeline with a canned fake LLM (no model download, instant
# responses). Useful for developing the UI and testing the pipeline.
FAKE_LLM = os.environ.get("FAKE_LLM", "").lower() in ("1", "true", "yes")

N_CTX = int(os.environ.get("N_CTX", "8192"))
N_THREADS = int(os.environ.get("N_THREADS", "0")) or None  # None -> llama.cpp default
MAX_WEB_SOURCES = int(os.environ.get("MAX_WEB_SOURCES", "8"))
MAX_WIKI_SOURCES = int(os.environ.get("MAX_WIKI_SOURCES", "2"))
