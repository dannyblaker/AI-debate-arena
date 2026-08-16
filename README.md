# 🎙️ AI Debate Arena

Two AIs debate any topic you choose. A neutral AI judge scores the debate
on a 100-point ballot and declares a winner. Everything — both debaters and
the judge — runs on a **single local LLM** (a GGUF model from HuggingFace,
via `llama.cpp`), with live web research feeding the debaters through a RAG
pipeline. The whole thing runs with one command:

```bash
docker compose up
```

then open **http://localhost:8000**.

## How it works

```
topic ──► research (DuckDuckGo + Wikipedia, trafilatura extraction)
              │
              ▼
        BM25 chunk index  ──────────────┐
                                        ▼
   PRO debater ◄── shared local LLM ──► CON debater
        │      (llama.cpp, one GGUF)      │
        └────────── transcript ───────────┘
                        │
                        ▼
        1 AI judge (a neutral debate adjudicator)
        100-point ballot, 8 criteria: evidence, logic, refutation,
        defense, persuasion, rhetoric, structure, clarity
                        │
                        ▼
              verdict + PDF export
```

1. **Setup** — enter a motion, pick a model, optionally give each debater a
   personality ("proper and Oxford-like", "sassy and sarcastic", …) and choose
   the number of rebuttal rounds. Motions work best phrased as claims
   ("X is true"), the way real debate motions are; questions are also
   handled (PRO argues yes, CON argues no).
2. **Model** — on *Begin Debate* the app checks available RAM (respecting
   container memory limits) and downloads the highest-quality quantization of
   the chosen model that fits in memory. Already-downloaded models are flagged
   in the UI and reused.
3. **Research** — the app searches the web and Wikipedia, extracts article
   text, and splits it into overlapping chunks indexed two ways: BM25
   keywords and dense vectors from a tiny (~35 MB) GGUF embedding model
   (BGE-small, also run by llama.cpp — auto-downloaded, gracefully skipped
   offline). Each debater's turn retrieves the passages most relevant to the
   motion and the opponent's last speech, fusing both rankings with
   reciprocal rank fusion. You can also upload your own research materials (PDF, Word,
   text/Markdown or HTML) in the setup screen — they are indexed alongside
   the web research and cited by filename, or used exclusively if you tick
   *Use only my materials*.
4. **Debate** — classic format, streamed live to the browser token by token:
   openings (PRO, CON) → alternating rebuttal rounds → closings (CON, then PRO
   gets the final word). The debaters are instructed to cite sources inline.
5. **Judging** — a neutral judge scores each of 8 fine-grained criteria in
   its own focused pass, writing separate reasoning for each side before
   committing to a score (JSON output is grammar-constrained by llama.cpp),
   then writes a closing summary of what each side did well and less well.
6. **Verdict** — winner by total points on the judge's ballot.
   Export the full transcript, ballots and verdict as a PDF.

## Models

| Model | Size | Notes |
|---|---|---|
| **Dolphin 3.0 (Llama 3.1 8B)** — default | 8B | Community fine-tune with no built-in guardrails; debates any topic without refusals |
| Mistral 7B Instruct v0.3 | 7B | Light alignment |
| Qwen 2.5 3B Instruct | 3B | For low-RAM machines |
| Qwen 2.5 1.5B Instruct | 1.5B | Tiny fallback |

Quantization is chosen automatically (preference order Q5_K_M → … → Q2_K)
based on the file sizes reported by the HuggingFace API and the RAM available
to the process. Weights are cached in `./models/` (mounted as a Docker
volume), so they download once.

> ⚠️ The default model is an "uncensored" community fine-tune: it will argue
> either side of contentious motions without refusing. That is the point of a
> debate app — but the output is unmoderated, so use judgement when sharing it.

## Running

```bash
docker compose up --build      # first run builds the image (~a few minutes)
```

- UI: http://localhost:8000
- Runs on CPU by default; no GPU required. An 8B model at Q5/Q4 wants ~8 GB
  free RAM; the 3B/1.5B models run in much less. Expect a few minutes per
  debate on CPU — or use the GPU image below.
- Debates run fully locally — no LLM API keys needed. Network access is used
  only for the one-off model download and per-debate research (if research
  fails or you're offline, the debate proceeds on the model's own knowledge).

### GPU acceleration (NVIDIA)

With an NVIDIA GPU and the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on the host:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The first build is slow (~15+ minutes: it compiles llama.cpp's CUDA kernels
from source), but generation is then typically 10-50x faster than CPU. An 8B
model at Q4 fits in ~6 GB of VRAM. If the model doesn't fit in your VRAM, set
`N_GPU_LAYERS` in `docker-compose.gpu.yml` to a positive number to offload
only part of it and split the rest onto the CPU.

### Development without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
uvicorn backend.main:app --reload
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODELS_DIR` | `./models` | Where GGUF weights are stored |
| `FAKE_LLM` | off | `1` = canned responses, no model download — full-pipeline demo/dev mode |
| `N_CTX` | `16384` | Context window — raise it if you have the memory (RAM/VRAM budgeting adapts automatically) |
| `N_THREADS` | auto | CPU threads for inference |
| `N_GPU_LAYERS` | `0` | Model layers offloaded to GPU (`-1` = all; needs the CUDA build, see `Dockerfile.gpu`) |
| `MAX_WEB_SOURCES` | `8` | Web pages fetched during research |
| `MAX_WIKI_SOURCES` | `2` | Wikipedia articles fetched |
| `RAG_TOP_K` | `6` | Research excerpts given to each debater turn |
| `RAG_EXCERPT_CHARS` | `0` | Characters per excerpt (`0` = the full chunk) |
| `EMBEDDINGS` | on | `0` = disable semantic search (keyword-only BM25 retrieval) |
| `EMBED_MODEL_REPO` | `CompendiumLabs/bge-small-en-v1.5-gguf` | HF repo of the GGUF embedding model |
| `EMBED_MODEL_FILE` | `bge-small-en-v1.5-q8_0.gguf` | Embedding model file within the repo |

## Project layout

```
backend/
  main.py            FastAPI app, WebSocket live stream, REST API
  debate.py          pipeline orchestration + debater prompting
  judging.py         judge persona, 100-point ballot, tally
  research.py        DuckDuckGo/Wikipedia scraping
  materials.py       user-uploaded research documents (PDF/DOCX/text/HTML)
  rag.py             chunking + hybrid BM25/semantic retrieval (RRF)
  embeddings.py      GGUF embedding model wrapper (llama.cpp)
  models_registry.py model catalog, RAM-aware quant selection, downloads
  llm.py             llama.cpp wrapper (+ fake mode)
  pdf_export.py      PDF transcript
frontend/            static single-page UI (no build step)
```

## License

MIT — see [LICENSE](LICENSE).
