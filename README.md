# 🎙️ AI Debate Arena

Two AIs debate any topic you choose. Three neutral AI judges score the debate
on a 100-point ballot and declare a winner. Everything — both debaters and all
three judges — runs on a **single local LLM** (a GGUF model from HuggingFace,
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
        3 AI judges (logician / policy analyst / rhetorician)
        100-point ballots: Content 30 · Rebuttal 25 · Style 25 · Organization 20
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
   text, and indexes it into overlapping chunks with BM25. Each debater's turn
   retrieves the passages most relevant to the motion and the opponent's last
   speech.
4. **Debate** — classic format, streamed live to the browser token by token:
   openings (PRO, CON) → alternating rebuttal rounds → closings (CON, then PRO
   gets the final word). The debaters are instructed to cite sources inline.
5. **Judging** — three judges with distinct neutral personas each score both
   sides per criterion (JSON output is grammar-constrained by llama.cpp).
6. **Verdict** — winner by majority of ballots, tie-broken on total points.
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
- CPU-only; no GPU required. An 8B model at Q5/Q4 wants ~8 GB free RAM;
  the 3B/1.5B models run in much less. Expect a few minutes per debate on CPU.
- Debates run fully locally — no LLM API keys needed. Network access is used
  only for the one-off model download and per-debate research (if research
  fails or you're offline, the debate proceeds on the model's own knowledge).

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
| `N_CTX` | `8192` | Context window |
| `N_THREADS` | auto | CPU threads for inference |
| `MAX_WEB_SOURCES` | `8` | Web pages fetched during research |
| `MAX_WIKI_SOURCES` | `2` | Wikipedia articles fetched |

## Project layout

```
backend/
  main.py            FastAPI app, WebSocket live stream, REST API
  debate.py          pipeline orchestration + debater prompting
  judging.py         judge personas, 100-point ballot, tally
  research.py        DuckDuckGo/Wikipedia scraping
  rag.py             chunking + BM25 retrieval
  models_registry.py model catalog, RAM-aware quant selection, downloads
  llm.py             llama.cpp wrapper (+ fake mode)
  pdf_export.py      PDF transcript
frontend/            static single-page UI (no build step)
```

## License

MIT — see [LICENSE](LICENSE).
