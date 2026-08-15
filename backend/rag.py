"""Lightweight retrieval over the researched documents.

Documents are split into overlapping chunks and indexed with BM25. This keeps
the stack dependency-free (no embedding model to download) while giving the
debaters good keyword-relevant excerpts; the single LLM stays dedicated to
debating and judging.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .research import Doc

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Chunk:
    text: str
    source_title: str
    source_url: str


class ResearchIndex:
    def __init__(self, docs: list[Doc]):
        self.chunks: list[Chunk] = []
        for doc in docs:
            step = CHUNK_CHARS - CHUNK_OVERLAP
            for start in range(0, max(len(doc.text), 1), step):
                piece = doc.text[start:start + CHUNK_CHARS].strip()
                if len(piece) > 200:
                    self.chunks.append(Chunk(piece, doc.title, doc.url))
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks]) if self.chunks else None

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]

    def format_excerpts(self, query: str, k: int = 5, max_chars: int = 900) -> str:
        chunks = self.retrieve(query, k=k)
        if not chunks:
            return "(No research material was found for this topic; rely on your own expert knowledge.)"
        return "\n\n".join(f"[{c.source_title}]\n{c.text[:max_chars]}" for c in chunks)
