"""Hybrid retrieval over the researched documents.

Documents are split into overlapping chunks. Every chunk is indexed two ways:
BM25 for keyword relevance, and (when the embedding model is available) dense
vectors for semantic relevance. The two rankings are fused with reciprocal
rank fusion, so a chunk that says the same thing as the query in different
words is still found. Without an embedder the index degrades to BM25-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import RAG_EXCERPT_CHARS, RAG_TOP_K
from .research import Doc

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

# Reciprocal rank fusion: score = sum over rankings of 1/(K + rank).
RRF_K = 60
# Rank positions past this contribute noise, not signal.
RRF_DEPTH = 50


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Chunk:
    text: str
    source_title: str
    source_url: str


class ResearchIndex:
    def __init__(self, docs: list[Doc], embedder=None):
        self.chunks: list[Chunk] = []
        for doc in docs:
            step = CHUNK_CHARS - CHUNK_OVERLAP
            for start in range(0, max(len(doc.text), 1), step):
                piece = doc.text[start:start + CHUNK_CHARS].strip()
                if len(piece) > 200:
                    self.chunks.append(Chunk(piece, doc.title, doc.url))
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks]) if self.chunks else None
        self._embedder = None
        self._vectors = None
        if embedder is not None and self.chunks:
            try:
                self._vectors = embedder.embed_docs([c.text for c in self.chunks])
                self._embedder = embedder
            except Exception:
                self._vectors = None  # keyword-only fallback

    @property
    def semantic(self) -> bool:
        return self._vectors is not None

    @staticmethod
    def _ranked(values, positive_only=False) -> list[int]:
        idx = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        if positive_only:
            idx = [i for i in idx if values[i] > 0]
        return idx

    def retrieve(self, query: str, k: int = RAG_TOP_K) -> list[Chunk]:
        if not self._bm25:
            return []
        # BM25 ranking; zero-score chunks share no words with the query, so
        # their rank order is meaningless — drop them from this ranking.
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        rankings = [self._ranked(bm25_scores, positive_only=True)]
        if self._vectors is not None:
            try:
                sims = self._vectors @ self._embedder.embed_query(query)
                rankings.append(self._ranked(sims.tolist()))
            except Exception:
                pass  # embed failure mid-debate: keyword ranking still stands
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking[:RRF_DEPTH]):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(fused, key=lambda i: fused[i], reverse=True)
        return [self.chunks[i] for i in top[:k]]

    def format_excerpts(self, query: str, k: int = RAG_TOP_K,
                        max_chars: int = RAG_EXCERPT_CHARS) -> str:
        chunks = self.retrieve(query, k=k)
        if not chunks:
            return "(No research material was found for this topic; rely on your own expert knowledge.)"
        cut = max_chars if max_chars > 0 else CHUNK_CHARS
        return "\n\n".join(f"[{c.source_title}]\n{c.text[:cut]}" for c in chunks)
