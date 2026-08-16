"""Hybrid retrieval over the researched documents.

Documents are split into overlapping chunks. Every chunk is indexed two ways:
BM25 for keyword relevance, and (when the embedding model is available) dense
vectors for semantic relevance; the two rankings are fused with reciprocal
rank fusion. Without an embedder the index degrades to BM25-only.

gather_research() is the debaters' entry point: it fuses results across
several search queries, stitches each hit with its neighbouring chunks from
the original document (retrieved fragments of scripts and prose need their
surroundings to make sense), merges overlapping regions, and packs the best
material into a fixed character budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import RAG_MAX_CHARS, RAG_NEIGHBORS, RAG_TOP_K
from .research import Doc

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

# Reciprocal rank fusion: score = sum over rankings of 1/(K + rank).
RRF_K = 60
# Rank positions past this contribute noise, not signal.
RRF_DEPTH = 50

NO_RESEARCH = ("(No research material was found for this topic; "
               "rely on your own expert knowledge.)")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Chunk:
    text: str
    source_title: str
    source_url: str
    doc_idx: int
    start: int
    end: int


class ResearchIndex:
    def __init__(self, docs: list[Doc], embedder=None):
        self.docs = docs
        self.chunks: list[Chunk] = []
        step = CHUNK_CHARS - CHUNK_OVERLAP
        for di, doc in enumerate(docs):
            for start in range(0, max(len(doc.text), 1), step):
                end = min(start + CHUNK_CHARS, len(doc.text))
                piece = doc.text[start:end].strip()
                if len(piece) > 200:
                    self.chunks.append(Chunk(piece, doc.title, doc.url,
                                             di, start, end))
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

    def _hybrid_ranking(self, query: str) -> list[int]:
        """Chunk indices for one query, best first (BM25 + vectors, RRF)."""
        # Zero-score BM25 chunks share no words with the query, so their
        # rank order is meaningless — drop them from that ranking.
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
        return sorted(fused, key=lambda i: fused[i], reverse=True)

    def retrieve(self, query: str, k: int = RAG_TOP_K) -> list[Chunk]:
        if not self._bm25:
            return []
        return [self.chunks[i] for i in self._hybrid_ranking(query)[:k]]

    def gather_research(self, queries: list[str],
                        total_chars: int = RAG_MAX_CHARS) -> str:
        """Best material across all queries, with neighbouring context
        stitched in, packed into `total_chars`."""
        if not self._bm25:
            return NO_RESEARCH
        fused: dict[int, float] = {}
        for query in queries:
            for rank, i in enumerate(self._hybrid_ranking(query)[:RRF_DEPTH]):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(fused, key=lambda i: fused[i], reverse=True)[:RAG_TOP_K]

        pad = RAG_NEIGHBORS * (CHUNK_CHARS - CHUNK_OVERLAP)
        selected: list[tuple[int, int, int]] = []  # (doc_idx, start, end)
        used = 0
        parts: list[str] = []
        for i in ordered:
            if used >= total_chars:
                break
            c = self.chunks[i]
            doc = self.docs[c.doc_idx]
            s = max(0, c.start - pad)
            e = min(len(doc.text), c.end + pad)
            # Clip against regions of this doc that are already included.
            for dj, s2, e2 in selected:
                if dj == c.doc_idx and s < e2 and e > s2:
                    if s >= s2 and e <= e2:
                        s = e  # fully covered
                    elif s >= s2:
                        s = e2
                    else:
                        e = s2
            if e - s < 200:
                continue
            e = min(e, s + (total_chars - used))
            excerpt = doc.text[s:e].strip()
            if len(excerpt) < 200:
                continue
            selected.append((c.doc_idx, s, e))
            used += e - s
            parts.append(f"[{c.source_title}]\n{excerpt}")
        return "\n\n".join(parts) if parts else NO_RESEARCH
