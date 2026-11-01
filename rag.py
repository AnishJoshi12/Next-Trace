"""Knowledge index + hybrid retrieval (design doc section 7).

    exact DTC lookup  ->  BM25 top-20 (SQLite FTS5)  +  dense top-20 (FAISS / NumPy)
                      ->  reciprocal-rank fusion (k=60)  ->  optional cross-encoder rerank (top 8 -> top 3)

The index holds only unstructured text (DTC dictionary, manuals, incident summaries). Numbers never come from here.
Graceful degradation: if the vectors are missing or were built by a different embedder, search falls back to FTS5 only.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embeddings import Embedder
from utils import get_logger, memory_headroom_mb

log = get_logger("vgpt.rag")

DTC_RE = re.compile(r"\b([PBCU][0-9A-F]{4})\b", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset("a an the is are was were my me i you of to in on at for with and or what when where which how why does do mean means".split())


@dataclass
class Chunk:
    chunk_id: str
    corpus: str
    title: str
    text: str
    score: float = 0.0


def find_dtc_codes(text: str) -> list[str]:
    return list(dict.fromkeys(m.upper() for m in DTC_RE.findall(text)))


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion over ranked lists of ids."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class VectorIndex:
    """Exact inner-product search over L2-normalised vectors (FAISS IndexFlatIP, or NumPy if FAISS is absent)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._faiss = None
        try:
            import faiss  # type: ignore

            idx = faiss.IndexFlatIP(self.vectors.shape[1])
            idx.add(self.vectors)
            self._faiss = idx
        except Exception:  # faiss-cpu wheel unavailable for this Python: NumPy is fine below ~50k vectors
            pass

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        q = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return ids[0], scores[0]
        sims = self.vectors @ q[0]
        ids = np.argpartition(-sims, k - 1)[:k]
        ids = ids[np.argsort(-sims[ids])]
        return ids, sims[ids]


class CrossEncoderReranker:
    """Optional ms-marco-MiniLM-L-6 reranker through fastembed (ONNX). Disabled by default."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore

        self._m = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, chunks: list[Chunk], keep: int) -> list[Chunk]:
        scores = list(self._m.rerank(query, [c.text for c in chunks]))
        for c, s in zip(chunks, scores):
            c.score = float(s)
        return sorted(chunks, key=lambda c: -c.score)[:keep]


class KnowledgeBase:
    def __init__(self, index_dir: str, embedder: Embedder, reranker: CrossEncoderReranker | None = None):
        self.dir = Path(index_dir)
        self.embedder = embedder
        self.reranker = reranker
        self.conn: sqlite3.Connection | None = None
        self.vindex: VectorIndex | None = None
        self.n_chunks = 0
        self.dense_ok = False
        db = self.dir / "kb.sqlite"
        if not db.exists():
            log.warning("knowledge index not found at %s (run scripts/build_index.py): knowledge answers disabled", self.dir)
            return
        self.conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vec_path, meta_path = self.dir / "vectors.npy", self.dir / "meta.json"
        if vec_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("backend") == embedder.name and meta.get("dim") == embedder.dim:
                self.vindex = VectorIndex(np.load(vec_path))
                self.dense_ok = True
            else:
                log.warning("index built with %s but runtime embedder is %s: dense search off (FTS5 only)",
                            meta.get("backend"), embedder.name)

    @property
    def available(self) -> bool:
        return self.conn is not None and self.n_chunks > 0

    # ---- exact DTC lookup -------------------------------------------------
    def dtc(self, code: str) -> Chunk | None:
        if not self.conn:
            return None
        r = self.conn.execute("SELECT chunk_id, corpus, title, text FROM chunks WHERE chunk_id=?",
                              (f"DTC:{code.upper()}",)).fetchone()
        return Chunk(r["chunk_id"], r["corpus"], r["title"], r["text"], 1.0) if r else None

    # ---- sparse + dense ---------------------------------------------------
    def bm25(self, query: str, k: int = 20) -> list[str]:
        if not self.conn:
            return []
        toks = [t for t in _TOKEN.findall(query) if t.lower() not in _STOP]
        if not toks:
            return []
        match = " OR ".join(f'"{t}"' for t in toks)
        try:
            rows = self.conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (match, k)).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("fts query failed: %s", exc)
            return []
        return [r["chunk_id"] for r in rows]

    def dense(self, query: str, k: int = 20) -> list[str]:
        if not (self.dense_ok and self.vindex and self.conn):
            return []
        ids, _ = self.vindex.search(self.embedder.embed_query([query])[0], k)
        if len(ids) == 0:
            return []
        rows = self.conn.execute(
            f"SELECT idx, chunk_id FROM chunks WHERE idx IN ({','.join('?' * len(ids))})", [int(i) for i in ids]).fetchall()
        by_idx = {r["idx"]: r["chunk_id"] for r in rows}
        return [by_idx[int(i)] for i in ids if int(i) in by_idx]

    def _get(self, chunk_id: str) -> Chunk | None:
        r = self.conn.execute("SELECT chunk_id, corpus, title, text FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return Chunk(r["chunk_id"], r["corpus"], r["title"], r["text"]) if r else None

    # ---- hybrid search ----------------------------------------------------
    def search(self, query: str, top: int = 3, rerank_pool: int = 8) -> list[Chunk]:
        if not self.available:
            return []
        out: list[Chunk] = []
        seen: set[str] = set()
        for code in find_dtc_codes(query):  # alphanumeric ids: exact match first, embeddings are poor at them
            c = self.dtc(code)
            if c and c.chunk_id not in seen:
                c.score = 1.0
                out.append(c)
                seen.add(c.chunk_id)
        fused = rrf([self.bm25(query, 20), self.dense(query, 20)], k=60)
        pool: list[Chunk] = []
        for cid, score in fused[:rerank_pool]:
            if cid in seen:
                continue
            c = self._get(cid)
            if c:
                c.score = score
                pool.append(c)
        headroom = memory_headroom_mb()
        if self.reranker and len(pool) > top and (headroom is None or headroom > 600):
            try:
                pool = self.reranker.rerank(query, pool, keep=top)
            except Exception as exc:  # reranking is an optimisation, never a failure point
                log.warning("rerank skipped: %s", exc)
        return (out + pool)[:max(top, len(out))]
