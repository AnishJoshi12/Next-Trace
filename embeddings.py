"""Embedding backends.

* ``FastEmbedder``    bge-small-en-v1.5 (384-d) through ONNX Runtime (the `fastembed` package). The model files must be
                      staged on the device beforehand (no network at runtime), see README "Stage the models".
* ``HashingEmbedder`` dependency-free fallback (hashed word / bigram / character-trigram TF vectors). It is lexical,
                      not semantic, so it is only meant for CI, laptops without the model, and graceful degradation.

Index and query MUST use the same backend; build_index.py records the backend name in meta.json and rag.py checks it.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Protocol

import numpy as np

from config import Settings
from utils import get_logger

log = get_logger("vgpt.embed")

_STOP = frozenset("""a an the is are was were be been am do does did my me i you your it its of to in on at for with by from as
and or but if then than that this these those there here what when where which who whom how why can could would should will
shall may might please tell show give about""".split())


class Embedder(Protocol):
    name: str
    dim: int
    default_router_threshold: float
    default_router_floor: float

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, texts: list[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashingEmbedder:
    name = "hashing-v1"
    # Calibrated on knowledge/intents.json + tests/test_router.py. Lexical similarity is far more spread out
    # than bge cosine, so the numbers differ from the 0.78 used for bge in the design document.
    default_router_threshold = 0.50
    default_router_floor = 0.28

    def __init__(self, dim: int = 1024):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = self._tokens(text)
        feats: dict[str, float] = {}
        for t in toks:
            feats[f"w:{t}"] = feats.get(f"w:{t}", 0.0) + 1.0
            if len(t) >= 5:
                for i in range(len(t) - 2):  # char trigrams: typo / plural tolerance
                    k = f"c:{t[i:i + 3]}"
                    feats[k] = feats.get(k, 0.0) + 0.25
        for a, b in zip(toks, toks[1:]):
            feats[f"b:{a}_{b}"] = feats.get(f"b:{a}_{b}", 0.0) + 0.8
        for k, c in feats.items():
            v[zlib.crc32(k.encode()) % self.dim] += 1.0 + math.log(c) if c >= 1 else c
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._vec(t) for t in texts]))

    embed_query = embed


class FastEmbedder:
    name = "fastembed-bge-small-en-v1.5"
    dim = 384
    default_router_threshold = 0.78   # design document value; recalibrate with eval/ on the target unit
    default_router_floor = 0.55

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        try:
            self._m = TextEmbedding(model_name=model_name, cache_dir=cache_dir, local_files_only=True)
        except TypeError:  # older fastembed without local_files_only
            self._m = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.name = f"fastembed-{model_name.split('/')[-1]}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(np.asarray(list(self._m.embed(texts)), dtype=np.float32)) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        fn = getattr(self._m, "query_embed", None)  # adds the bge retrieval prefix when the model needs it
        vecs = list(fn(texts)) if fn else list(self._m.embed(texts))
        return _l2(np.asarray(vecs, dtype=np.float32))


def get_embedder(settings: Settings, backend: str | None = None) -> Embedder:
    backend = backend or settings.embed_backend
    if backend == "hashing":
        return HashingEmbedder()
    if backend in ("fastembed", "auto"):
        try:
            return FastEmbedder(settings.embed_model, settings.model_cache)
        except Exception as exc:  # ImportError, missing model files, ...
            if backend == "fastembed":
                raise
            log.warning("fastembed unavailable (%s): falling back to the lexical hashing embedder. "
                        "Routing accuracy will be lower; see README 'Stage the models'.", exc)
            return HashingEmbedder()
    raise ValueError(f"unknown embed backend: {backend}")
