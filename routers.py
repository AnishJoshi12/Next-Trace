"""Semantic intent router (design doc section 7).

The query is embedded once and compared with a bank of exemplar questions per intent. A confident match goes straight
to a deterministic tool, with no LLM tokens generated, which is how the < 2 s target is reachable on CPU-only hardware.

Scores are cosine similarities of L2-normalised vectors, so the threshold is a property of the *embedder*:
bge-small needs ~0.78, the lexical fallback ~0.5. Always re-calibrate with eval/run_eval.py on the target unit.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embeddings import Embedder
from utils import get_logger

log = get_logger("vgpt.router")


@dataclass(frozen=True)
class Route:
    intent: str
    score: float
    runner_up: str | None
    runner_up_score: float
    confident: bool       # score >= threshold
    plausible: bool       # score >= floor (usable in router-only mode when the LLM is down)


class IntentRouter:
    def __init__(self, embedder: Embedder, exemplars: dict[str, list[str]],
                 threshold: float | None = None, floor: float | None = None):
        self.embedder = embedder
        self.threshold = embedder.default_router_threshold if threshold is None or threshold < 0 else threshold
        self.floor = embedder.default_router_floor if floor is None or floor < 0 else floor
        self.labels: list[str] = []
        texts: list[str] = []
        for intent, items in exemplars.items():
            if intent.startswith("_"):
                continue
            for q in items:
                self.labels.append(intent)
                texts.append(q)
        self.texts = texts
        self.matrix = embedder.embed(texts)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        log.info("router ready: %d exemplars, %d intents, threshold %.2f, backend %s",
                 len(texts), len(set(self.labels)), self.threshold, embedder.name)

    @classmethod
    def from_file(cls, path: str | Path, embedder: Embedder, threshold: float | None = None,
                  floor: float | None = None) -> "IntentRouter":
        return cls(embedder, json.loads(Path(path).read_text(encoding="utf-8")), threshold, floor)

    def _embed(self, query: str) -> np.ndarray:
        key = " ".join(query.lower().split())
        v = self._cache.get(key)
        if v is None:
            v = self.embedder.embed([key])[0]
            self._cache[key] = v
            if len(self._cache) > 256:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return v

    def route(self, query: str) -> Route:
        sims = self.matrix @ self._embed(query)
        best_by_intent: dict[str, float] = {}
        for lab, s in zip(self.labels, sims):
            if s > best_by_intent.get(lab, -1.0):
                best_by_intent[lab] = float(s)
        ranked = sorted(best_by_intent.items(), key=lambda kv: -kv[1])
        (top, s1) = ranked[0]
        (ru, s2) = ranked[1] if len(ranked) > 1 else (None, 0.0)
        return Route(top, s1, ru, s2, s1 >= self.threshold, s1 >= self.floor)
