"""Dependency-free TF-IDF retrieval and optional semantic retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
import re

from .corpus import Passage


STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "for", "how", "i", "in",
    "is", "my", "of", "on", "or", "should", "the", "to", "what", "when", "where",
    "who", "with", "must",
}


def tokenize(value: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", value.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float

    def to_mapping(self) -> dict:
        return {"score": round(self.score, 6), **self.passage.to_mapping()}


class TfidfRetriever:
    """Smoothed IDF and cosine similarity, matching the proposal's pilot."""

    def __init__(self, passages: list[Passage]) -> None:
        if not passages:
            raise ValueError("Retriever needs at least one passage")
        self.passages = passages
        counts = [Counter(tokenize(p.text)) for p in passages]
        doc_frequency = Counter(token for c in counts for token in c)
        n = len(passages)
        self.idf = {term: log((1 + n) / (1 + df)) + 1 for term, df in doc_frequency.items()}
        self.vectors = [self._weight(c) for c in counts]
        self.norms = [self._norm(v) for v in self.vectors]

    def _weight(self, counts: Counter) -> dict[str, float]:
        # Unseen query words use the same smoothed IDF with df=0.
        n = len(self.passages)
        return {t: count * self.idf.get(t, log(1 + n) + 1) for t, count in counts.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return sqrt(sum(x * x for x in vector.values()))

    def search(self, question: str, top_k: int = 3, min_score: float = 0.0) -> list[Hit]:
        if top_k <= 0 or min_score < 0:
            raise ValueError("top_k must be positive and min_score >= 0")
        q = self._weight(Counter(tokenize(question)))
        qnorm = self._norm(q)
        if qnorm == 0:
            return []
        hits: list[Hit] = []
        for passage, vec, dnorm in zip(self.passages, self.vectors, self.norms):
            numerator = sum(value * vec.get(term, 0.0) for term, value in q.items())
            score = numerator / (qnorm * dnorm) if dnorm else 0.0
            # Never return zero-similarity passages as evidence.
            if score > 0 and score >= min_score:
                hits.append(Hit(passage, score))
        return sorted(hits, key=lambda h: (-h.score, h.passage.passage_id))[:top_k]


class SemanticRetriever:
    """Optional Sentence-Transformers model; model files download on first use."""

    def __init__(self, passages: list[Passage], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Semantic retrieval requires pip install -e '.[semantic]'") from exc
        self.passages = passages
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([p.text for p in passages], normalize_embeddings=True)

    def search(self, question: str, top_k: int = 3, min_score: float = 0.0) -> list[Hit]:
        query_vec = self.model.encode([question], normalize_embeddings=True)[0]
        scored = [
            Hit(p, float(query_vec @ vec))
            for p, vec in zip(self.passages, self.embeddings)
        ]
        return sorted(
            (h for h in scored if h.score > 0 and h.score >= min_score),
            key=lambda h: (-h.score, h.passage.passage_id),
        )[:top_k]

