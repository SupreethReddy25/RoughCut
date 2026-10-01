"""Semantic features (PLAN 6.3 U4, 12): MiniLM phrase embeddings -> redundancy + centrality; TF-IDF keywords.

The linear model only ever sees the derived scalars; raw embeddings go to embeddings.npy (for U8).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "if", "so", "of", "to", "in", "on", "at", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "be", "been", "am", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "there", "here", "what", "which", "who", "whom", "do", "does", "did", "done", "have", "has", "had",
    "having", "not", "no", "nor", "than", "then", "too", "very", "can", "could", "would", "should",
    "will", "just", "about", "into", "out", "up", "down", "over", "also", "more", "most", "some", "any",
    "each", "other", "such", "only", "own", "same", "s", "t", "don't", "i'm", "it's", "that's",
])


def embed_texts(texts: list[str]) -> np.ndarray:
    """L2-normalised MiniLM embeddings, shape (n, 384), float32."""
    global _model
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    e = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(e, dtype=np.float32)


def redundancy_centrality(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """redundancy[i] = max cosine to any *other* phrase; centrality[i] = cosine to the video mean embedding."""
    n = len(emb)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    red = sims.max(axis=1) if n > 1 else np.zeros(n)
    mean = emb.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    cen = emb @ mean
    return np.clip(red, 0.0, 1.0), cen


def _tok(text: str) -> str:
    return re.sub(r"[^a-z0-9']", "", text.lower().replace("’", "'"))


def tfidf_scores(docs: list[list[str]]) -> dict[str, float]:
    """TF-IDF keyword score per word type, with phrases as documents; max-normalised to 0..1.

    score(w) = (1 + log tf_total(w)) * idf(w), idf = log((1+N)/(1+df)) + 1. Stopwords score 0.
    """
    docs_n = [[_tok(w) for w in d if _tok(w)] for d in docs]
    n = len(docs_n)
    df: Counter = Counter()
    tf: Counter = Counter()
    for d in docs_n:
        tf.update(d)
        df.update(set(d))
    scores = {}
    for w, c in tf.items():
        if w in STOPWORDS:
            scores[w] = 0.0
            continue
        scores[w] = (1 + math.log(c)) * (math.log((1 + n) / (1 + df[w])) + 1)
    top = max(scores.values(), default=0.0)
    return {w: (s / top if top > 0 else 0.0) for w, s in scores.items()}


def word_tfidf(word_text: str, scores: dict[str, float]) -> float:
    return scores.get(_tok(word_text), 0.0)
