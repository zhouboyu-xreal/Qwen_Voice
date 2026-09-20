"""Shared low-level helpers for memory processing."""

from __future__ import annotations

from typing import Any, Optional
import math
import numpy as np

def _cohesion(embeddings: Sequence[np.ndarray]) -> float:
    if not embeddings:
        return 0.0
    center = _centroid(embeddings)
    return float(np.mean([_cal_embedding_cosine_similarity(item, center) for item in embeddings]))

def _centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot compute centroid for an empty segment")
    vectors = [_as_embedding_vector(item) for item in embeddings]
    if any(vector is None for vector in vectors):
        raise ValueError("Embedding provider returned an invalid vector")
    return np.mean(np.vstack(vectors), axis=0)

def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)

def _as_embedding_vector(value: Any) -> Optional[np.ndarray]:
    """Normalize an embedding-like value into a flat float32 vector."""
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim > 1:
        vector = vector.reshape(-1)
    if vector.size == 0:
        return None
    return vector


def _cal_embedding_cosine_similarity(left: Any, right: Any) -> float:
    """Return cosine similarity, or zero when either embedding is invalid."""
    left_vector = _as_embedding_vector(left)
    right_vector = _as_embedding_vector(right)
    if left_vector is None or right_vector is None:
        return 0.0
    keep = min(left_vector.size, right_vector.size)
    if keep <= 0:
        return 0.0
    left_vector = left_vector[:keep].reshape(-1)
    right_vector = right_vector[:keep].reshape(-1)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)
