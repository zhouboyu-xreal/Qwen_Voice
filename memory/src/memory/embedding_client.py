#!/usr/bin/env python3
"""Small embedding client used by the unified memory prototype.

The production voice_recording project can call remote embedding APIs. When the
provider is unavailable, this client returns ``None`` so callers can skip
semantic matching instead of treating a synthetic vector as real evidence.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "local_hash",
    "model": "local-hash-embedding",
    "base_url": "",
    "api_key": "",
    "api_key_env": "",
    "dimensions": 384,
    "normalize": True,
    "timeout": 30,
    "batch_size": 32,
}


def _load_project_config() -> Dict[str, Any]:
    try:
        import yaml

        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        manager_config = loaded.get("memory_manager")
        if isinstance(manager_config, dict) and isinstance(manager_config.get("embedding"), dict):
            return dict(manager_config["embedding"])
    except Exception:
        pass
    return {}


def _resolve_env(value: Any) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"\${([A-Za-z_][A-Za-z0-9_]*)}", text)
    if match:
        return os.environ.get(match.group(1), "").strip()
    return text


def _build_openai_embedding_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return "https://api.openai.com/v1/embeddings"
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1") or re.search(r"/api/paas/v\d+$", base):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


class EmbeddingClient:
    """OpenAI-compatible embedding client without synthetic-vector fallback."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(DEFAULT_CONFIG)
        merged.update(_load_project_config())
        if config:
            merged.update(config)
        self._config = merged
        self._provider = str(merged.get("provider") or "local_hash").strip().lower()
        self._model = str(merged.get("model") or "local-hash-embedding")
        self._base_url = str(merged.get("base_url") or "")
        self._api_key = _resolve_env(merged.get("api_key"))
        api_key_env = str(merged.get("api_key_env") or "").strip()
        if not self._api_key and api_key_env:
            self._api_key = os.environ.get(api_key_env, "").strip()
        self._dim = max(8, int(merged.get("dimensions") or 384))
        self._normalize = bool(merged.get("normalize", True))
        self._timeout = int(merged.get("timeout") or 30)
        self._batch_size = max(1, int(merged.get("batch_size") or 32))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._provider == "openai" and self._api_key:
            remote = self._embed_openai(text)
            if remote is not None:
                return remote
        return None

    def embed_batch(self, texts: Iterable[str]) -> List[Optional[np.ndarray]]:
        input_texts = list(texts)
        if not input_texts:
            return []
        if self._provider != "openai" or not self._api_key:
            return [None] * len(input_texts)

        vectors: List[Optional[np.ndarray]] = []
        for start in range(0, len(input_texts), self._batch_size):
            batch = input_texts[start : start + self._batch_size]
            vectors.extend(self._embed_openai_batch(batch))
        return vectors

    def _embed_openai(self, text: str) -> Optional[np.ndarray]:
        url = _build_openai_embedding_url(self._base_url)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self._model, "input": text or " "}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            vector = data.get("data", [{}])[0].get("embedding")
            if not vector:
                return None
            return self._as_vector(vector)
        except Exception as exc:
            logger.warning("Remote embedding failed; returning no embedding: %s", exc)
            return None

    def _embed_openai_batch(
        self,
        texts: Sequence[str],
    ) -> List[Optional[np.ndarray]]:
        """Request one provider-side embedding batch and preserve input order."""
        if not texts:
            return []
        url = _build_openai_embedding_url(self._base_url)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "input": [text or " " for text in texts],
        }
        empty_result: List[Optional[np.ndarray]] = [None] * len(texts)
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(raw_items, list):
                return empty_result

            result: List[Optional[np.ndarray]] = [None] * len(texts)
            used_indices: set[int] = set()
            for fallback_index, item in enumerate(raw_items):
                if not isinstance(item, dict):
                    continue
                raw_index = item.get("index", fallback_index)
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(texts) or index in used_indices:
                    continue
                embedding = item.get("embedding")
                if not embedding:
                    continue
                try:
                    result[index] = self._as_vector(embedding)
                except (TypeError, ValueError):
                    result[index] = None
                used_indices.add(index)
            return result
        except Exception as exc:
            logger.warning(
                "Remote batch embedding failed; returning no embeddings: %s",
                exc,
            )
            return empty_result

    def _as_vector(self, raw: Any) -> np.ndarray:
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vector.size != self._dim:
            resized = np.zeros((self._dim,), dtype=np.float32)
            keep = min(self._dim, vector.size)
            resized[:keep] = vector[:keep]
            vector = resized
        if self._normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        return vector.reshape(1, -1).astype(np.float32)
