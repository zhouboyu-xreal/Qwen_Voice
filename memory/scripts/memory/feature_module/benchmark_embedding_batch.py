#!/usr/bin/env python3
"""Compare single-text and batch embedding latency.

Use the same texts and public EmbeddingClient methods as Runtime, without
starting PWM, extracting facts or modifying memory databases. Timings include
the client's HTTP and vector normalization work. Failed or inconsistent vectors
must not be reported as a successful latency improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import split_memory_config  # noqa: E402
from embedding_client import EmbeddingClient  # noqa: E402


DEFAULT_TEXTS = [
    "用户计划下周讨论办公室搬迁和装修安排。用户计划下周讨论办公室搬迁和装修安排。用户计划下周讨论办公室搬迁和装修安排。",
    "团队正在评估手机产品的目标人群和差异化设计。团队正在评估手机产品的目标人群和差异化设计。团队正在评估手机产品的目标人群和差异化设计。团队正在评估手机产品的目标人群和差异化设计。",
    "之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。",
    "用",
    "需要。",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EmbeddingClient single and batch embedding latency."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Text to embed; may be supplied multiple times.",
    )
    parser.add_argument(
        "--texts-file",
        type=Path,
        help="UTF-8 text file with one embedding input per non-empty line.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Repeat the input set this many times in one benchmark request.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of single and batch measurements to run.",
    )
    parser.add_argument("--batch-size", type=int,
                        help="Override embedding_model.batch_size (provider inputs per HTTP request).")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Unmeasured warmup calls per mode, each using one text; 0 disables warmup.")
    parser.add_argument("--min-cosine", type=float, default=0.999,
                        help="Minimum same-position single/batch cosine similarity (default: 0.999).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    args = parser.parse_args(argv)
    if args.count <= 0 or args.runs <= 0:
        parser.error("--count and --runs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not math.isfinite(args.min_cosine) or not 0 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be between 0 and 1")
    return args


def load_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.texts_file is not None:
        path = args.texts_file.expanduser().resolve()
        texts.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    texts.extend(
        str(text).strip()
        for text in args.texts or []
        if str(text).strip()
    )
    if not texts:
        texts = list(DEFAULT_TEXTS)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    return texts * args.count


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _valid_vector(vector: Any, dimension: int) -> bool:
    if vector is None:
        return False
    try:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        return bool(values.size == dimension and np.isfinite(values).all()
                    and np.linalg.norm(values) > 0)
    except (TypeError, ValueError):
        return False


def _measurement(
    client: EmbeddingClient, texts: Sequence[str], vectors: Sequence[Any], elapsed_ms: float,
) -> Dict[str, Any]:
    valid_count = sum(_valid_vector(vector, client.dimension) for vector in vectors)
    return {
        "elapsed_ms": elapsed_ms,
        "text_count": len(texts),
        "returned_vector_count": len(vectors),
        "valid_vector_count": valid_count,
        "missing_or_invalid_vector_count": max(0, len(texts) - valid_count),
        "succeeded": len(vectors) == len(texts) and valid_count == len(texts),
        "average_ms_per_text": round(elapsed_ms / max(1, len(texts)), 3),
        "texts_per_second": round(len(texts) * 1000 / elapsed_ms, 3) if elapsed_ms else None,
    }


def benchmark_single(client: EmbeddingClient, texts: Sequence[str]) -> tuple[Dict[str, Any], List[Any]]:
    started_at = time.perf_counter()
    vectors = [client.embed_text(text) for text in texts]
    elapsed_ms = _elapsed_ms(started_at)
    return _measurement(client, texts, vectors, elapsed_ms), vectors


def benchmark_batch(client: EmbeddingClient, texts: Sequence[str]) -> tuple[Dict[str, Any], List[Any]]:
    started_at = time.perf_counter()
    vectors = client.embed_batch(texts)
    elapsed_ms = _elapsed_ms(started_at)
    return _measurement(client, texts, vectors, elapsed_ms), vectors


def compare_vectors(single: Sequence[Any], batch: Sequence[Any], dimension: int,
                    min_cosine: float) -> Dict[str, Any]:
    similarities: List[float] = []
    differences: List[float] = []
    for left, right in zip(single, batch):
        if not _valid_vector(left, dimension) or not _valid_vector(right, dimension):
            continue
        a = np.asarray(left, dtype=np.float64).reshape(-1)
        b = np.asarray(right, dtype=np.float64).reshape(-1)
        similarity = float(np.sum((a / np.linalg.norm(a)) * (b / np.linalg.norm(b))))
        similarities.append(float(np.clip(similarity, -1, 1)))
        differences.append(float(np.max(np.abs(a - b))))
    return {
        "compared_vector_count": len(similarities),
        "minimum_cosine_similarity": min(similarities) if similarities else None,
        "maximum_absolute_difference": max(differences) if differences else None,
        "succeeded": bool(single) and len(single) == len(batch) == len(similarities)
                     and min(similarities) >= min_cosine,
    }


def average(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    embedding_config = dict(split_memory_config(config_path).embedding_model)
    if args.batch_size is not None:
        embedding_config["batch_size"] = args.batch_size
    batch_size = max(1, int(embedding_config.get("batch_size") or 32))

    texts = load_texts(args)
    client = EmbeddingClient(dict(embedding_config))
    if embedding_config.get("enabled") is False or client._provider != "openai" or not client._api_key:
        raise ValueError("Benchmark requires enabled OpenAI-compatible embeddings and a configured API key")
    for _ in range(args.warmup):
        for measure in (benchmark_single, benchmark_batch):
            measurement, _vectors = measure(client, texts[:1])
            if not measurement["succeeded"]:
                raise RuntimeError("Embedding warmup failed; no latency comparison is valid")
    single_runs: List[Dict[str, Any]] = []
    batch_runs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    for index in range(args.runs):
        order = ("single", "batch") if index % 2 == 0 else ("batch", "single")
        vectors_by_mode = {}
        for mode in order:
            measure = benchmark_single if mode == "single" else benchmark_batch
            measurement, vectors_by_mode[mode] = measure(client, texts)
            measurement.update(run=index + 1, execution_order=list(order))
            (single_runs if mode == "single" else batch_runs).append(measurement)
            print(f"run={index + 1} mode={mode} elapsed_ms={measurement['elapsed_ms']} "
                  f"valid={measurement['valid_vector_count']}/{len(texts)}", file=sys.stderr)
        comparisons.append(compare_vectors(vectors_by_mode["single"], vectors_by_mode["batch"],
                                           client.dimension, args.min_cosine))

    single_average_ms = average([item["elapsed_ms"] for item in single_runs])
    batch_average_ms = average([item["elapsed_ms"] for item in batch_runs])
    speedup: Optional[float] = None
    succeeded = all(item["succeeded"] for item in [*single_runs, *batch_runs, *comparisons])
    if succeeded and single_average_ms is not None and batch_average_ms and batch_average_ms > 0:
        speedup = round(single_average_ms / batch_average_ms, 3)

    result = {
        "status": "ok" if succeeded else "failed",
        "config_path": str(config_path),
        "provider": embedding_config.get("provider"),
        "model": embedding_config.get("model"),
        "dimensions": embedding_config.get("dimensions"),
        "text_count": len(texts),
        "runs": args.runs,
        "batch_size": batch_size,
        "warmup_calls_per_mode": args.warmup,
        "input_character_count": sum(len(text) for text in texts),
        "expected_http_requests_per_run": {
            "single": len(texts), "batch": math.ceil(len(texts) / batch_size),
        },
        "single": {
            "runs": single_runs,
            "average_total_elapsed_ms": single_average_ms,
            "median_total_elapsed_ms": statistics.median(item["elapsed_ms"] for item in single_runs),
        },
        "batch": {
            "runs": batch_runs,
            "average_total_elapsed_ms": batch_average_ms,
            "median_total_elapsed_ms": statistics.median(item["elapsed_ms"] for item in batch_runs),
        },
        "vector_comparisons": comparisons,
        "batch_speedup_vs_single": speedup,
        "note": (
            "Both modes use the same input order and EmbeddingClient HTTP behavior; "
            "there is no concurrency or persistent-session optimization. Warmup is excluded. "
            "Average per-text batch latency is amortized cost, not time to the first available vector. "
            "Speedup is omitted on any failure or vector mismatch. Provider/network variance "
            "can affect results; vector agreement alone does not prove fact-extraction quality."
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
