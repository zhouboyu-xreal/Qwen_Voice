#!/usr/bin/env python3
"""Compare single-text and batch embedding latency.

The benchmark intentionally uses the public EmbeddingClient methods. It also
makes the current implementation visible: if embed_batch internally loops over
embed_text, its measured latency will be close to the single request loop until
a real provider-side batch request is implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory.config import split_memory_config  # noqa: E402
from memory.embedding_client import EmbeddingClient  # noqa: E402


DEFAULT_TEXTS = [
    "用户计划下周讨论办公室搬迁和装修安排。用户计划下周讨论办公室搬迁和装修安排。用户计划下周讨论办公室搬迁和装修安排。",
    "团队正在评估手机产品的目标人群和差异化设计。团队正在评估手机产品的目标人群和差异化设计。团队正在评估手机产品的目标人群和差异化设计。团队正在评估手机产品的目标人群和差异化设计。",
    "之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。之前讨论过是否通过电商平台和直播方式推广手机。",
    "用",
    "需要。",
]


def parse_args() -> argparse.Namespace:
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
        default=1,
        help="Number of single and batch measurements to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required to load config.yaml") from exc
    loaded = yaml.safe_load(
        path.expanduser().resolve().read_text(encoding="utf-8")
    ) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")
    return loaded


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


def benchmark_single(
    client: EmbeddingClient,
    texts: Sequence[str],
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    vectors = [client.embed_text(text) for text in texts]
    elapsed_ms = _elapsed_ms(started_at)
    valid_count = sum(vector is not None for vector in vectors)
    return {
        "elapsed_ms": elapsed_ms,
        "text_count": len(texts),
        "valid_vector_count": valid_count,
        "none_vector_count": len(texts) - valid_count,
        "average_ms_per_text": round(elapsed_ms / max(1, len(texts)), 3),
    }


def benchmark_batch(
    client: EmbeddingClient,
    texts: Sequence[str],
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    vectors = client.embed_batch(texts)
    elapsed_ms = _elapsed_ms(started_at)
    valid_count = sum(vector is not None for vector in vectors)
    return {
        "elapsed_ms": elapsed_ms,
        "text_count": len(texts),
        "returned_vector_count": len(vectors),
        "valid_vector_count": valid_count,
        "none_vector_count": len(texts) - valid_count,
        "average_ms_per_text": round(elapsed_ms / max(1, len(texts)), 3),
    }


def average(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    _, memory_manager_config = split_memory_config(config)
    embedding_config = memory_manager_config.get("embedding") or {}
    if not isinstance(embedding_config, dict):
        raise ValueError("memory_manager.embedding must be a mapping")

    texts = load_texts(args)
    client = EmbeddingClient(dict(embedding_config))
    single_runs: List[Dict[str, Any]] = []
    batch_runs: List[Dict[str, Any]] = []
    for _ in range(args.runs):
        single_runs.append(benchmark_single(client, texts))
        batch_runs.append(benchmark_batch(client, texts))

    single_average_ms = average([item["elapsed_ms"] for item in single_runs])
    batch_average_ms = average([item["elapsed_ms"] for item in batch_runs])
    speedup: Optional[float] = None
    if single_average_ms is not None and batch_average_ms and batch_average_ms > 0:
        speedup = round(single_average_ms / batch_average_ms, 3)

    result = {
        "config_path": str(config_path),
        "provider": embedding_config.get("provider"),
        "model": embedding_config.get("model"),
        "dimensions": embedding_config.get("dimensions"),
        "text_count": len(texts),
        "runs": args.runs,
        "single": {
            "runs": single_runs,
            "average_total_elapsed_ms": single_average_ms,
        },
        "batch": {
            "runs": batch_runs,
            "average_total_elapsed_ms": batch_average_ms,
        },
        "batch_speedup_vs_single": speedup,
        "note": (
            "A speedup close to 1.0 means embed_batch is still executing "
            "one embed_text call per input, or the provider/network is the bottleneck."
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
