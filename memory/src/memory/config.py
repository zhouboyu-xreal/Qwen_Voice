"""Configuration helpers for the memory manager and runtime."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def split_memory_config(
    config: Dict[str, Any],
    *,
    include_voice_runtime: bool = False,
) -> Tuple[Dict[str, Any], ...]:
    """Split project config into runtime, manager, and optional voice mappings.

    The project config keeps settings in three sections:
    ``voice_runtime`` owns audio/VAD/ASR/speaker settings,
    ``memory_manager`` owns LLM, embedding, storage, and recall settings, while
    ``memory_runtime`` owns frontend batching settings. The returned manager
    mapping retains its nested ``llm`` and ``embedding`` mappings.

    Existing callers keep the two-value return by default. Set
    ``include_voice_runtime=True`` to append ``voice_runtime_config``.
    """
    manager_section = config.get("memory_manager")
    runtime_section = config.get("memory_runtime")
    memory_manager_config = (
        dict(manager_section) if isinstance(manager_section, dict) else {}
    )
    memory_runtime_config = (
        dict(runtime_section) if isinstance(runtime_section, dict) else {}
    )
    voice_section = config.get("voice_runtime")
    voice_runtime_config = (
        dict(voice_section) if isinstance(voice_section, dict) else {}
    )

    for nested_key in (
        "vad",
        "streaming_asr",
        "nonstreaming_asr",
        "kws",
        "speaker_identification",
    ):
        nested_value = voice_runtime_config.get(nested_key)
        if isinstance(nested_value, dict):
            voice_runtime_config[nested_key] = dict(nested_value)

    for nested_key in ("llm", "embedding"):
        nested_value = memory_manager_config.get(nested_key)
        memory_manager_config[nested_key] = (
            dict(nested_value) if isinstance(nested_value, dict) else {}
        )

    if include_voice_runtime:
        return memory_runtime_config, memory_manager_config, voice_runtime_config
    return memory_runtime_config, memory_manager_config
