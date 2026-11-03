"""Project configuration loading and Personal World Model config splitting."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional


_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PWM_MODULES = (
    "llm_model",
    "embedding_model",
    "memory_runtime",
    "task_scheduling",
    "memory_manager",
    "user_state",
)


class PersonalWorldModelConfig(NamedTuple):
    llm_model: Dict[str, Any]
    embedding_model: Dict[str, Any]
    memory_runtime: Dict[str, Any]
    task_scheduling: Dict[str, Any]
    memory_manager: Dict[str, Any]
    user_state: Dict[str, Any]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.yaml"


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.environ.get(match.group(1), ""), value)
    return value


def _apply_pwm_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Desktop config.env overrides; YAML remains the default for every sidecar."""
    pwm = config.get("personal_world_model")
    if not isinstance(pwm, dict):
        return config
    fields = {
        "LLM_BASE_URL": ("llm_model", "llm_base_url", str),
        "LLM_MODEL": ("llm_model", "llm_name", str),
        "LLM_API_KEY": ("llm_model", "llm_api_key", str),
        "LLM_TIMEOUT": ("llm_model", "llm_timeout", int),
        "LLM_MAX_TOKENS": ("llm_model", "llm_max_tokens", int),
        "LLM_THINKING": ("llm_model", "llm_thinking", str),
        "LLM_JSON_MODE": ("llm_model", "llm_json_mode", bool),
        "EMBEDDING_BASE_URL": ("embedding_model", "base_url", str),
        "EMBEDDING_MODEL": ("embedding_model", "model", str),
        "EMBEDDING_API_KEY": ("embedding_model", "api_key", str),
        "EMBEDDING_TIMEOUT": ("embedding_model", "timeout", int),
        "EMBEDDING_DIMENSIONS": ("embedding_model", "dimensions", int),
        "EMBEDDING_BATCH_SIZE": ("embedding_model", "batch_size", int),
        "EMBEDDING_NORMALIZE": ("embedding_model", "normalize", bool),
    }
    for suffix, (section, field, convert) in fields.items():
        value = os.environ.get(f"QWEN_AUDIO_PWM_{suffix}")
        if value is None or not value.strip():
            continue
        if not isinstance(pwm.get(section), dict):
            continue  # split_memory_config owns missing-section validation.
        if convert is bool:
            text = value.strip().lower()
            if text not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                raise ValueError(f"Invalid PWM boolean: {suffix}")
            value = text in {"1", "true", "yes", "on"}
        else:
            value = convert(value.strip())
        pwm[section][field] = value
        if section == "embedding_model":
            # 'openai' selects the API protocol, independently of the vendor.
            pwm[section]["provider"] = "openai"
    return config


def load_project_config(source: Optional[Path | str | Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load the project YAML (or normalize an already loaded YAML mapping)."""
    if isinstance(source, Mapping):
        loaded = deepcopy(dict(source))
    else:
        path = Path(source).expanduser().resolve() if source is not None else _default_config_path()
        if not path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is project-owned
            raise RuntimeError("PyYAML is required to load config.yaml") from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
    config = _expand_env_refs(loaded)
    return _apply_pwm_environment(config)


def split_memory_config(
    source: Optional[Path | str | Mapping[str, Any]] = None,
) -> PersonalWorldModelConfig:
    """Return the independent configuration sections under ``personal_world_model``.

    ``source`` may be a config.yaml path or an already loaded project mapping.
    With no argument, the repository-level config.yaml is loaded. Every returned
    module value is a deep copy so callers can apply local overrides safely.
    """
    config = load_project_config(source)
    pwm_config = config.get("personal_world_model")
    if not isinstance(pwm_config, dict):
        raise ValueError("config.yaml must define personal_world_model as a mapping")

    modules: Dict[str, Dict[str, Any]] = {}
    for module_name in _PWM_MODULES:
        value = pwm_config.get(module_name)
        if not isinstance(value, dict):
            raise ValueError(
                f"personal_world_model.{module_name} must be a mapping"
            )
        modules[module_name] = deepcopy(value)
    return PersonalWorldModelConfig(**modules)


def load_voice_runtime_config(
    source: Optional[Path | str | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the independent top-level voice runtime configuration."""
    config = load_project_config(source)
    voice_config = config.get("voice_runtime")
    if not isinstance(voice_config, dict):
        raise ValueError("config.yaml must define voice_runtime as a mapping")
    return deepcopy(voice_config)
