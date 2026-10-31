"""Shared chat-completion client used by memory domain components."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional

import requests


DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"


class LLMClient:
    """Small OpenAI-compatible chat-completion client with format fallbacks."""

    def __init__(
        self,
        *,
        memory_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        config = dict(memory_config or {})
        llm_config = config.get("llm")
        self._config = dict(llm_config) if isinstance(llm_config, dict) else {}
        self._logger = logger or logging.getLogger(__name__)
        self._model = str(self._config.get("llm_name") or DEFAULT_LLM_MODEL)
        self._base_url = self._normalize_base_url(
            str(self._config.get("llm_base_url") or DEFAULT_LLM_BASE_URL)
        )
        self._api_key = self._resolve_env(self._config.get("llm_api_key"))
        self._timeout = int(self._config.get("llm_timeout", 120) or 120)
        self._max_tokens = max(
            1, int(self._config.get("llm_max_tokens", 4096) or 4096)
        )
        self._json_mode = self._config_bool(
            self._config.get("llm_json_mode", True), True,
        )
        self._thinking = str(
            self._config.get("llm_thinking", "disabled") or "disabled"
        )

    def set_logger(self, logger: Optional[logging.Logger]) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def call_llm(self, prompt: str) -> Optional[str]:
        if (
            not self._api_key
            or not self._base_url
            or self._base_url.strip().lower() == "none"
        ):
            self._logger.debug(
                "Skipping LLM call because llm_api_key or llm_base_url is not configured"
            )
            return None
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
            "max_tokens": self._max_tokens,
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._thinking in {"disabled", "enabled"}:
            payload["thinking"] = {"type": self._thinking}

        attempts = [payload]
        if "thinking" in payload:
            stripped = dict(payload)
            stripped.pop("thinking", None)
            attempts.append(stripped)
        if "response_format" in payload:
            stripped = dict(payload)
            stripped.pop("response_format", None)
            attempts.append(stripped)

        seen = set()
        call_id = uuid.uuid4().hex[:12]
        call_started_at = time.monotonic()
        request_number = 0
        for item in attempts:
            marker = json.dumps(sorted(item.keys()), ensure_ascii=False)
            if marker in seen:
                continue
            seen.add(marker)
            request_number += 1
            request_started_at = time.monotonic()
            response: Optional[requests.Response] = None
            self._logger.info(
                "LLM request started call_id=%s attempt=%s model=%s prompt_chars=%s "
                "timeout_seconds=%s json_mode=%s thinking=%s",
                call_id, request_number, self._model, len(prompt), self._timeout,
                "response_format" in item, "thinking" in item,
            )
            try:
                response = requests.post(
                    url,
                    json=item,
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    content = message.get("content")
                    if content:
                        self._logger.info(
                            "LLM request finished call_id=%s attempt=%s outcome=content "
                            "status_code=%s elapsed_ms=%.1f response_chars=%s",
                            call_id, request_number, response.status_code,
                            (time.monotonic() - request_started_at) * 1000,
                            len(str(content)),
                        )
                        return str(content)
                self._logger.warning(
                    "LLM request finished call_id=%s attempt=%s outcome=empty_content "
                    "status_code=%s elapsed_ms=%.1f",
                    call_id, request_number, response.status_code,
                    (time.monotonic() - request_started_at) * 1000,
                )
            except requests.RequestException as exc:
                text = str(exc).lower()
                self._logger.warning(
                    "LLM request finished call_id=%s attempt=%s outcome=request_error "
                    "status_code=%s elapsed_ms=%.1f error_type=%s",
                    call_id, request_number,
                    response.status_code if response is not None else None,
                    (time.monotonic() - request_started_at) * 1000,
                    type(exc).__name__,
                )
                if "response_format" in text or "json" in text or "thinking" in text:
                    continue
                return None
        self._logger.warning(
            "LLM call finished call_id=%s outcome=no_content attempts=%s elapsed_ms=%.1f",
            call_id, request_number, (time.monotonic() - call_started_at) * 1000,
        )
        return None

    @staticmethod
    def parse_json_object_from_llm_text(text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("```"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
        else:
            start = raw.find("{")
            end = raw.rfind("}")
            if 0 <= start < end and (start > 0 or end < len(raw) - 1):
                raw = raw[start : end + 1]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _config_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        return default

    @staticmethod
    def _resolve_env(value: Any) -> str:
        text = str(value or "").strip()
        match = re.fullmatch(r"\${([A-Za-z_][A-Za-z0-9_]*)}", text)
        if match:
            return os.environ.get(match.group(1), "").strip()
        return text

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        text = str(value or DEFAULT_LLM_BASE_URL).strip().rstrip("/")
        if text == "https://api.deepseek.com":
            return "https://api.deepseek.com/v1"
        return text or DEFAULT_LLM_BASE_URL
