"""JSON log formatting without domain-specific field or event checks."""

import json
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompactLogValue:
    """Mark a value for single-line rendering inside an indented JSON log."""

    value: Any


class LogJSONEncoder(json.JSONEncoder):
    """Honor compact values while retaining normal JSON objects and arrays."""

    def encode(self, value: Any) -> str:
        compact_values = {}

        def serialize(item: Any) -> str:
            if isinstance(item, CompactLogValue):
                marker = f"__compact_log_{uuid.uuid4().hex}__"
                compact_values[marker] = json.dumps(
                    item.value, ensure_ascii=self.ensure_ascii, default=str,
                )
                return marker
            return str(item)

        body = json.JSONEncoder(
            ensure_ascii=self.ensure_ascii, sort_keys=self.sort_keys,
            indent=self.indent, default=serialize,
        ).encode(value)
        for marker, compact in compact_values.items():
            body = body.replace(json.dumps(marker), compact)
        return body
