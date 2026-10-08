"""Shared in-memory models for the Runtime-to-Manager memory pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MemoryUnit:
    """One ordered piece of memory evidence.

    ``unit_type`` is either ``transcript`` or ``interaction``.  The values at
    each index of ``texts``, ``speakers``, ``started_ats``, and ``ended_ats``
    describe one aligned source entry.  A transcript normally has one entry;
    an interaction normally has the ordered user and assistant entries.

    The unit-level time envelope used by online chapter segmentation is always
    derived from the first ``started_ats`` value and the last ``ended_ats``
    value. ``embedding`` remains Runtime-only and is never persisted or sent
    to the Manager.
    """

    texts: List[str]
    speakers: List[str]
    token_count: int
    started_ats: List[str] = field(default_factory=list)
    ended_ats: List[str] = field(default_factory=list)
    unit_type: str = "transcript"
    tags: List[str] = field(default_factory=list)
    source_unit_row_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.unit_type not in {"transcript", "interaction"}:
            raise ValueError(
                "MemoryUnit.unit_type must be 'transcript' or 'interaction'",
            )

    @property
    def text(self) -> str:
        """Return a joined view used only by segmentation and lexical logic."""
        return " ".join(str(value).strip() for value in self.texts if str(value).strip())

    @property
    def start_time(self) -> str:
        """Return the first source entry's start time, if available."""
        return str(self.started_ats[0] or "").strip() if self.started_ats else ""

    @property
    def end_time(self) -> str:
        """Return the last source entry's end time, falling back to start."""
        if self.ended_ats:
            value = str(self.ended_ats[-1] or "").strip()
            if value:
                return value
        return self.start_time

    def entries(self) -> List[Dict[str, Any]]:
        """Return the aligned source entries in their original order."""
        entries: List[Dict[str, Any]] = []
        for index, value in enumerate(self.texts or []):
            text = str(value or "").strip()
            if not text:
                continue
            started_at = (
                str(self.started_ats[index] or "").strip()
                if index < len(self.started_ats)
                else ""
            )
            ended_at = (
                str(self.ended_ats[index] or "").strip()
                if index < len(self.ended_ats)
                else started_at
            )
            entries.append({
                "text": text,
                "speaker": (
                    str(self.speakers[index] or "").strip()
                    if index < len(self.speakers)
                    else ""
                ),
                "started_at": started_at,
                "ended_at": ended_at or started_at,
            })
        return entries
