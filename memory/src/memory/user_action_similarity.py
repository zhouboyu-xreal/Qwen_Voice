"""Broad action-text similarity for observed-pattern evidence retrieval."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any


_OWNER_WORDS = ("speaker_用户本人", "用户本人", "本人", "the user", "user self")
_TIME_WORDS = (
    "今天", "昨天", "前天", "刚刚", "最近", "近期", "周末", "早上", "上午",
    "中午", "下午", "晚上", "夜里", "每周", "每天", "每月", "经常", "通常",
    "today", "yesterday", "recently", "morning", "afternoon", "evening",
    "every", "often", "usually",
)
_DATE_WORDS = re.compile(
    r"\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?"
    r"|(?:每)?(?:星期|周)[一二三四五六日天1-7]"
    r"|\d{1,2}(?::\d{2}|点(?:\d{1,2}分?)?)"
)


def normalize_action_text(value: Any) -> str:
    """Remove speaker and time phrasing while retaining the action itself."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _DATE_WORDS.sub("", text)
    for word in (*_OWNER_WORDS, *_TIME_WORDS):
        text = text.replace(word, "")
    return "".join(char for char in text if char.isalnum())


def action_lexical_similarity(left: Any, right: Any) -> float:
    """Find broad lexical neighbors; the LLM makes the pattern decision."""
    a, b = normalize_action_text(left), normalize_action_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if min(len(a), len(b)) >= 3 and (a in b or b in a):
        return 0.88
    left_pairs = {a[index:index + 2] for index in range(len(a) - 1)}
    right_pairs = {b[index:index + 2] for index in range(len(b) - 1)}
    overlap = (
        2.0 * len(left_pairs & right_pairs) / (len(left_pairs) + len(right_pairs))
        if left_pairs and right_pairs else 0.0
    )
    return max(SequenceMatcher(None, a, b).ratio(), overlap)
