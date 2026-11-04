"""Pure Feishu Markdown parsing shared by desktop import and the CLI fixture.

Relative offsets are utterance starts, never fabricated speech durations.
Preview leaves all speaker identities unchanged; import requires an explicit
owner selection and maps only that exact name (or its room suffix).
"""
from __future__ import annotations
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SHANGHAI = timezone(timedelta(hours=8))
OWNER_SPEAKER = "speaker_用户本人"
HEADER_RE = re.compile(r"^(?P<speaker>.+?)\s+(?P<hours>\d{1,3}):(?P<minutes>\d{2}):(?P<seconds>\d{2})$")
DATE_RE = re.compile(r"(?P<year>\d{4})[年/-](?P<month>\d{1,2})[月/-](?P<day>\d{1,2})日?")
TIME_RANGE_RE = re.compile(r"(?P<start>\d{1,2}:\d{2})\s*[-–—~～至]\s*(?P<end>\d{1,2}:\d{2})")
TIMEZONE_RE = re.compile(r"GMT\s*(?P<sign>[+-])(?P<hours>\d{1,2})(?::(?P<minutes>\d{2}))?")
MARKDOWN_ESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~\\])")

def plain_markdown(text: str) -> str:
    """Undo Markdown punctuation escapes without interpreting document commands."""
    return MARKDOWN_ESCAPE_RE.sub(r"\1", text).strip()


def speaker_name(text: str) -> str:
    text = plain_markdown(text).strip()
    return text[1:].strip() if text.startswith("@") else text


def resolve_meeting_start(
    *, title: str, time_description: str, filename: str, override: str,
) -> datetime:
    if override:
        value = datetime.fromisoformat(override.replace("Z", "+00:00"))
        return value if value.tzinfo is not None else value.replace(tzinfo=SHANGHAI)
    match = DATE_RE.search(time_description) or DATE_RE.search(title) or DATE_RE.search(filename)
    time_range = TIME_RANGE_RE.search(time_description)
    if match is None or time_range is None:
        raise ValueError("Cannot determine meeting date/start time; supply --session-start ISO_TIMESTAMP")
    day = date(**{key: int(value) for key, value in match.groupdict().items()})
    tz = SHANGHAI
    zone_match = TIMEZONE_RE.search(time_description)
    if zone_match:
        hours = int(zone_match["hours"])
        minutes = int(zone_match["minutes"] or 0)
        if minutes >= 60:
            raise ValueError("Invalid GMT offset in meeting metadata")
        offset = timedelta(hours=hours, minutes=minutes)
        tz = timezone(offset if zone_match["sign"] == "+" else -offset)
    hour, minute = map(int, time_range["start"].split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)


def read_meeting_metadata(lines: Iterable[str]) -> Dict[str, str]:
    """Read only the export header, never treating speech as metadata."""
    metadata = {"title": "", "heading": "", "time_description": ""}
    for raw_line in lines:
        line = plain_markdown(raw_line)
        if HEADER_RE.fullmatch(line):
            break
        if line.startswith("# ") and not metadata["heading"]:
            metadata["heading"] = line[2:].strip()
            if not metadata["title"]:
                metadata["title"] = metadata["heading"]
        elif line.lstrip("> ").startswith("会议主题："):
            metadata["title"] = line.lstrip("> ").split("：", 1)[1].strip() or metadata["title"]
        elif line.lstrip("> ").startswith("会议时间："):
            metadata["time_description"] = line.lstrip("> ").split("：", 1)[1].strip()
    return metadata


def parse_meeting_text(text: str, *, filename: str, owner_name: str = "", session_start: str = "") -> Dict[str, Any]:
    """Read Feishu speaker/HH:MM:SS blocks, keeping text and source line numbers."""
    path = Path(filename)
    owner = speaker_name(owner_name)
    lines = text.lstrip("\ufeff").splitlines()
    metadata = read_meeting_metadata(lines)
    title = metadata["title"]
    time_description = metadata["time_description"]
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    body: List[str] = []

    def finish_block() -> None:
        if current is None:
            return
        text = plain_markdown("\n".join(body))
        if not text:
            raise ValueError(f"Empty speech block at line {current['source_line']} in {path}")
        blocks.append({**current, "text": text})

    for line_number, raw_line in enumerate(lines, 1):
        line = plain_markdown(raw_line)
        header = HEADER_RE.fullmatch(line)
        if header:
            finish_block()
            hour, minute, second = (int(header[key]) for key in ("hours", "minutes", "seconds"))
            if minute >= 60 or second >= 60:
                raise ValueError(f"Invalid speech offset at line {line_number}: {line}")
            current = {
                "original_speaker": speaker_name(header["speaker"]),
                "offset_seconds": hour * 3600 + minute * 60 + second,
                "source_line": line_number,
            }
            body = []
        elif current is not None:
            body.append(raw_line)
    finish_block()
    if not blocks:
        raise ValueError(f"No speaker/timestamp speech blocks found in {path}")
    # The topic may omit the date while the H1 includes it.
    start = resolve_meeting_start(
        title=metadata["heading"] or title, time_description=time_description,
        filename=path.name, override=session_start,
    )
    source_speakers = list(dict.fromkeys(item["original_speaker"] for item in blocks))
    owner_labels = {
        name for name in source_speakers
        if name == owner or name.startswith(f"{owner} 在 ")
    }
    if owner and not owner_labels:
        raise ValueError(f"Owner {owner!r} not found among speakers: {source_speakers}")
    # Never overwrite a non-owner export label that already uses the reserved ID.
    if owner and OWNER_SPEAKER in source_speakers and owner != OWNER_SPEAKER:
        raise ValueError("A non-owner speaker already uses the reserved owner label")
    mapping = {name: OWNER_SPEAKER if name in owner_labels else name for name in source_speakers}
    reordered = any(
        left["offset_seconds"] > right["offset_seconds"]
        for left, right in zip(blocks, blocks[1:])
    )
    blocks.sort(key=lambda item: item["offset_seconds"])  # Stable for simultaneous speakers.
    segments = []
    for index, item in enumerate(blocks, 1):
        timestamp = (start + timedelta(seconds=item["offset_seconds"])).isoformat()
        segments.append({
            "speaker": mapping[item["original_speaker"]],
            "role": "ambient_speaker", "text": item["text"],
            "started_at": timestamp, "ended_at": timestamp,
            "segment_index": index,
            "metadata": {
                "source": "feishu_markdown", "source_file": str(path),
                "source_line": item["source_line"],
                "original_speaker": item["original_speaker"],
                "offset_seconds": item["offset_seconds"],
                "audio_start_s": item["offset_seconds"],
                "audio_end_s": item["offset_seconds"],
                "end_time_basis": "unknown_use_start_time",
            },
        })
    return {
        "source_file": str(path), "meeting_title": title,
        "meeting_time_description": time_description,
        "session_start": start.isoformat(), "session_start_overridden": bool(session_start),
        "last_utterance_start": segments[-1]["started_at"],
        "timestamp_policy": "exported relative starts; ends unknown and set equal to starts",
        "owner_name": owner, "owner_speaker": OWNER_SPEAKER,
        "speaker_mapping": mapping, "speakers": source_speakers, "source_segment_count": len(segments),
        "reordered_by_timestamp": reordered, "segments": segments,
    }


def parse_meeting(path: Path, *, owner_name: str, session_start: str = "") -> Dict[str, Any]:
    if not speaker_name(owner_name):
        raise ValueError("--owner-name must not be empty")
    return parse_meeting_text(path.read_text(encoding="utf-8-sig"), filename=str(path),
                              owner_name=owner_name, session_start=session_start)
