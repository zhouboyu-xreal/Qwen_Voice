#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
print(SRC)
from voice_recording.textgrid import load_textgrid_segments


DEFAULT_TEXTGRID = ROOT / "external/Eval_Ali/Eval_Ali_near/textgrid_dir/R8001_M8004_N_SPK8016.TextGrid"
DEFAULT_OUTPUT = ROOT / "external/Eval_Ali/Eval_Ali_near/transcript_dir/R8001_M8004_N_SPK8016_transcript.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Praat TextGrid intervals into a time-ordered transcript txt file."
    )
    parser.add_argument(
        "--textgrid",
        type=Path,
        default=DEFAULT_TEXTGRID,
        help=f"Input TextGrid path. Default: {DEFAULT_TEXTGRID}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output txt path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    textgrid_path = args.textgrid.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not textgrid_path.exists():
        raise FileNotFoundError(f"TextGrid file not found: {textgrid_path}")

    segments = load_textgrid_segments(textgrid_path)
    if not segments:
        raise ValueError(f"No non-empty intervals found in TextGrid: {textgrid_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for segment in segments:
            speaker = segment.speaker_name or f"speaker_{segment.speaker_id}"
            text = (segment.reference_text or segment.text).strip()
            file.write(f"{speaker}\t{segment.start:.3f}-{segment.end:.3f}\t{text}\n")

    print(f"Exported {len(segments)} intervals to {output_path}")


if __name__ == "__main__":
    main()
