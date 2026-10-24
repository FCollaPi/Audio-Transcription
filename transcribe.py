#!/usr/bin/env python3
"""
transcribe.py — Simple CLI tool to transcribe audio using open-source Whisper models.

Uses faster-whisper (CTranslate2-based reimplementation of OpenAI Whisper) for
fast, low-memory transcription that runs well on CPU or GPU.

Attribution / sources:
- OpenAI Whisper (MIT, (c) 2022 OpenAI): format_timestamp() and the TXT/SRT/VTT
  writers below are adapted from whisper/utils.py (https://github.com/openai/whisper).
- SYSTRAN/faster-whisper (MIT, (c) 2023 SYSTRAN): transcription engine and CLI
  conventions (https://github.com/SYSTRAN/faster-whisper).
- Full third-party notices: see THIRD_PARTY_NOTICES.md.

Setup:
    pip install faster-whisper

Basic usage:
    python transcribe.py audio.mp3
    python transcribe.py audio.wav --model small --language en
    python transcribe.py podcast.m4a --model medium --format srt --output podcast.srt
    python transcribe.py meeting.mp3 --device cuda --compute-type float16

Model sizes (speed vs. accuracy tradeoff):
    tiny, base, small, medium, large-v2, large-v3, distil-large-v3
    (Larger = more accurate but slower and more memory. "small" is a good default.)
"""

import argparse
import sys
import time
from pathlib import Path


def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    """Format seconds as SRT/VTT timestamp: HH:MM:SS,mmm or HH:MM:SS.mmm

    Adapted from OpenAI Whisper's format_timestamp() (whisper/utils.py).
    MIT License, Copyright (c) 2022 OpenAI. See THIRD_PARTY_NOTICES.md.
    """
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}{decimal_marker}{milliseconds:03d}"


def write_txt(segments, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


def write_srt(segments, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start, decimal_marker=",")
            end = format_timestamp(seg.end, decimal_marker=",")
            f.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")


def write_vtt(segments, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_timestamp(
                seg.start, always_include_hours=False, decimal_marker="."
            )
            end = format_timestamp(
                seg.end, always_include_hours=False, decimal_marker="."
            )
            f.write(f"{start} --> {end}\n{seg.text.strip()}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using open-source Whisper models (via faster-whisper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio", help="Path to the audio file (mp3, wav, m4a, flac, etc.)"
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=[
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v1",
            "large-v2",
            "large-v3",
            "distil-large-v3",
        ],
        help="Model size to use.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Quantization/precision, e.g. int8, int8_float16, float16, float32. "
        "int8 is fastest/lightest on CPU; float16 is typical for GPU.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g. en, ja, es). Default: auto-detect.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'translate' converts speech in any language to English text.",
    )
    parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "srt", "vtt"],
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Default: same name as input with new extension.",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        default=True,
        help="Use voice-activity detection to skip silence (recommended, on by default).",
    )
    parser.add_argument(
        "--no-vad-filter",
        action="store_false",
        dest="vad_filter",
        help="Disable voice-activity filtering.",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "Error: faster-whisper is not installed.\n"
            "Install it with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = (
        Path(args.output) if args.output else audio_path.with_suffix(f".{args.format}")
    )

    print(f"Loading model '{args.model}' on {args.device} ({args.compute_type})...")
    t0 = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    print(f"Model loaded in {time.time() - t0:.1f}s")

    print(f"Transcribing {audio_path.name}...")
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        task=args.task,
        vad_filter=args.vad_filter,
    )

    print(
        f"Detected language: {info.language} (confidence {info.language_probability:.2f})"
    )

    # segments is a generator; materialize while printing progress
    collected = []
    for seg in segments:
        collected.append(seg)
        print(
            f"  [{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] {seg.text.strip()}"
        )

    elapsed = time.time() - t0
    print(f"Transcription finished in {elapsed:.1f}s ({info.duration:.1f}s of audio)")

    writer = {"txt": write_txt, "srt": write_srt, "vtt": write_vtt}[args.format]
    writer(collected, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
