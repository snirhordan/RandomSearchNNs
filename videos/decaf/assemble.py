"""Concatenate the rendered scenes into one video and build subtitles.

    python assemble.py [--out build/DeCAF_explained.mp4]

Reads build/videos/<Scene>.mp4 (written by manimgl) and build/timings/<Scene>.json
(written by DecafScene.tear_down), writes the final MP4 with a soft English
subtitle track, plus a standalone .srt next to it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from narration import SCENES, caption

HERE = Path(__file__).resolve().parent
VIDEOS = HERE / "build" / "videos"
TIMINGS = HERE / "build" / "timings"
ORDER = list(SCENES)

_W_LETTER, _W_SPACE = 0.0588, 0.049
_W_PUNCT = {".": 0.199, ",": 0.229, ":": 0.361, ";": 0.361, "?": 0.199}


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def weights(s: str) -> float:
    return sum(_W_LETTER if c.isalnum() else _W_SPACE if c == " " else _W_PUNCT.get(c, 0.0) for c in s)


def chunks(text: str, max_chars: int = 84) -> list[str]:
    """Split narration into subtitle-sized pieces at sentence/clause boundaries."""
    sentences = re.split(r"(?<=[.:?])\s+", text.strip())
    out = []
    for sent in sentences:
        while len(sent) > max_chars:
            cut = max(sent.rfind(", ", 0, max_chars), sent.rfind(" ", 0, max_chars))
            cut = cut + 1 if sent[cut] == "," else cut
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out


def fmt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(offsets: dict[str, float]) -> str:
    entries = []
    for scene in ORDER:
        info = json.loads((TIMINGS / f"{scene}.json").read_text())
        for beat in info["beats"]:
            pieces = chunks(caption(beat["text"]))
            w = np.array([weights(p) for p in pieces])
            ends = np.cumsum(w) / w.sum() * beat["duration"]
            starts = np.concatenate([[0], ends[:-1]])
            base = offsets[scene] + beat["start"]
            for p, a, b in zip(pieces, starts, ends):
                entries.append((base + a, base + b, p))
    lines = []
    for i, (a, b, p) in enumerate(entries, 1):
        lines += [str(i), f"{fmt(a)} --> {fmt(b - 0.05)}", p, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "build" / "DeCAF_explained.mp4"))
    ap.add_argument("--crf", default="20")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [VIDEOS / f"{s}.mp4" for s in ORDER]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing renders: {missing}")

    offsets, t = {}, 0.0
    for scene, f in zip(ORDER, files):
        offsets[scene] = t
        t += duration(f)
    print(f"total duration: {t / 60:.2f} min")

    srt_path = out.with_suffix(".srt")
    srt_path.write_text(build_srt(offsets))

    listing = out.parent / "concat.txt"
    listing.write_text("".join(f"file '{f}'\n" for f in files))
    joined = out.parent / "joined_tmp.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", "libx264", "-preset", "medium", "-crf", args.crf, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(joined),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-i", str(srt_path),
        "-map", "0:v", "-map", "0:a", "-map", "1:s", "-c:v", "copy", "-c:a", "copy",
        "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
        "-metadata", "title=DeCAF: few-step cofolding with all-atom flow maps, explained",
        "-movflags", "+faststart", str(out),
    ], check=True)
    joined.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {srt_path.name}")


if __name__ == "__main__":
    main()
