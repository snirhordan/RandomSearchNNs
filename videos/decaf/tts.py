"""Generate the voice-over, one WAV per narration beat, with Kokoro TTS.

Usage:
    python tts.py --model kokoro-v1.0.onnx --voices voices-v1.0.bin

Writes sounds/<Scene>__<beat>.wav and sounds/durations.json.  The model files
come from https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0).
Only beats whose text changed since the last run are re-synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from narration import SCENES

HERE = Path(__file__).resolve().parent
OUT = HERE / "sounds"
TARGET_RMS = 0.1
VERSION = "2"  # bump to force re-synthesis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    from kokoro_onnx import Kokoro

    kokoro = Kokoro(args.model, args.voices)
    OUT.mkdir(exist_ok=True)
    meta_path = OUT / "durations.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for scene, beats in SCENES.items():
        for key, text in beats:
            name = f"{scene}__{key}"
            digest = hashlib.sha1(f"{VERSION}|{args.voice}|{args.speed}|{text}".encode()).hexdigest()
            wav = OUT / f"{name}.wav"
            if wav.exists() and meta.get(name, {}).get("sha1") == digest:
                continue
            samples, sr = kokoro.create(text, voice=args.voice, speed=args.speed, lang="en-us")
            samples = np.asarray(samples, dtype=np.float32)
            # Trim leading/trailing near-silence so beat timing is predictable.
            loud = np.flatnonzero(np.abs(samples) > 0.01)
            if loud.size:
                pad = int(0.05 * sr)
                samples = samples[max(loud[0] - pad, 0): loud[-1] + pad]
            # Loudness: match RMS over voiced frames, then keep peaks below 0.95.
            voiced = samples[np.abs(samples) > 0.01]
            if voiced.size:
                samples = samples * (TARGET_RMS / np.sqrt(np.mean(voiced**2)))
            peak = np.abs(samples).max()
            if peak > 0.95:
                samples = samples * (0.95 / peak)
            sf.write(wav, samples, sr)
            meta[name] = {"sha1": digest, "duration": len(samples) / sr}
            print(f"{name:32s} {len(samples) / sr:6.2f}s")

    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    total = sum(v["duration"] for v in meta.values())
    print(f"total speech: {total / 60:.2f} min")


if __name__ == "__main__":
    main()
