#!/usr/bin/env python3
"""
Convert raw 8-bit signed PCM files in a sounds directory to .wav
(8-bit unsigned, mono).

Default input/output directory is repo-root data/sounds, but a custom
directory can be passed with --sounds-dir.
"""
import argparse
import io
from pathlib import Path
import shutil
import struct
import subprocess
import sys

# Amiga AB3DI.s sets AUDxPER = 443 for all SFX channels.
# Paula PAL audio clock is 3546895 Hz (half the 7.09 MHz master clock).
SAMPLE_RATE = (3546895 + (443 // 2)) // 443  # 8007 Hz
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOUNDS_DIR = REPO_ROOT / "data" / "sounds"


def write_wav_header(f, num_samples: int) -> None:
    """Write minimal 44-byte WAV header (RIFF + fmt + data)."""
    data_size = num_samples
    riff_size = 36 + data_size
    # RIFF header
    f.write(b"RIFF")
    f.write(struct.pack("<I", riff_size))
    f.write(b"WAVE")
    # fmt chunk: PCM 8-bit mono
    f.write(b"fmt ")
    f.write(struct.pack("<I", 16))  # chunk size
    f.write(struct.pack("<H", 1))   # PCM
    f.write(struct.pack("<H", 1))   # mono
    f.write(struct.pack("<I", SAMPLE_RATE))
    f.write(struct.pack("<I", SAMPLE_RATE))  # byte rate
    f.write(struct.pack("<H", 1))   # block align
    f.write(struct.pack("<H", 8))   # bits per sample
    # data chunk
    f.write(b"data")
    f.write(struct.pack("<I", data_size))


def build_wav_bytes(raw: bytes) -> bytes:
    """Return complete WAV bytes for signed 8-bit raw samples."""
    samples = bytearray((b + 128) & 0xFF for b in raw)
    out = io.BytesIO()
    write_wav_header(out, len(samples))
    out.write(samples)
    return out.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Amiga raw SFX files to WAV.")
    parser.add_argument(
        "--sounds-dir",
        type=Path,
        default=DEFAULT_SOUNDS_DIR,
        help=f"Directory containing raw sound files (default: {DEFAULT_SOUNDS_DIR})",
    )
    return parser.parse_args()


def convert_mt_to_wav(mt_path: Path, out_path: Path) -> bool:
    """
    Convert a tracker module (.mt) to WAV via ffmpeg/libopenmpt.
    Returns True when output was created/updated, False when unchanged.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"[raw_to_wav] ffmpeg not found; skipping {mt_path.name}", file=sys.stderr)
        return False

    if out_path.exists() and out_path.stat().st_mtime >= mt_path.stat().st_mtime:
        return False

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(mt_path),
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(tmp),
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"[raw_to_wav] ffmpeg failed for {mt_path}", file=sys.stderr)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False

    try:
        if out_path.exists() and out_path.read_bytes() == tmp.read_bytes():
            tmp.unlink()
            return False
    except OSError:
        pass

    tmp.replace(out_path)
    return True


def main() -> int:
    args = parse_args()
    sounds_dir = args.sounds_dir

    if not sounds_dir.is_dir():
        print(f"[raw_to_wav] No directory: {sounds_dir}", file=sys.stderr)
        return 0  # not fatal

    def is_convertible(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.suffix.lower() == ".wav":
            return False
        if p.name.startswith(".") or p.name.lower() == "readme.md":
            return False

        rel = p.relative_to(sounds_dir)
        # Keep existing behaviour for top-level SFX files, and also include
        # tracker music dumps in nested paths (e.g. sounds/mt/*.mt).
        return rel.parent == Path(".") or p.suffix.lower() == ".mt"

    converted = 0
    for p in sorted(sounds_dir.rglob("*")):
        if not is_convertible(p):
            continue

        if p.suffix.lower() == ".mt":
            out = p.parent / (p.stem.lower() + ".wav")
            legacy = p.parent / (p.name.lower() + ".wav")
            if convert_mt_to_wav(p, out):
                converted += 1
                print(f"  {p.name} -> {out.name}")
            if legacy != out and legacy.exists():
                try:
                    legacy.unlink()
                    print(f"  removed legacy {legacy.name}")
                except OSError:
                    pass
            continue

        raw = p.read_bytes()
        if len(raw) == 0:
            continue

        out = p.parent / (p.name.lower() + ".wav")
        wav_bytes = build_wav_bytes(raw)
        if out.exists():
            try:
                if out.read_bytes() == wav_bytes:
                    continue
            except OSError:
                pass

        with open(out, "wb") as f:
            f.write(wav_bytes)
        converted += 1
        print(f"  {p.name} -> {out.name}")

    if converted:
        print(f"[raw_to_wav] Converted {converted} file(s) to .wav in {sounds_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
