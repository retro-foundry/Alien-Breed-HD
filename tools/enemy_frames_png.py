#!/usr/bin/env python3
"""
Export Alien Breed 3D I enemy sprite animation frames as PNG files.

This reads sprite assets from data/includes (*.wad, *.ptr) and data/pal (*.pal),
then decodes frame tables mirrored from src/renderer.c.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path


# Frame tables mirrored from src/renderer.c (subset relevant to enemy sprite slots).
def _frames_alien() -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(16):
        out.append((64 * 4 * i, 0))
    # Gib sets packed in 16x16 cells on the alien sheet.
    for dy in (0, 16, 32, 48):
        for dx in (0, 16, 32, 48):
            out.append((4 * (64 * 16 + dx), dy))
    out.append((64 * 4 * 17, 0))
    out.append((64 * 4 * 18, 0))
    return out


def _frames_flying() -> list[tuple[int, int]]:
    return [(64 * 4 * i, 0) for i in range(21)]


def _frames_marine() -> list[tuple[int, int]]:
    return [((64 * i) * 4, 0) for i in range(19)]


def _frames_bigalien() -> list[tuple[int, int]]:
    return [(128 * 4 * i, 0) for i in range(4)]


def _frames_worm() -> list[tuple[int, int]]:
    return [(90 * 4 * i, 0) for i in range(21)]


def _frames_bigclaws() -> list[tuple[int, int]]:
    return [(128 * 4 * i, 0) for i in range(18)]


def _frames_tree() -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for _ in range(4):
        out.extend([(0, 0), (64 * 4, 0), (64 * 2 * 4, 0), (64 * 3 * 4, 0)])
    out.extend([(0, 0), (0, 0), (32 * 8 * 4, 0), (32 * 9 * 4, 0), (32 * 10 * 4, 0), (32 * 11 * 4, 0)])
    return out


SPRITES = [
    {
        "name": "alien",
        "slot": 0,
        "wad": "alien2.wad",
        "ptr": "alien2.ptr",
        "pal": "alien2.pal",
        "eff_cols": 64,
        "eff_rows": 64,
        "frames": _frames_alien(),
    },
    {
        "name": "flying_nasty",
        "slot": 4,
        "wad": "flyingalien.wad",
        "ptr": "flyingalien.ptr",
        "pal": "flyingalien.pal",
        "eff_cols": 64,
        "eff_rows": 64,
        "frames": _frames_flying(),
    },
    {
        "name": "marine_mutant",
        "slot": 10,
        "wad": "newmarine.wad",
        "ptr": "newmarine.ptr",
        "pal": "newmarine.pal",
        "eff_cols": 64,
        "eff_rows": 64,
        "frames": _frames_marine(),
    },
    {
        "name": "big_ugly",
        "slot": 11,
        "wad": "bigscaryalien.wad",
        "ptr": "bigscaryalien.ptr",
        "pal": "bigscaryalien.pal",
        "eff_cols": 128,
        "eff_rows": 128,
        "frames": _frames_bigalien(),
    },
    {
        "name": "worm",
        "slot": 13,
        "wad": "worm.wad",
        "ptr": "worm.ptr",
        "pal": "worm.pal",
        "eff_cols": 90,
        "eff_rows": 100,
        "frames": _frames_worm(),
    },
    {
        "name": "red_big_or_small",
        "slot": 14,
        "wad": "bigclaws.wad",
        "ptr": "bigclaws.ptr",
        "pal": "bigclaws.pal",
        "eff_cols": 128,
        "eff_rows": 128,
        "frames": _frames_bigclaws(),
    },
    {
        "name": "tree",
        "slot": 15,
        "wad": "tree.wad",
        "ptr": "tree.ptr",
        "pal": "tree.pal",
        "eff_cols": 64,
        "eff_rows": 64,
        "frames": _frames_tree(),
    },
    {
        "name": "marine_tough",
        "slot": 16,
        "wad": "newmarine.wad",
        "ptr": "newmarine.ptr",
        "pal": "toughmutant.pal",
        "eff_cols": 64,
        "eff_rows": 64,
        "frames": _frames_marine(),
    },
    {
        "name": "marine_flame",
        "slot": 17,
        "wad": "newmarine.wad",
        "ptr": "newmarine.ptr",
        "pal": "flamemutant.pal",
        "eff_cols": 64,
        "eff_rows": 64,
        "frames": _frames_marine(),
    },
]


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(payload, zlib.crc32(tag)) & 0xFFFFFFFF)
    )


def write_png_rgba(path: Path, width: int, height: int, rgba: bytes) -> None:
    stride = width * 4
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(rgba[y * stride : (y + 1) * stride])

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
    png.extend(_png_chunk(b"IEND", b""))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def amiga12_to_rgb(word: int) -> tuple[int, int, int]:
    r = ((word >> 8) & 0xF) * 0x11
    g = ((word >> 4) & 0xF) * 0x11
    b = (word & 0xF) * 0x11
    return r, g, b


def find_case_insensitive(base: Path, wanted_name: str) -> Path:
    wanted = wanted_name.lower()
    for p in base.iterdir():
        if p.name.lower() == wanted:
            return p
    raise FileNotFoundError(f"Missing file in {base}: {wanted_name}")


def read_palette_32_mid(pal_path: Path) -> list[tuple[int, int, int]]:
    data = pal_path.read_bytes()
    if len(data) < 64:
        raise ValueError(f"Palette too small ({len(data)} bytes): {pal_path}")

    # Use brightness level 7 when available (15 levels x 64 bytes).
    base = 0
    if len(data) >= 64 * 15:
        base = 64 * 7

    cols: list[tuple[int, int, int]] = []
    for i in range(32):
        off = base + i * 2
        w = (data[off] << 8) | data[off + 1]
        cols.append(amiga12_to_rgb(w))
    return cols


def decode_frame_rgba(
    wad: bytes,
    ptr: bytes,
    pal_rgb: list[tuple[int, int, int]],
    ptr_off: int,
    down_strip: int,
    eff_cols: int,
    eff_rows: int,
) -> bytes:
    out = bytearray(eff_cols * eff_rows * 4)

    for x in range(eff_cols):
        entry_off = ptr_off + x * 4
        if entry_off + 4 > len(ptr):
            continue

        mode = ptr[entry_off]
        wad_off = (ptr[entry_off + 1] << 16) | (ptr[entry_off + 2] << 8) | ptr[entry_off + 3]
        if mode == 0 and wad_off == 0:
            continue
        if wad_off >= len(wad):
            continue

        src = wad[wad_off:]
        for y in range(eff_rows):
            row_idx = down_strip + y
            row_off = row_idx * 2
            if row_off + 1 >= len(src):
                continue

            w = (src[row_off] << 8) | src[row_off + 1]
            if mode == 0:
                idx = w & 31
            elif mode == 1:
                idx = (w >> 5) & 31
            else:
                idx = (w >> 10) & 31

            if idx == 0:
                continue

            r, g, b = pal_rgb[idx]
            o = (y * eff_cols + x) * 4
            out[o + 0] = r
            out[o + 1] = g
            out[o + 2] = b
            out[o + 3] = 255

    return bytes(out)


def export_enemy_frames(data_dir: Path, out_dir: Path) -> None:
    includes_dir = data_dir / "includes"
    pal_dir = data_dir / "pal"
    if not includes_dir.is_dir():
        raise FileNotFoundError(f"Missing includes directory: {includes_dir}")
    if not pal_dir.is_dir():
        raise FileNotFoundError(f"Missing pal directory: {pal_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "ab3d1_enemy_frames_v1",
        "source_data_dir": str(data_dir),
        "entries": [],
    }

    total = 0
    skipped: list[str] = []
    for spec in SPRITES:
        try:
            wad_path = find_case_insensitive(includes_dir, spec["wad"])
            ptr_path = find_case_insensitive(includes_dir, spec["ptr"])
            pal_path = find_case_insensitive(pal_dir, spec["pal"])
        except FileNotFoundError as exc:
            name = str(spec["name"])
            skipped.append(name)
            print(f"[enemy-frames] warning: skipping {name}: {exc}")
            continue

        wad = wad_path.read_bytes()
        ptr = ptr_path.read_bytes()
        pal = read_palette_32_mid(pal_path)

        sprite_name = str(spec["name"])
        sprite_out = out_dir / sprite_name
        sprite_out.mkdir(parents=True, exist_ok=True)

        frames = spec["frames"]
        eff_cols = int(spec["eff_cols"])
        eff_rows = int(spec["eff_rows"])

        for fi, (ptr_off, down_strip) in enumerate(frames):
            rgba = decode_frame_rgba(wad, ptr, pal, ptr_off, down_strip, eff_cols, eff_rows)
            write_png_rgba(sprite_out / f"frame_{fi:03d}.png", eff_cols, eff_rows, rgba)
            total += 1

        manifest["entries"].append(
            {
                "name": sprite_name,
                "slot": int(spec["slot"]),
                "wad": str(wad_path),
                "ptr": str(ptr_path),
                "pal": str(pal_path),
                "eff_size": [eff_cols, eff_rows],
                "frame_count": len(frames),
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[enemy-frames] Exported {total} frame PNGs to: {out_dir}")
    if skipped:
        print(f"[enemy-frames] Skipped {len(skipped)} set(s): {', '.join(skipped)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export AB3D1 enemy animation frames to PNG files.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path containing includes/ and pal/ folders (for example build/data).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write enemy frame PNGs into.",
    )
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        export_enemy_frames(args.data_dir.resolve(), args.out_dir.resolve())
    except Exception as exc:  # noqa: BLE001
        print(f"[enemy-frames] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
