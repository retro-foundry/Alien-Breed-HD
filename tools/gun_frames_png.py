#!/usr/bin/env python3
"""
Export/import Alien Breed 3D I in-hand weapon frames as PNG files.

This tool round-trips the in-game gun overlay assets:
  - export: newgunsinhand.wad/.ptr/.pal -> gunX_frameY.png files
  - import: edited PNG files -> replacement newgunsinhand.wad/.ptr/.pal

PNG implementation is built-in (RGBA8, non-interlaced), so no extra deps needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

GUN_COLS = 96
GUN_LINES = 58
GUN_STRIDE = GUN_COLS * 4
PTR_FRAME_COUNT = 28
SLOT_COUNT = 32  # 8 guns * 4 animation frames

# Matches src/renderer.c gun_ptr_frame_offsets.
GUN_PTR_FRAME_OFFSETS = [
    GUN_STRIDE * 20,
    GUN_STRIDE * 21,
    GUN_STRIDE * 22,
    GUN_STRIDE * 23,  # gun 0
    GUN_STRIDE * 4,
    GUN_STRIDE * 5,
    GUN_STRIDE * 6,
    GUN_STRIDE * 7,  # gun 1
    GUN_STRIDE * 16,
    GUN_STRIDE * 17,
    GUN_STRIDE * 18,
    GUN_STRIDE * 19,  # gun 2
    GUN_STRIDE * 12,
    GUN_STRIDE * 13,
    GUN_STRIDE * 14,
    GUN_STRIDE * 15,  # gun 3
    GUN_STRIDE * 24,
    GUN_STRIDE * 25,
    GUN_STRIDE * 26,
    GUN_STRIDE * 27,  # gun 4
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,  # guns 5,6 are intentionally unused
    GUN_STRIDE * 0,
    GUN_STRIDE * 1,
    GUN_STRIDE * 2,
    GUN_STRIDE * 3,  # gun 7
]

UNUSED_GUNS = {5, 6}


def slot_name(slot: int) -> str:
    gun = slot // 4
    frame = slot % 4
    return f"gun{gun}_frame{frame}.png"


def amiga12_to_rgb(word: int) -> tuple[int, int, int]:
    r = ((word >> 8) & 0xF) * 0x11
    g = ((word >> 4) & 0xF) * 0x11
    b = (word & 0xF) * 0x11
    return r, g, b


def read_palette(pal_path: Path) -> list[tuple[int, int, int]]:
    data = pal_path.read_bytes()
    if len(data) < 64:
        raise ValueError(f"Palette too small ({len(data)} bytes): {pal_path}")
    cols: list[tuple[int, int, int]] = []
    for i in range(32):
        w = (data[i * 2] << 8) | data[i * 2 + 1]
        cols.append(amiga12_to_rgb(w))
    return cols


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(payload, zlib.crc32(tag)) & 0xFFFFFFFF)
    )


def write_png_rgba(path: Path, width: int, height: int, rgba: bytes) -> None:
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA buffer size mismatch")
    stride = width * 4
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter 0 (None)
        row = rgba[y * stride : (y + 1) * stride]
        raw.extend(row)
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # RGBA8
    png.extend(_png_chunk(b"IHDR", ihdr))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
    png.extend(_png_chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgba(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")

    off = 8
    width = 0
    height = 0
    idat = bytearray()
    while off + 12 <= len(data):
        length = struct.unpack(">I", data[off : off + 4])[0]
        off += 4
        ctype = data[off : off + 4]
        off += 4
        if off + length + 4 > len(data):
            raise ValueError(f"Corrupt PNG chunk length in {path}")
        payload = data[off : off + length]
        off += length
        _crc = data[off : off + 4]
        off += 4

        if ctype == b"IHDR":
            if length != 13:
                raise ValueError(f"Unexpected IHDR size in {path}")
            width, height, bit_depth, color_type, comp, flt, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if bit_depth != 8 or color_type != 6 or comp != 0 or flt != 0 or interlace != 0:
                raise ValueError(
                    f"Unsupported PNG format in {path} "
                    f"(need RGBA8, non-interlaced; got bit_depth={bit_depth}, color_type={color_type}, interlace={interlace})"
                )
        elif ctype == b"IDAT":
            idat.extend(payload)
        elif ctype == b"IEND":
            break

    if width <= 0 or height <= 0:
        raise ValueError(f"PNG missing valid IHDR: {path}")
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    expected = (stride + 1) * height
    if len(raw) != expected:
        raise ValueError(f"Unexpected decompressed PNG size for {path}: {len(raw)} != {expected}")

    out = bytearray(width * height * 4)
    src_off = 0
    dst_off = 0
    prev = bytearray(stride)
    for _y in range(height):
        f = raw[src_off]
        src_off += 1
        cur = bytearray(raw[src_off : src_off + stride])
        src_off += stride
        if f == 0:
            pass
        elif f == 1:
            for i in range(stride):
                left = cur[i - 4] if i >= 4 else 0
                cur[i] = (cur[i] + left) & 0xFF
        elif f == 2:
            for i in range(stride):
                cur[i] = (cur[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(stride):
                left = cur[i - 4] if i >= 4 else 0
                cur[i] = (cur[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = cur[i - 4] if i >= 4 else 0
                b = prev[i]
                c = prev[i - 4] if i >= 4 else 0
                cur[i] = (cur[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter type {f} in {path}")
        out[dst_off : dst_off + stride] = cur
        dst_off += stride
        prev = cur
    return width, height, bytes(out)


def decode_slot_to_rgba(slot: int, wad: bytes, ptr: bytes, pal_rgb: list[tuple[int, int, int]]) -> bytes:
    out = bytearray(GUN_COLS * GUN_LINES * 4)
    gun = slot // 4
    ptr_off = GUN_PTR_FRAME_OFFSETS[slot]
    if gun in UNUSED_GUNS:
        return bytes(out)
    if ptr_off + GUN_STRIDE > len(ptr):
        raise ValueError(f"PTR too small for slot {slot} (need offset {ptr_off})")

    for y in range(GUN_LINES):
        for x in range(GUN_COLS):
            p = ptr_off + x * 4
            mode = ptr[p]
            wad_off = (ptr[p + 1] << 16) | (ptr[p + 2] << 8) | ptr[p + 3]
            if mode == 0 and wad_off == 0:
                continue
            row_off = wad_off + y * 2
            if row_off + 1 >= len(wad):
                continue
            w = (wad[row_off] << 8) | wad[row_off + 1]
            if mode == 0:
                idx = w & 31
            elif mode == 1:
                idx = (w >> 5) & 31
            else:
                idx = (w >> 10) & 31
            if idx == 0:
                continue
            r, g, b = pal_rgb[idx]
            o = (y * GUN_COLS + x) * 4
            out[o + 0] = r
            out[o + 1] = g
            out[o + 2] = b
            out[o + 3] = 255
    return bytes(out)


def quantize_rgba_to_indices(rgba: bytes, pal_rgb: list[tuple[int, int, int]]) -> bytearray:
    out = bytearray(GUN_COLS * GUN_LINES)
    colors = pal_rgb[1:32]
    for i in range(GUN_COLS * GUN_LINES):
        r = rgba[i * 4 + 0]
        g = rgba[i * 4 + 1]
        b = rgba[i * 4 + 2]
        a = rgba[i * 4 + 3]
        if a < 128:
            out[i] = 0
            continue
        best_idx = 1
        br, bg, bb = colors[0]
        best_dist = (r - br) * (r - br) + (g - bg) * (g - bg) + (b - bb) * (b - bb)
        for rel_idx, (pr, pg, pb) in enumerate(colors[1:], start=2):
            d = (r - pr) * (r - pr) + (g - pg) * (g - pg) + (b - pb) * (b - pb)
            if d < best_dist:
                best_dist = d
                best_idx = rel_idx
        out[i] = best_idx
    return out


def write_palette_preview(path: Path, pal_rgb: list[tuple[int, int, int]]) -> None:
    sw = 24
    sh = 24
    width = sw * 8
    height = sh * 4
    out = bytearray(width * height * 4)
    for i, (r, g, b) in enumerate(pal_rgb):
        cx = i % 8
        cy = i // 8
        x0 = cx * sw
        y0 = cy * sh
        for y in range(y0, y0 + sh):
            for x in range(x0, x0 + sw):
                o = (y * width + x) * 4
                out[o + 0] = r
                out[o + 1] = g
                out[o + 2] = b
                out[o + 3] = 255 if i != 0 else 96
    write_png_rgba(path, width, height, bytes(out))


def export_cmd(data_dir: Path, out_dir: Path) -> None:
    wad_path = data_dir / "includes" / "newgunsinhand.wad"
    ptr_path = data_dir / "includes" / "newgunsinhand.ptr"
    pal_path = data_dir / "pal" / "newgunsinhand.pal"
    if not pal_path.exists():
        pal_path = data_dir / "includes" / "newgunsinhand.pal"

    if not wad_path.exists():
        raise FileNotFoundError(f"Missing WAD: {wad_path}")
    if not ptr_path.exists():
        raise FileNotFoundError(f"Missing PTR: {ptr_path}")
    if not pal_path.exists():
        raise FileNotFoundError(f"Missing PAL: {pal_path}")

    wad = wad_path.read_bytes()
    ptr = ptr_path.read_bytes()
    pal_rgb = read_palette(pal_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    for slot in range(SLOT_COUNT):
        rgba = decode_slot_to_rgba(slot, wad, ptr, pal_rgb)
        write_png_rgba(out_dir / slot_name(slot), GUN_COLS, GUN_LINES, rgba)

    shutil.copyfile(pal_path, out_dir / "newgunsinhand.pal")
    write_palette_preview(out_dir / "newgunsinhand_palette_preview.png", pal_rgb)

    manifest = {
        "format": "ab3d1_gun_frames_v1",
        "image_size": [GUN_COLS, GUN_LINES],
        "file_pattern": "gun{gun}_frame{frame}.png",
        "notes": [
            "guns 5 and 6 are unused in this game build and exported as transparent placeholders",
            "palette index 0 is transparent",
            "import quantizes colors to newgunsinhand.pal (31 visible colors + transparent)",
        ],
        "source": {
            "wad": str(wad_path),
            "ptr": str(ptr_path),
            "pal": str(pal_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[gun-frames] Exported {SLOT_COUNT} frame PNGs to: {out_dir}")
    print("[gun-frames] Files are named gun0_frame0.png .. gun7_frame3.png")


def import_cmd(input_dir: Path, out_dir: Path, pal_path: Path | None) -> None:
    if pal_path is None:
        local_pal = input_dir / "newgunsinhand.pal"
        if local_pal.exists():
            pal_path = local_pal
        else:
            raise FileNotFoundError(
                "Palette not provided. Pass --palette or place newgunsinhand.pal in the input directory."
            )
    pal_rgb = read_palette(pal_path)
    pal_bytes = pal_path.read_bytes()
    if len(pal_bytes) < 64:
        raise ValueError(f"Palette too small ({len(pal_bytes)} bytes): {pal_path}")
    pal_bytes = pal_bytes[:64]

    slot_indices: dict[int, bytearray] = {}
    for slot in range(SLOT_COUNT):
        gun = slot // 4
        png_path = input_dir / slot_name(slot)
        if gun in UNUSED_GUNS:
            if png_path.exists():
                print(f"[gun-frames] note: {png_path.name} exists but gun {gun} is unused; ignoring.")
            continue
        if not png_path.exists():
            raise FileNotFoundError(f"Missing frame PNG: {png_path}")
        w, h, rgba = read_png_rgba(png_path)
        if w != GUN_COLS or h != GUN_LINES:
            raise ValueError(
                f"Unexpected image size for {png_path}: {w}x{h}, expected {GUN_COLS}x{GUN_LINES}"
            )
        slot_indices[slot] = quantize_rgba_to_indices(rgba, pal_rgb)

    slot_for_ptr_frame: dict[int, int] = {}
    for slot, ptr_off in enumerate(GUN_PTR_FRAME_OFFSETS):
        gun = slot // 4
        if gun in UNUSED_GUNS:
            continue
        if ptr_off % GUN_STRIDE != 0:
            raise ValueError(f"Invalid ptr offset {ptr_off} for slot {slot}")
        ptr_frame = ptr_off // GUN_STRIDE
        if ptr_frame in slot_for_ptr_frame:
            raise ValueError(f"PTR frame collision at frame {ptr_frame}")
        slot_for_ptr_frame[ptr_frame] = slot

    ptr = bytearray(PTR_FRAME_COUNT * GUN_STRIDE)
    wad = bytearray()
    transparent = bytearray(GUN_COLS * GUN_LINES)

    for ptr_frame in range(PTR_FRAME_COUNT):
        slot = slot_for_ptr_frame.get(ptr_frame)
        frame_indices = slot_indices[slot] if slot is not None else transparent
        base = ptr_frame * GUN_STRIDE
        for x in range(GUN_COLS):
            wad_off = len(wad)
            p = base + x * 4
            ptr[p + 0] = 0  # mode 0 => low 5 bits
            ptr[p + 1] = (wad_off >> 16) & 0xFF
            ptr[p + 2] = (wad_off >> 8) & 0xFF
            ptr[p + 3] = wad_off & 0xFF
            for y in range(GUN_LINES):
                idx = frame_indices[y * GUN_COLS + x] & 31
                wad.extend(((idx >> 8) & 0xFF, idx & 0xFF))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "newgunsinhand.wad").write_bytes(bytes(wad))
    (out_dir / "newgunsinhand.ptr").write_bytes(bytes(ptr))
    (out_dir / "newgunsinhand.pal").write_bytes(pal_bytes)

    print(f"[gun-frames] Built replacement set in: {out_dir}")
    print("[gun-frames] Output files: newgunsinhand.wad, newgunsinhand.ptr, newgunsinhand.pal")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export/import AB3D1 gun frames as PNGs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export current gun frames to PNG files.")
    p_export.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to executable-local data directory (contains includes/ and pal/).",
    )
    p_export.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where gun PNG files will be written.",
    )

    p_import = sub.add_parser("import", help="Import edited PNGs and build replacement WAD/PTR/PAL.")
    p_import.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Directory containing edited gun PNG files (gunX_frameY.png).",
    )
    p_import.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write newgunsinhand.wad/.ptr/.pal replacement files.",
    )
    p_import.add_argument(
        "--palette",
        type=Path,
        default=None,
        help="Palette file to quantize to (defaults to in-dir/newgunsinhand.pal).",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.cmd == "export":
            export_cmd(args.data_dir.resolve(), args.out_dir.resolve())
        elif args.cmd == "import":
            import_cmd(args.in_dir.resolve(), args.out_dir.resolve(), None if args.palette is None else args.palette.resolve())
        else:
            raise ValueError(f"Unknown command: {args.cmd}")
    except Exception as exc:  # noqa: BLE001
        print(f"[gun-frames] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
