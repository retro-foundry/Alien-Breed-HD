#!/usr/bin/env python3
"""Compare direct indexed DOS sampling against the RGB565 renderer.

Compare a fixed center rectangle of the world: the native VGA and VESA modes
have different screen heights and therefore different HUD placement. Expected
colors are the RGB565 source colors mapped through the packaged level palette.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys


def bmp_world(path):
    data = path.read_bytes()
    width, height = struct.unpack_from('<ii', data, 18)
    if width != 320 or height not in (200, 240) or struct.unpack_from('<H', data, 28)[0] != 24:
        raise ValueError(f'Expected 320x200/240 24-bit capture: {path}')
    offset = struct.unpack_from('<I', data, 10)[0]
    stride = (width * 3 + 3) & ~3
    top = (height - 200) // 2
    pixels = []
    for y in range(top, top + 155):
        row = offset + (height - 1 - y) * stride + 20 * 3
        pixels.extend((data[i+2], data[i+1], data[i]) for i in range(row, row+280*3, 3))
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', type=Path, help='Run both variants first; otherwise compare existing captures')
    parser.add_argument('--rgb-build', type=Path, default=Path('build/dos'))
    parser.add_argument('--indexed-build', type=Path, default=Path('build/dos-indexed'))
    parser.add_argument('--levels', type=int, nargs='+', default=list(range(1, 17)))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    results = []
    for level in args.levels:
        if not 1 <= level <= 16:
            parser.error('Levels must be 1..16')
        if args.dosbox:
            for build in (args.rgb_build, args.indexed_build):
                subprocess.run([sys.executable, str(root/'tools/check_dos.py'), '--dosbox', str(args.dosbox),
                                '--build-dir', str(build), '--cycles', '40000', '--frames', '4',
                                '--memory', '4', '--level', str(level), '--sound', 'off', '--fixed-ticks'],
                               cwd=root, check=True)
        case = f'40000-4m-320x200-l{level}-off-fixed'
        rgb = bmp_world(root/args.rgb_build/'checks'/case/'FRAME.BMP')
        indexed = bmp_world(root/args.indexed_build/'checks'/case/'FRAME.BMP')
        palette = (root/args.indexed_build/'generated'/f'palette{level:02d}.bin').read_bytes()
        if len(palette) != 4612 or palette[:4] != b'ABP8' or 255 in palette[516:]:
            raise SystemExit(f'Invalid source-derived palette for level {level}')
        words = struct.unpack_from('<256H', palette, 4)
        colors = [((c>>8)*17, ((c>>4)&15)*17, (c&15)*17) for c in words]
        mapping = palette[516:]
        expected = [colors[mapping[((r>>4)<<8)|((g>>4)<<4)|(b>>4)]] for r,g,b in rgb]
        different = sum(a != b for a,b in zip(indexed, expected))
        if different:
            raise SystemExit(f'Level {level}: {different} world pixels differ beyond palette quantization')
        error = sum(abs(a-b) for x,y in zip(indexed,rgb) for a,b in zip(x,y))/(len(rgb)*3)
        results.append({'level':level, 'compared_world_pixels':len(rgb), 'sampling_mismatches':different,
                        'mean_rgb_error_255':round(error,3)})
        print(f'PASS level {level}: {len(rgb)} world pixels exactly match source sampling after palette mapping')
    (root/args.indexed_build/'indexed-comparison.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
