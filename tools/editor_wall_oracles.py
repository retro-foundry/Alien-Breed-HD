"""Compare editor wall pixels with the actual native io.c texture dump routines."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile


def function(source, name):
    match = re.search(r'^(?:static )?[^\n;{]+\b' + name + r'\([^;]*?\)\s*\n\{.*?^\}', source, re.M | re.S)
    if not match:
        raise ValueError(f'Cannot find the native {name} definition; review the oracle extraction.')
    return match.group()


def table(source, name):
    at = source.index(name + '[]')
    start = source.rfind('static const struct', 0, at)
    return source[start:source.index('\n};', at) + 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unpacker', required=True, type=Path)
    parser.add_argument('--cc', default='clang')
    parser.add_argument('--output', type=Path, default=Path('editor/tests/fixtures/original-walls.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = (root / 'src/io.c').read_text(encoding='utf-8')
    texture_table = table(source, 'wall_texture_table')
    entries = re.findall(r'\{\s*"([^"]+)"\s*,\s*(\d+)\s*\}', texture_table)
    limit = re.search(r'^#define MAX_WALL_TILES\s+\d+', (root / 'src/renderer.h').read_text(encoding='utf-8'), re.M)
    if not limit:
        raise ValueError('Cannot read native MAX_WALL_TILES.')
    declarations = limit.group() + '\n' + '''#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "renderer_pixel.h"
#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0777)
#endif
typedef struct { const uint8_t *wall_palettes[MAX_WALL_TILES], *walltiles[MAX_WALL_TILES], *floor_tile, *floor_pal; } RendererState;
RendererState g_renderer;
size_t g_wall_loaded_size[MAX_WALL_TILES];
'''
    native = '\n'.join([texture_table, table(source, 'wall_dump_dim_override')] + [function(source, name) for name in
        ('wall_texture_dims_from_size', 'amiga12_to_argb', 'write_bmp', 'io_dump_textures')])
    driver = '''
int main(int argc, char **argv) {
    if (argc != COUNT + 1) return 2;
    for (int i = 0; i < COUNT; i++) {
        FILE *file = fopen(argv[i + 1], "rb"); if (!file) return 3;
        if (fseek(file, 0, SEEK_END)) return 4;
        long size = ftell(file); if (size <= 2048 || fseek(file, 0, SEEK_SET)) return 5;
        uint8_t *bytes = malloc((size_t)size); if (!bytes) return 6;
        if (fread(bytes, 1, (size_t)size, file) != (size_t)size) return 7;
        fclose(file); g_wall_loaded_size[i] = (size_t)size;
        g_renderer.wall_palettes[i] = bytes; g_renderer.walltiles[i] = bytes + 2048;
    }
    io_dump_textures();
    for (int i = 0; i < COUNT; i++) free((void *)g_renderer.wall_palettes[i]);
    return 0;
}
'''.replace('COUNT', str(len(entries)))
    unpacker = args.unpacker.resolve()
    wall_dir = root / 'amiga/data/includes/walls'
    names = {path.name.lower(): path for path in wall_dir.iterdir()}
    fixtures = []
    with tempfile.TemporaryDirectory(prefix='ab3d-wall-oracle-') as temporary:
        work = Path(temporary)
        harness = work / 'oracle.c'; harness.write_text(declarations + native + driver, encoding='utf-8')
        executable = work / 'oracle.exe'
        subprocess.run([args.cc, '-std=c11', '-I', str(root / 'src'), str(harness), '-o', str(executable)], check=True)
        paths = []
        for index, (name, _) in enumerate(entries):
            path = work / f'{index}.wad'
            subprocess.run([str(unpacker), str(names[name.lower()]), str(path)], check=True, capture_output=True)
            paths.append(path)
        subprocess.run([str(executable), *map(str, paths)], cwd=work, check=True, capture_output=True)
        for index, (name, _) in enumerate(entries):
            bitmap = (work / 'textures' / f'wall_{index:02}.bmp').read_bytes()
            offset = struct.unpack_from('<I', bitmap, 10)[0]
            width, height = struct.unpack_from('<ii', bitmap, 18)
            if struct.unpack_from('<H', bitmap, 28)[0] != 32 or height <= 0:
                raise ValueError('The native BMP layout changed; review the oracle reader.')
            rgba = bytearray()
            for y in range(height - 1, -1, -1):
                row = bitmap[offset + y * width * 4:offset + (y + 1) * width * 4]
                for x in range(width):
                    b, g, r, a = row[x * 4:x * 4 + 4]; rgba.extend((r, g, b, a))
            fixtures.append({'id': index, 'name': name, 'width': width, 'height': height,
                'unpackedSha256': hashlib.sha256(paths[index].read_bytes()).hexdigest(), 'rgbaSha256': hashlib.sha256(rgba).hexdigest()})
    output = root / args.output
    output.write_text(json.dumps({'authority': 'src/io.c native io_dump_textures at palette level 16; src/sb_decompress.c via unpacker',
        'nativeSourceSha256': hashlib.sha256(source.encode()).hexdigest(),
        'nativePixelSourceSha256': hashlib.sha256((root / 'src/renderer_pixel.h').read_text(encoding='utf-8').encode()).hexdigest(),
        'walls': fixtures}, indent=2) + '\n', encoding='utf-8')
    print(f'Recorded {len(fixtures)} native texture sheet pixel oracles.')


if __name__ == '__main__':
    main()
