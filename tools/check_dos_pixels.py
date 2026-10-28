#!/usr/bin/env python3
"""Compare native DOS framebuffer pixels with the original 0RGB render pipeline.

The reference converter exists only in an ignored build snapshot. Production
retains only the native SDL surface path. Both variants use fixed logic ticks.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', type=Path, required=True)
    parser.add_argument('--levels', type=int, nargs='+', default=list(range(1, 17)))
    parser.add_argument('--reference-commit', default='487c1aae41ae3443b8ea118b4d14550647597c61',
                        help='Original v1.0.2 renderer commit, before DOS changes')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    oracle = root / 'build/dos-pixel-oracle'
    source = oracle / 'source'
    reference = oracle / 'reference'
    source.mkdir(parents=True, exist_ok=True)
    for name in ('src', 'cmake', 'third_party'):
        shutil.copytree(root / name, source / name, dirs_exist_ok=True)
    shutil.copy2(root / 'CMakeLists.txt', source / 'CMakeLists.txt')
    # Freeze the reference renderer before all DOS pixel/cache optimizations.
    legacy = subprocess.check_output(['git', 'show', args.reference_commit + ':src/renderer.c'], cwd=root).decode()
    legacy = legacy.replace('void renderer_init(void)', 'void renderer_init(int width, int height)')
    legacy = legacy.replace('renderer_set_sky_assets(const uint8_t *', 'renderer_set_sky_assets(uint8_t *')
    legacy = legacy.replace('allocate_buffers(RENDER_WIDTH, RENDER_HEIGHT);', 'allocate_buffers(width, height);')
    legacy = legacy.replace('    build_argb24_to_amiga12_lut();', '    /* Small-table reference: same colors, no 32 MB allocation. */')
    legacy = legacy.replace('return 262144u;', 'return level && level->num_floor_lines>0 ? (uint32_t)level->num_floor_lines*4u : 256u;')
    start = legacy.index('int renderer_automap_collect_line_segments(')
    end = legacy.index('\n/* -----------------------------------------------------------------------', start)
    section = legacy[start:end].replace('uint16_t *c12, int max_lines)', 'uint16_t *c12, int max_lines, int w, int h)')
    section = section.replace('    int w = g_renderer.width;\n    int h = g_renderer.height;\n', '')
    legacy = legacy[:start] + section + legacy[end:]
    (source / 'src/renderer.c').write_text(legacy)
    (source / 'src/renderer_3dobj.c').write_bytes(subprocess.check_output(
        ['git', 'show', args.reference_commit + ':src/renderer_3dobj.c'], cwd=root))
    cmake = source / 'cmake/dos.cmake'
    cmake.write_text(cmake.read_text().replace('AB3D_NATIVE_RGB565', ''))
    display = source / 'src/display.c'
    text = display.read_text()
    begin = text.index('static void display_dos_copy_cw(')
    end = text.index('{', begin) + 1
    depth = 1
    while depth:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    # Independently reproduce the previous full-frame 0RGB -> RGB565 pass.
    converter = '''static void display_dos_copy_cw(const uint16_t *src,int w,int h,const SDL_Rect *rect)
{
    SDL_Surface *surface=SDL_GetWindowSurface(g_window);
    if(!surface || !SDL_LockSurface(surface))exit(1);
    for(int y=0;y<h;y++) {
        uint16_t *out=(uint16_t *)((uint8_t *)surface->pixels+(y+rect->y)*surface->pitch)+rect->x;
        for(int x=0;x<w;x++) {
            unsigned c=src[y*w+x],r=(c>>8)&15,g=(c>>4)&15,b=c&15;
            out[x]=(uint16_t)((((r<<1)|(r>>3))<<11)|(((g<<2)|(g>>2))<<5)|((b<<1)|(b>>3)));
        }
    }
    SDL_UnlockSurface(surface);
}'''
    display.write_text(text[:begin] + converter + text[end:])
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    env = dict(os.environ, PATH=str(Path(deps['djgpp']) / 'bin') + os.pathsep + os.environ['PATH'])
    with (oracle / 'build.log').open('w') as log:
        commands = [
            [sys.executable, 'tools/package_dos.py', '--build-dir', str(reference)],
            ['cmake', '-S', str(source), '-B', str(reference), '-G', 'Ninja',
             '-DCMAKE_TOOLCHAIN_FILE=' + str(source / 'cmake/djgpp.cmake'),
             '-DDJGPP_ROOT=' + deps['djgpp'], '-DDOS_SDL_ROOT=' + deps['sdl'], '-DCMAKE_BUILD_TYPE=Release'],
            ['cmake', '--build', str(reference), '--parallel', '4'],
            [str(Path(deps['djgpp']) / 'i586-pc-msdosdjgpp/bin/stubedit.exe'), str(reference / 'ABHD.exe'), 'minstack=1M'],
        ]
        for command in commands:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=log, check=True)
    for name in ('ABHD.EXE', 'CWSDPMI.EXE'):
        src = reference / 'ABHD.exe' if name == 'ABHD.EXE' else root / 'build/dos/game' / name
        shutil.copy2(src, reference / 'game' / name)
    for level in args.levels:
        case = f'40000-4m-320x200-l{level}-off-fixed'
        reference_case = f'40000-16m-320x200-l{level}-off-fixed'
        with (oracle / f'level-{level}.log').open('w') as log:
            for build,memory in ((root / 'build/dos', 4), (reference, 16)):
                subprocess.run([sys.executable, 'tools/check_dos.py', '--dosbox', str(args.dosbox.resolve()),
                                '--build-dir', str(build), '--cycles', '40000', '--frames', '4',
                                '--level', str(level), '--memory', str(memory), '--sound', 'off', '--fixed-ticks'], cwd=root,
                               stdout=log, stderr=log, check=True)
        actual = (root / 'build/dos/checks' / case / 'FRAME.BMP').read_bytes()
        expected = (reference / 'checks' / reference_case / 'FRAME.BMP').read_bytes()
        if actual != expected:
            mismatches = sum(a != b for a, b in zip(actual[54:], expected[54:]))
            raise SystemExit(f'Level {level}: native pixel mismatch ({mismatches} byte components); see {oracle}')
        print(f'PASS level {level}: complete SDL capture exactly matches original color pipeline', flush=True)


if __name__ == '__main__':
    main()
