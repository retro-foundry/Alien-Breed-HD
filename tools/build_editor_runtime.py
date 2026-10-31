"""Link the React editor's preview against the actual single-threaded web game.

First build the existing emscripten-release preset (README.md). This script
reuses those objects/assets, replacing only the executable entry point.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emcc', default=os.environ.get('EMCC', 'emcc'))
    parser.add_argument('--game-build', type=Path, default=root / 'build/emscripten-release')
    args = parser.parse_args()
    build = args.game_build.resolve()
    cache = (build / 'CMakeCache.txt').read_text(encoding='utf-8')
    if 'AB3D_NO_THREADS:BOOL=ON' not in cache:
        parser.error('Preview requires the single-threaded emscripten-release build.')
    subprocess.run(['cmake', '--build', str(build), '--parallel'], cwd=root, check=True)
    source_list = (root / 'cmake/game_sources.cmake').read_text(encoding='utf-8')
    match = re.search(r'set\(SOURCES\s+(.*?)\n\)', source_list, re.S)
    if match is None:
        parser.error('Cannot read the game source list in cmake/game_sources.cmake.')
    objects = [build / 'CMakeFiles/ab3d1.dir' / (source + '.o')
               for source in match.group(1).split() if source != 'src/main.c']
    for path in objects:
        if not path.is_file():
            parser.error(f'Missing game object: {path}. Build the emscripten-release preset first.')
    output = root / 'editor/public/runtime'
    output.mkdir(parents=True, exist_ok=True)
    compiler = [sys.executable, args.emcc] if args.emcc.endswith('.py') else [args.emcc]
    subprocess.run([*compiler, str(root / 'editor/runtime/preview.c'), *map(str, objects),
                    '-Isrc', '-Ithird_party', '-DAB3D_NO_THREADS=1', '-O2',
                    '-sUSE_SDL=2', '-sFULL_ES3=1', '-sMIN_WEBGL_VERSION=2', '-sMAX_WEBGL_VERSION=2',
                    '-sALLOW_MEMORY_GROWTH=1', '-sINITIAL_MEMORY=134217728', '-sSHARED_MEMORY=0',
                    '-sMODULARIZE=1', '-sEXPORT_NAME=createEditorEngine', '-sEXIT_RUNTIME=0',
                    '-sEXPORTED_RUNTIME_METHODS=["FS"]',
                    '--pre-js', str(root / 'editor/runtime/storage.js'),
                    '--preload-file', (build / 'em_preload').as_posix() + '@/',
                    '-o', str(output / 'engine.js')], cwd=root, check=True)
    shutil.copyfile(root / 'editor/runtime/frame.html', output / 'frame.html')
    print('Editor engine preview built in editor/public/runtime/')


if __name__ == '__main__':
    main()
