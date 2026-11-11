#!/usr/bin/env python3
"""Build and run the actual 486 raster/arithmetic kernels in a 4 MB DOSBox."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
from dos_wall import prepare_wall
from dos_sprite import prepare_sprite
from pal_to_header import rgb565_words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    run = root / 'build/dos-raster-regression'
    run.mkdir(parents=True, exist_ok=True)
    palette = struct.pack('>1024H', *((i * 37) & 4095 for i in range(1024)))
    packed = palette + struct.pack('>65536H', *range(65536))
    (run / 'WALL.BIN').write_bytes(prepare_wall(packed))
    (run / 'COLORS.BIN').write_bytes(rgb565_words(struct.pack('>4096H', *range(4096))))
    (run / 'SWORDS.BIN').write_bytes(prepare_sprite(struct.pack('>65536H', *range(65536)), b'\x01\0\0\0'))
    for words, pointers in [(b'', b''), (b'\0', b''), (b'\0\0', b'\0'),
                             (b'\0\0\0\0', b'\x01\0\0\x01')]:
        try:
            prepare_sprite(words, pointers)
        except ValueError:
            continue
        raise AssertionError('Malformed sprite transport input was accepted')
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp = Path(deps['djgpp'])
    env = dict(os.environ, PATH=str(djgpp / 'bin') + os.pathsep + os.environ['PATH'])
    # Match the game's source precedence, including amiga/data/pal overrides.
    palettes = run / 'pal'
    palettes.mkdir(exist_ok=True)
    for folder in (root / 'amiga/pal', root / 'amiga/data/pal'):
        for source in folder.rglob('*'):
            if source.is_file() and source.suffix.lower() == '.pal':
                shutil.copy2(source, palettes / source.name.lower())
    for name, options in (('sprite_palettes_source.h', []),
                          ('sprite_palettes_rgb565_data.h', ['--native-rgb565'])):
        subprocess.run([sys.executable, 'tools/pal_to_header.py', '--pal-dir', str(palettes),
                        '--output', str(run / name), *options], cwd=root, check=True)
    subprocess.run([str(djgpp / 'bin/i586-pc-msdosdjgpp-gcc.exe'), '-march=i486', '-mtune=i486',
                    '-O2', '-std=c11', '-fomit-frame-pointer', '-fno-strict-aliasing', '-DAB3D_DOS', '-DAB3D_NATIVE_RGB565',
                    '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
                    '-Isrc', '-Isrc/platform/dos', '-I' + str(run),
                    '-include', 'src/platform/dos/math_compat.h',
                    'tests/dos_raster.c', 'tests/dos_movement.c', 'src/platform/dos/raster_dos.c',
                    'src/platform/dos/objects_dos.c',
                    'src/sprite_palettes.c',
                    '-o', str(run / 'RASTER.EXE')], cwd=root, env=env, check=True)
    subprocess.run([str(djgpp / 'i586-pc-msdosdjgpp/bin/stubedit.exe'), str(run / 'RASTER.EXE'),
                    'minstack=128K'], env=env, check=True)
    shutil.copy2(Path(deps['dpmi']) / 'bin/CWSDPMI.EXE', run / 'CWSDPMI.EXE')
    (run / 'RASTER.LOG').unlink(missing_ok=True)
    (run / 'PASS.TXT').unlink(missing_ok=True)
    (run / 'RASTER.ERR').unlink(missing_ok=True)
    config = run / 'profile.conf'
    # This is a deterministic pixel/arithmetic oracle, not a timing benchmark.
    # A higher cycle budget keeps the expanded boundary cases within its timeout.
    config.write_text(f'''[sdl]
fullscreen=false
[mixer]
nosound=true
[dosbox]
machine=svga_s3
memsize=4
[cpu]
core=normal
cputype=486_slow
cycles=fixed 200000
[dos]
xms=true
ems=false
umb=false
[autoexec]
mount c "{run}"
c:
CWSDPMI.EXE -s-
RASTER.EXE >RASTER.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
''', encoding='ascii')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (run / 'host.log').open('w') as host:
        subprocess.run([str(args.dosbox.resolve()), '-conf', str(config), '--exit'], cwd=run,
                        stdout=host, stderr=host, creationflags=flags, timeout=60, check=True)
    log = (run / 'RASTER.LOG').read_text(errors='replace')
    print(log, end='')
    if (run / 'RASTER.ERR').exists():
        print((run / 'RASTER.ERR').read_text(errors='replace'), end='')
    if not (run / 'PASS.TXT').exists() or 'DOS raster regression passed' not in log:
        raise SystemExit(f'DOS raster regression failed; see {run}')


if __name__ == '__main__':
    main()
