#!/usr/bin/env python3
"""Check the actual DOS spill-geometry helpers against their original definitions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    work = root / 'build/dos-geometry-regression'
    work.mkdir(parents=True, exist_ok=True)
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp = Path(deps['djgpp'])
    env = dict(os.environ, PATH=str(djgpp / 'bin') + os.pathsep + os.environ['PATH'])
    command = [str(djgpp / 'bin/i586-pc-msdosdjgpp-gcc.exe'), '-march=i486', '-mtune=i486',
               '-O3', '-flto', '-fwhole-program', '-fomit-frame-pointer', '-fno-strict-aliasing',
               '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections', '-std=gnu11',
               '-DAB3D_DOS', '-DAB3D_NATIVE_RGB565', '-DAB3D_NO_THREADS', '-DAB3D_RELEASE',
               '-DSDL_MAIN_HANDLED',
               '-DAB3D_CW_COL_MAJOR=0', '-DAB3D_ENABLE_FLOOR_COL_FAST=1', '-DSKY_BILINEAR_FILTER=0',
               '-Isrc', '-Isrc/platform/dos', '-Ithird_party',
               '-I' + str(Path(deps['sdl']) / 'SDL/include'),
               '-include', 'src/platform/dos/math_compat.h', 'tests/dos_geometry.c',
               '-lm', '-o', str(work / 'GEOMETRY.EXE')]
    subprocess.run(command, cwd=root, env=env, check=True)
    subprocess.run([str(djgpp / 'i586-pc-msdosdjgpp/bin/stubedit.exe'),
                    str(work / 'GEOMETRY.EXE'), 'minstack=128K'], env=env, check=True)
    shutil.copyfile(Path(deps['dpmi']) / 'bin/CWSDPMI.EXE', work / 'CWSDPMI.EXE')
    (work / 'GEOMETRY.LOG').unlink(missing_ok=True)
    (work / 'PASS.TXT').unlink(missing_ok=True)
    config = work / 'geometry.conf'
    config.write_text(f'''[sdl]
fullscreen=false
[mixer]
nosound=true
[dosbox]
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
mount c "{work}"
c:
CWSDPMI.EXE -s-
GEOMETRY.EXE >GEOMETRY.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
''', encoding='ascii')
    startup = None
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    with (work / 'host.log').open('w') as output:
        subprocess.run([str(args.dosbox.resolve()), '--conf', str(config), '--exit'], cwd=work,
                        stdout=output, stderr=subprocess.STDOUT, startupinfo=startup,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                        timeout=60, check=True)
    log = (work / 'GEOMETRY.LOG').read_text(errors='replace')
    print(log, end='')
    if not (work / 'PASS.TXT').exists() or 'DOS geometry regression passed' not in log:
        raise SystemExit(f'DOS geometry regression failed; see {work}')


if __name__ == '__main__':
    main()
