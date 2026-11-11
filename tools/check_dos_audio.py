#!/usr/bin/env python3
"""Check the actual DOS mixer against independent scalar PCM equations."""
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
    run = root / 'build/dos-audio-regression'
    run.mkdir(parents=True, exist_ok=True)
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    sdk = Path(deps['djgpp'])
    env = dict(os.environ, PATH=str(sdk / 'bin') + os.pathsep + os.environ['PATH'])
    subprocess.run([str(sdk / 'bin/i586-pc-msdosdjgpp-gcc.exe'), '-march=i486', '-mtune=i486',
                    '-O2', '-flto', '-fwhole-program', '-std=gnu11', '-UNDEBUG', '-fomit-frame-pointer',
                    '-fno-strict-aliasing', '-DAB3D_DOS', '-DAB3D_NATIVE_RGB565', '-DSDL_MAIN_HANDLED',
                    '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
                    '-Isrc/platform/dos', '-Isrc', '-I' + str(Path(deps['sdl']) / 'SDL/include'),
                    '-include', 'src/platform/dos/math_compat.h', 'tests/dos_audio.c',
                    str(Path(deps['sdl']) / 'SDL-build/libSDL3.a'), '-lm', '-o', str(run / 'AUDIO.EXE')],
                   cwd=root, env=env, check=True)
    subprocess.run([str(sdk / 'i586-pc-msdosdjgpp/bin/stubedit.exe'), str(run / 'AUDIO.EXE'),
                    'minstack=128K'], env=env, check=True)
    shutil.copy2(Path(deps['dpmi']) / 'bin/CWSDPMI.EXE', run / 'CWSDPMI.EXE')
    for name in ['AUDIO.LOG', 'AUDIO.ERR', 'PASS.TXT']:
        (run / name).unlink(missing_ok=True)
    # Pure PCM/state oracle: high fixed cycles shorten the exhaustive checks.
    # Actual Sound Blaster transport and pacing are checked in game separately.
    config = run / 'check.conf'
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
AUDIO.EXE >AUDIO.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
''', encoding='ascii')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (run / 'host.log').open('w') as output:
        subprocess.run([str(args.dosbox.resolve()), '-conf', str(config), '--exit'], cwd=run,
                       stdout=output, stderr=subprocess.STDOUT, creationflags=flags, timeout=60, check=True)
    log = (run / 'AUDIO.LOG').read_text(errors='replace')
    print(log, end='')
    if (run / 'AUDIO.ERR').exists():
        print((run / 'AUDIO.ERR').read_text(errors='replace'), end='')
    if not (run / 'PASS.TXT').exists() or 'DOS audio regression passed' not in log:
        raise SystemExit(f'DOS audio regression failed; see {run}')


if __name__ == '__main__':
    main()
