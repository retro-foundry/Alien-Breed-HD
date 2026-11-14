#!/usr/bin/env python3
"""Replay production instant-hit targeting on slow 486 and Pentium DOS CPUs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=Path('build/dos-shooting-tests'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    work = args.build_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp = Path(deps['djgpp'])
    env = dict(os.environ, PATH=str(djgpp / 'bin') + os.pathsep + os.environ['PATH'])
    command = [str(djgpp / 'bin/i586-pc-msdosdjgpp-gcc.exe'),
               '-march=i486', '-mtune=i486', '-O3', '-flto', '-fno-strict-aliasing',
               '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
               '-DSDL_MAIN_HANDLED', '-DAB3D_DOS', '-DAB3D_NATIVE_RGB565',
               '-Isrc', '-Ithird_party', '-Isrc/platform/dos',
               '-I' + str(Path(deps['sdl']) / 'SDL/include'),
               '-include', 'src/platform/dos/math_compat.h',
               'tests/feedback_shooting.c', 'src/math_tables.c', 'src/game_data.c',
               'src/objects.c', 'src/level.c', 'src/movement.c', 'src/visibility.c',
               'src/logging.c', '-lm', '-o', str(work / 'SHOOT.EXE')]
    build_log = work / 'build.log'
    with build_log.open('w') as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Native shooting fixture build failed; see {build_log}')
    subprocess.run([str(djgpp / 'i586-pc-msdosdjgpp/bin/stubedit.exe'),
                    str(work / 'SHOOT.EXE'), 'minstack=128K'], env=env, check=True,
                   stdout=subprocess.DEVNULL)

    profiles = [('486-slow', 'normal', '486_slow', 20000, 4),
                ('486', 'normal', '486_slow', 40000, 4),
                ('pentium', 'dynamic', 'pentium', 100000, 16)]
    for name, core, cpu, cycles, memory in profiles:
        run = work / name
        run.mkdir(exist_ok=True)
        shutil.copyfile(work / 'SHOOT.EXE', run / 'SHOOT.EXE')
        shutil.copyfile(Path(deps['dpmi']) / 'bin/CWSDPMI.EXE', run / 'CWSDPMI.EXE')
        for filename in ('PASS.TXT', 'CHECK.LOG'):
            (run / filename).unlink(missing_ok=True)
        config = run / 'check.conf'
        config.write_text(f'''[sdl]
fullscreen=false
output=texture
[render]
glshader=none
[mixer]
nosound=true
[dosbox]
memsize={memory}
[cpu]
core={core}
cputype={cpu}
cycles=fixed {cycles}
[dos]
xms=true
ems=false
umb=false
[autoexec]
mount c "{run}"
c:
CWSDPMI.EXE -s-
SHOOT.EXE >CHECK.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
''', encoding='ascii')
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        with (run / 'host.log').open('w') as log:
            subprocess.run([str(args.dosbox.resolve()), '-exit', '-conf', str(config)],
                           cwd=run, stdout=log, stderr=subprocess.STDOUT, check=True,
                           startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW,
                           timeout=60)
        output = (run / 'CHECK.LOG').read_text(errors='replace')
        if not (run / 'PASS.TXT').exists() or output.count('PASS:') != 2:
            raise RuntimeError(f'{name} shooting checks failed; see {run}')
        print(f'PASS: {name}, {cycles} cycles, {memory} MB\n{output.strip()}', flush=True)


if __name__ == '__main__':
    main()
