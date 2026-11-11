#!/usr/bin/env python3
"""Verify real DOS barrel scenes render identically with 4 and 16 MB RAM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from profile_dos import command_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build/dos/pgo'))
    parser.add_argument('--dosbox', type=Path, required=True)
    parser.add_argument('--levels', type=int, choices=(1, 6), nargs='+', default=[1, 6])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = (root / args.build_dir).resolve()
    box = args.dosbox.resolve()
    entries = json.loads((build / 'compile_commands.json').read_text())
    entry = next(v for v in entries if Path(v['file']).name == 'main.c')
    command = command_arguments(entry)
    if '-DAB3D_DOS' not in command or '-DAB3D_NATIVE_RGB565' not in command:
        parser.error('--build-dir must name a native RGB565 DOS build')
    work = build / 'model-check'
    (work / 'game').mkdir(parents=True, exist_ok=True)
    (work / 'generated').mkdir(exist_ok=True)
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp = Path(deps['djgpp'])
    env = dict(os.environ, PATH=str(djgpp / 'bin') + os.pathsep + os.environ['PATH'])

    def run(cmd, name, cwd=root):
        log = work / (name + '.log')
        with log.open('w', encoding='utf-8') as output:
            result = subprocess.run(cmd, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}\n' + log.read_text(errors='replace')[-3000:])

    # Replace only the entry point; retain the release's optimized engine objects.
    stripped = []
    i = 0
    while i < len(command):
        arg = command[i]
        if arg in ('-o', '-c', '-dumpbase', '-dumpbase-ext', '-dumpdir'):
            i += 2
            continue
        if not arg.startswith(('-fprofile-', '-Werror=coverage-mismatch', '-Werror=missing-profile')):
            stripped.append(arg)
        i += 1
    obj = work / 'scene.obj'
    run(stripped + ['-c', str(root / 'tests/dos_model_scene.c'), '-o', str(obj)], 'compile')
    objects = [str(Path(v['directory']) / v['output']) for v in entries if v is not entry]
    if any(not Path(p).is_file() for p in objects):
        raise RuntimeError('Build the complete DOS target before running its model oracle')
    exe = work / 'game/ABHD.EXE'
    run([command[0], '-O3', '-flto', '-fwhole-program', '-Wl,--gc-sections', str(obj),
         *objects, str(Path(deps['sdl']) / 'SDL-build/libSDL3.a'), '-lm', '-o', str(exe)], 'link')
    run([str(djgpp / 'i586-pc-msdosdjgpp/bin/stubedit.exe'), str(exe), 'minstack=128K'], 'stack')
    shutil.copyfile(exe, work / 'ABHD.exe')
    mapping = json.loads((build / 'generated/assets.json').read_text())
    for name in ['CWSDPMI.EXE', *mapping.values()]:
        target = work / 'game' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build / 'game' / name, target)
    for name in ['assets.json', 'default.ini']:
        shutil.copyfile(build / 'generated' / name, work / 'generated' / name)
    records = []
    for level in args.levels:
        captures = []
        hashes = []
        for memory in (4, 16):
            case = work / 'checks' / f'40000-{memory}m-320x240-l{level}-on-mouse-fixed'
            for frame in range(64):
                (case / f'SC{frame:02d}.RGB').unlink(missing_ok=True)
            run([sys.executable, str(root / 'tools/check_dos.py'), '--build-dir', str(work),
                 '--dosbox', str(box), '--memory', str(memory), '--cycles', '40000',
                 '--height', '240', '--frames', '64', '--level', str(level), '--sound', 'on',
                 '--mouse', '--fixed-ticks'], f'level-{level}-{memory}m')
            values = re.findall(r'^\[MODELCHECK\] frame=(\d+) hash=([0-9a-f]{8})$',
                                (case / 'AB3D.LOG').read_text(), re.M)
            if len(values) != 64 or [int(v[0]) for v in values] != list(range(64)):
                raise RuntimeError(f'Missing scene updates in {case}')
            hashes.append(values)
            pixels = [(case / f'SC{frame:02d}.RGB').read_bytes() for frame in range(64)]
            if any(len(v) != 320 * 240 * 2 for v in pixels):
                raise RuntimeError(f'Incomplete native framebuffer in {case}')
            captures.append(pixels)
        if hashes[0] != hashes[1]:
            raise RuntimeError(f'Memory capacity changed level-{level} simulation')
        for frame, (small, large) in enumerate(zip(*captures)):
            if small != large:
                raise RuntimeError(f'4 MB dropped or changed level-{level} rendering at frame {frame}')
        records.append({'level': level, 'simulation_hashes': 64, 'exact_images': 64})
        print(f'PASS: level {level}, all 64 simulation hashes and native framebuffers match at 4/16 MB', flush=True)
    report = {'engine_exe_sha256': hashlib.sha256((build / 'game/ABHD.EXE').read_bytes()).hexdigest(),
              'scene_exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'scenes': records}
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
