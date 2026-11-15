#!/usr/bin/env python3
"""Check real DOS outcome fades pixel by pixel and run their music lifecycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

from profile_dos import command_arguments


def pixels(path):
    data = path.read_bytes()
    width, height = struct.unpack_from('<ii', data, 18)
    if width != 320 or height not in (200, 240) or struct.unpack_from('<H', data, 28)[0] != 24:
        raise ValueError(f'Expected 320x200/240 RGB capture: {path}')
    offset = struct.unpack_from('<I', data, 10)[0]
    stride = (width * 3 + 3) & ~3
    return [(data[p + 2], data[p + 1], data[p])
            for y in range(height) for x in range(width)
            for p in [offset + y * stride + x * 3]]


def check_pixels(case, indexed):
    base = pixels(case / 'WHITE0.BMP')
    palette = (case / 'PALETTE.BIN').read_bytes() if indexed else None
    if indexed:
        if len(palette) != 4608:
            raise ValueError('Invalid indexed palette capture')
        colors = struct.unpack_from('<256H', palette, 4096)
    captures = [('WHITE1.BMP', 55, (255, 255, 255)),
                ('WHITE2.BMP', 110, (255, 255, 255)),
                ('WHITE3.BMP', 220, (255, 255, 255)),
                ('REDMID.BMP', 110, (255, 0, 0)),
                ('REDEND.BMP', 220, (255, 0, 0))]
    for name, alpha, target in captures:
        actual = pixels(case / name)
        changed = 0
        for at, (source, result) in enumerate(zip(base, actual)):
            if indexed:
                rgb = [((s * (255 - alpha) + t * alpha + 127) // 255) >> 4
                       for s, t in zip(source, target)]
                c = colors[palette[(rgb[0] << 8) | (rgb[1] << 4) | rgb[2]]]
                expected = ((c >> 8) * 17, ((c >> 4) & 15) * 17, (c & 15) * 17)
                got = result
            else:
                expected = tuple(((s >> shift) * (255 - alpha) + (t >> shift) * alpha + 127) // 255
                                 for s, t, shift in zip(source, target, (3, 2, 3)))
                got = tuple(c >> shift for c, shift in zip(result, (3, 2, 3)))
            if got != expected:
                raise AssertionError(f'{case.name}/{name} pixel {at}: expected {expected}, got {got}')
            changed += source != result
        if not changed:
            raise AssertionError(f'{case.name}/{name} did not change the image')
    for first, second in [('WHITE2.BMP', 'WHITEREP.BMP'), ('WHITE3.BMP', 'WHITEEND.BMP'),
                          ('WHITE0.BMP', 'WHITECLR.BMP'), ('WHITE0.BMP', 'AFTER.BMP')]:
        if (case / first).read_bytes() != (case / second).read_bytes():
            raise AssertionError(f'{case.name}: {first} differs from {second}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=Path('build/dos'))
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--screen-sizes', type=int, choices=[40, 60, 80, 100], nargs='+', default=[100, 60])
    parser.add_argument('--cycles', type=int, default=200000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = args.build_dir.resolve()
    work = build / 'fade-checks'
    work.mkdir(parents=True, exist_ok=True)
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text())
    env = os.environ.copy()
    env['PATH'] = str(Path(deps['djgpp']) / 'bin') + os.pathsep + env['PATH']
    entries = json.loads((build / 'compile_commands.json').read_text())
    entry = next(e for e in entries if Path(e['file']).name == 'main.c')
    command = command_arguments(entry)
    # A separate harness has no matching production compiler feedback.
    command = [arg for arg in command if not arg.startswith('-fprofile-')]
    command[command.index('-c') + 1] = str(root / 'tests/dos_fades.c')
    command[command.index('-o') + 1] = str(work / 'fades.o')
    test_exe = work / 'FADES.EXE'
    with (work / 'build.log').open('w') as output:
        subprocess.run(command, cwd=entry['directory'], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        objects = [str(Path(e['directory']) / e['output']) for e in entries
                   if Path(e['file']).name not in ('main.c', 'control_loop.c')]
        subprocess.run([command[0], '-march=i486', '-mtune=i486', '-O3', '-flto', '-fwhole-program',
                        '-Wl,--gc-sections', str(work / 'fades.o'), *objects,
                        str(Path(deps['sdl']) / 'SDL-build/libSDL3.a'), '-lm', '-o', str(test_exe)],
                       env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(Path(deps['djgpp']) / 'i586-pc-msdosdjgpp/bin/stubedit.exe'),
                    str(test_exe), 'minstack=128K'], env=env, stdout=subprocess.DEVNULL, check=True)
    indexed = '-DAB3D_DOS_INDEXED' in command
    assets = json.loads((build / 'generated/assets.json').read_text()).values()
    default = (build / 'generated/default.ini').read_text()
    results = []
    scenarios = [(level, size, False) for level in args.levels for size in args.screen_sizes]
    scenarios.append((args.levels[0], args.screen_sizes[0], True))
    for level, size, death in scenarios:
        outcome = 'death' if death else 'complete'
        case = work / f'l{level}-view{size}-{outcome}'
        case.mkdir(exist_ok=True)
        for name in ['FADES.EXE', 'CWSDPMI.EXE', *assets]:
            source = test_exe if name == 'FADES.EXE' else build / 'game' / name
            dest = case / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        ini = re.sub(r'^start_level=.*$', f'start_level={level}', default, flags=re.M)
        ini = re.sub(r'^dos_screen_size=.*$', f'dos_screen_size={size}', ini, flags=re.M)
        (case / 'AB3D.INI').write_text(ini, encoding='ascii')
        (case / 'PASS.TXT').unlink(missing_ok=True)
        config = f'''[sdl]
fullscreen=false
[mixer]
nosound=true
[dosbox]
machine=svga_s3
memsize=4
[cpu]
core=normal
cputype=486_slow
cycles=fixed {args.cycles}
[sblaster]
sbtype=sb16
sbbase=220
irq=7
dma=1
hdma=5
[dos]
xms=true
ems=false
umb=false
[autoexec]
mount c "{case}"
c:
set BLASTER=A220 I7 D1 H5 T6
set AB3D_DOS_SOUND=1
set AB3D_DOS_MOUSE=1
CWSDPMI.EXE -s-
FADES.EXE --dos-bench 4 --dos-fixed-ticks{' --death' if death else ''} >BOOT.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
'''
        (case / 'profile.conf').write_text(config, encoding='ascii')
        with (case / 'host.log').open('w') as output:
            subprocess.run([str(args.dosbox.resolve()), '-exit', '-conf', str(case / 'profile.conf')], cwd=case,
                           stdout=output, stderr=subprocess.STDOUT, timeout=120, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        log = (case / 'AB3D.LOG').read_text()
        if not (case / 'PASS.TXT').exists() or '[FADECHECK]' not in log:
            raise RuntimeError(f'DOS fade/lifecycle check failed; see {case}')
        if 'audio underruns during profile=0' not in log:
            raise RuntimeError(f'DOS outcome music underrun; see {case}')
        check_pixels(case, indexed)
        results.append(dict(level=level, screen_size=size, death=death, exact_fade_pixels=True))
        print(f'PASS {case.name}: exact fades, repeated presentation, clearing and outcome lifecycle', flush=True)
    report = dict(memory_mb=4, cycles=args.cycles, indexed=indexed, cases=results,
                  production_exe_sha256=hashlib.sha256((build / 'game/ABHD.EXE').read_bytes()).hexdigest(),
                  harness_exe_sha256=hashlib.sha256(test_exe.read_bytes()).hexdigest(), audio_underruns=0)
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
