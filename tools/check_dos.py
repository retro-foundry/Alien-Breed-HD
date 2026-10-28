#!/usr/bin/env python3
"""Run a reproducible 486 DOSBox profile in an isolated DOS 8.3 game tree."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', required=True, type=Path)
    parser.add_argument('--build-dir', default='build/dos', type=Path)
    parser.add_argument('--cycles', type=int, default=20000)
    parser.add_argument('--memory', type=int, default=4)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=200)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--sound', choices=['on', 'off'], default='on')
    parser.add_argument('--frontend', action='store_true', help='Enter the map through the real title keyboard path')
    parser.add_argument('--exercise', action='store_true', help='Check title keyboard/music, then mix shotgun/flame samples')
    parser.add_argument('--sb', choices=['sb16', 'sbpro2'], default='sb16')
    parser.add_argument('--mouse', action='store_true', help='Enable the optional DOS mouse transport')
    parser.add_argument('--fixed-ticks', action='store_true', help='Deterministic pixel comparisons only; not a performance profile')
    args = parser.parse_args()
    build = args.build_dir.resolve()
    case = f'{args.cycles}-{args.memory}m-{args.width}x{args.height}-l{args.level}-{args.sound}'
    if args.frontend:
        case += '-frontend'
    if args.exercise:
        case += '-exercise'
    if args.sb != 'sb16':
        case += '-' + args.sb
    if args.mouse:
        case += '-mouse'
    if args.fixed_ticks:
        case += '-fixed'
    run = build / 'checks' / case
    run.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((build / 'generated/assets.json').read_text())
    for name in ['ABHD.EXE', 'CWSDPMI.EXE', *mapping.values()]:
        dest = run / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / 'game' / name, dest)
    ini = (build / 'generated/default.ini').read_text()
    for key, value in {'render_width': args.width, 'render_height': args.height,
                       'start_level': args.level}.items():
        ini = re.sub(rf'^{key}=.*$', f'{key}={value}', ini, flags=re.M)
    (run / 'AB3D.INI').write_text(ini, encoding='ascii')
    for name in ['ab3d.log', 'FRAME.BMP', 'MENU.BMP', 'MAP.BMP', 'HUD.BMP', 'KEYS.BMP',
                 'OPTIONS.BMP', 'INTRO1.BMP', 'INTRO2.BMP', 'PASS.TXT', 'SAVEGAME.BIN',
                 'HUD200.BMP', 'HUD240.BMP', 'BAR200.BMP', 'BAR240.BMP',
                 'AUTOS.BMP', 'LOAD.BMP', 'NEWGAME.BMP', 'PAUSE.BMP', 'BOOT.LOG',
                 'AUTOSAVE.BIN', 'AUTO1.BIN', 'AUTO2.BIN', 'AUTO3.BIN', 'AUTO4.BIN']:
        (run / name).unlink(missing_ok=True)
    config = run / 'profile.conf'
    config.write_text(f'''[sdl]
fullscreen=false
windowresolution=original
output=surface
[dosbox]
machine=svga_s3
memsize={args.memory}
[cpu]
core=normal
cputype=486_slow
cycles=fixed {args.cycles}
[sblaster]
sbtype={args.sb}
sbbase=220
irq=7
dma=1
hdma=5
[dos]
xms=true
ems=false
umb=false
[autoexec]
mount c "{run}"
c:
set BLASTER=A220 I7 D1 H5 T{6 if args.sb == 'sb16' else 4}
set AB3D_DOS_SOUND={1 if args.sound == 'on' else 0}
set AB3D_DOS_MOUSE={1 if args.mouse else 0}
CWSDPMI.EXE -s-
ABHD.EXE --dos-bench {args.frames} {'--dos-check' if args.exercise else '--dos-check-frontend' if args.frontend else ''} {'--dos-fixed-ticks' if args.fixed_ticks else ''} >BOOT.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
''', encoding='ascii')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (run / 'host.out').open('w') as out, (run / 'host.err').open('w') as err:
        process = subprocess.Popen([str(args.dosbox.resolve()), '-conf', str(config)], cwd=run,
                                   stdout=out, stderr=err, creationflags=flags)
        try:
            deadline = time.monotonic() + 240
            while process.poll() is None:
                log = (run / 'ab3d.log').read_text(errors='replace') if (run / 'ab3d.log').exists() else ''
                host = (run / 'host.err').read_text(errors='replace')
                if 'Exit blocked because program quit' in host and not (run / 'PASS.TXT').exists():
                    boot = (run / 'BOOT.LOG').read_text(errors='replace') if (run / 'BOOT.LOG').exists() else ''
                    raise RuntimeError(f'DOS could not start: {(boot or log)[-2000:].strip()}; see {run}')
                if any(word in log.lower() for word in ('fatal:', 'exiting due to signal', 'allocation failed', 'staging failed', 'state mismatch')) or time.monotonic() > deadline:
                    raise RuntimeError(f'DOS failure/timeout; see {run}')
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(f'DOSBox exited {process.returncode}; see {run}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    log = (run / 'ab3d.log').read_text()
    if not (run / 'PASS.TXT').exists() or '=== Exit (code 0) ===' not in log:
        raise SystemExit(f'DOS run failed; see {run}')
    if f'PlayTheGame: level {args.level - 1} ===' not in log or f'draws={args.frames} menu=0' not in log:
        raise SystemExit(f'DOS loaded the wrong map or skipped frames; see {run}')
    if f'present rect {args.width}x{args.height}' not in log:
        raise SystemExit(f'DOS did not present the requested resolution at 1:1; see {run}')
    if re.search(r'audio underruns during profile=[1-9]', log):
        raise SystemExit(f'Sound Blaster audio underrun; see {run}')
    if (args.frontend or args.exercise) and 'title, streamed music and keyboard New Game passed' not in log:
        raise SystemExit(f'DOS frontend traversal missing; see {run}')
    if args.exercise:
        for check in ['four keys/underlines, 9999 FPS, crosshair and map fit 320x200/240',
                      'main/options/autosaves/load/new-game/pause menus fit 320x200/240']:
            if check not in log:
                raise SystemExit(f'DOS screen-fit check missing: {check}; see {run}')
        if 'opaque grey UI images preserve exact cutout masks' not in log:
            raise SystemExit(f'DOS disabled-key rendering check failed; see {run}')
        for name in ['HUD200.BMP', 'BAR200.BMP']:
            capture = (run / name).read_bytes()
            width, height = struct.unpack_from('<ii', capture, 18)
            offset = struct.unpack_from('<I', capture, 10)[0]
            stride = (width * 3 + 3) & ~3
            if width < 320 or height < 200 or struct.unpack_from('<H', capture, 28)[0] != 24:
                raise SystemExit(f'Invalid DOS screen-fit capture: {name}')
            if any(capture[offset:offset + (height - 200) * stride]):
                raise SystemExit(f'DOS UI/map escaped the 200-line viewport: {name}; see {run}')
        for name in ['SAVEGAME.BIN', 'AUTOSAVE.BIN', 'AUTO1.BIN', 'AUTO2.BIN', 'AUTO3.BIN', 'AUTO4.BIN']:
            if (run / name).read_bytes()[:4] != b'AB3S':
                raise SystemExit(f'DOS save header check failed: {name}')
    image = (run / 'FRAME.BMP').read_bytes()
    if len(set(image[54:])) < 16:
        raise SystemExit(f'DOS capture is blank; see {run}')
    for line in log.splitlines():
        if any(tag in line for tag in ['[DOSMEM]', '[DOSPROFILE]', '[DOSPACING]', '[DOSBENCH]', '[DOS AUDIO]', '[DOSCHECK]']):
            print(line)
    print(f'PASS: {case}, {len(image)}-byte nonblank capture; logs in {run}')


if __name__ == '__main__':
    main()
