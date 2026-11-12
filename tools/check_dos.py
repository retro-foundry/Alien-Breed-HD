#!/usr/bin/env python3
"""Run a reproducible 486 DOSBox profile in an isolated DOS 8.3 game tree."""
import argparse
import ctypes
from ctypes import wintypes
import json
import math
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
    parser.add_argument('--height', type=int, help='Minimum video height (defaults to the packaged INI)')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--sound', choices=['on', 'off'], default='on')
    parser.add_argument('--frontend', action='store_true', help='Enter the map through the real title keyboard path')
    parser.add_argument('--exercise', action='store_true', help='Check title keyboard/music, then mix shotgun/flame samples')
    parser.add_argument('--sb', choices=['sb16', 'sbpro2'], default='sb16')
    parser.add_argument('--mouse', action='store_true', help='Enable the optional DOS mouse transport')
    parser.add_argument('--turn', action='store_true', help='Hold the real right-turn key during the benchmark')
    parser.add_argument('--visible', action='store_true', help='Show the emulator window for interactive input checks')
    parser.add_argument('--fixed-ticks', action='store_true', help='Deterministic pixel comparisons only; not a performance profile')
    parser.add_argument('--render-profile', action='store_true', help='Collect detailed renderer costs (adds measurement overhead)')
    parser.add_argument('--min-fps', type=float, help='Fail if the final real-time average is below this target')
    args = parser.parse_args()
    if args.min_fps is not None and (args.fixed_ticks or not math.isfinite(args.min_fps) or args.min_fps <= 0):
        parser.error('--min-fps requires a positive target and real-time ticks')
    build = args.build_dir.resolve()
    if args.height is None:
        packaged_ini = (build / 'generated/default.ini').read_text()
        args.height = int(re.search(r'^render_height=(\d+)$', packaged_ini, re.MULTILINE)[1])
    case = f'{args.cycles}-{args.memory}m-{args.width}x{args.height}-l{args.level}-{args.sound}'
    if args.frontend:
        case += '-frontend'
    if args.exercise:
        case += '-exercise'
    if args.sb != 'sb16':
        case += '-' + args.sb
    if args.mouse:
        case += '-mouse'
    if args.turn:
        case += '-turn'
    if args.fixed_ticks:
        case += '-fixed'
    if args.render_profile:
        case += '-rprof'
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
                 'OPTIONS.BMP', 'INTRO0.BMP', 'INTRO1.BMP', 'INTRO2.BMP', 'PASS.TXT', 'SAVEGAME.BIN',
                 'HUD200.BMP', 'HUD240.BMP', 'BAR200.BMP', 'BAR240.BMP',
                 'AUTOS.BMP', 'LOAD.BMP', 'NEWGAME.BMP', 'PAUSE.BMP', 'BOOT.LOG',
                 'AUTOSAVE.BIN', 'AUTO1.BIN', 'AUTO2.BIN', 'AUTO3.BIN', 'AUTO4.BIN']:
        (run / name).unlink(missing_ok=True)
    for capture in run.glob('BLURB??.BIN'):
        capture.unlink()
    config = run / 'profile.conf'
    config.write_text(f'''[sdl]
fullscreen=false
windowresolution=original
output=texture
[render]
glshader=none
[mouse]
mouse_capture={'onclick' if args.visible else 'nomouse'}
[mixer]
nosound=true
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
set AB3D_DOS_BENCH_TURN={1 if args.turn else 0}
set AB3D_PROFILE_RENDER={1 if args.render_profile else 0}
set AB3D_PROFILE_RENDER_EVERY_MS=1000
CWSDPMI.EXE -s-
ABHD.EXE --dos-bench {args.frames} {'--dos-check' if args.exercise else '--dos-check-frontend' if args.frontend else ''} {'--dos-fixed-ticks' if args.fixed_ticks else ''} >BOOT.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
''', encoding='ascii')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    startup = None
    if os.name == 'nt' and not args.visible:
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    with (run / 'host.out').open('w') as out, (run / 'host.err').open('w') as err:
        process = subprocess.Popen([str(args.dosbox.resolve()), '-conf', str(config)], cwd=run,
                                   stdout=out, stderr=err, creationflags=flags, startupinfo=startup)
        hide_windows = None
        if os.name == 'nt' and not args.visible:
            user32 = ctypes.WinDLL('user32', use_last_error=True)
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]

            @callback_type
            def hide_windows(window, _):
                owner = wintypes.DWORD()
                user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
                if owner.value == process.pid:
                    user32.ShowWindow(window, 0)
                return True
        try:
            # Full quickloads plus disk-backed canvas/UI oracles can exceed
            # four minutes at 40,000 cycles; this is outside the timed frames.
            deadline = time.monotonic() + (360 if args.exercise else 240)
            while process.poll() is None:
                if hide_windows is not None:
                    user32.EnumWindows(hide_windows, 0)
                log = (run / 'ab3d.log').read_text(errors='replace') if (run / 'ab3d.log').exists() else ''
                boot = (run / 'BOOT.LOG').read_text(errors='replace') if (run / 'BOOT.LOG').exists() else ''
                host = (run / 'host.err').read_text(errors='replace')
                if 'Exit blocked because program quit' in host and not (run / 'PASS.TXT').exists():
                    boot = (run / 'BOOT.LOG').read_text(errors='replace') if (run / 'BOOT.LOG').exists() else ''
                    raise RuntimeError(f'DOS could not start: {(boot or log)[-2000:].strip()}; see {run}')
                if any(word in (log + boot).lower() for word in ('fatal:', 'exiting due to signal', 'allocation failed', 'staging failed', 'state mismatch')) or time.monotonic() > deadline:
                    raise RuntimeError(f'DOS failure/timeout; see {run}')
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(f'DOSBox exited {process.returncode}; see {run}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    log = (run / 'ab3d.log').read_text()
    if args.turn and '[DOSCHECK] benchmark camera turned from ' not in log:
        raise SystemExit(f'DOS camera-turn diagnostic did not run; see {run}')
    if not (run / 'PASS.TXT').exists() or '=== Exit (code 0) ===' not in log:
        raise SystemExit(f'DOS run failed; see {run}')
    if f'PlayTheGame: level {args.level - 1} ===' not in log or f'draws={args.frames} menu=0' not in log:
        raise SystemExit(f'DOS loaded the wrong map or skipped frames; see {run}')
    native = re.search(r'\[DOS VIDEO\] render (\d+)x(\d+); native', log)
    actual_w, actual_h = (int(v) for v in native.groups()) if native else (args.width, args.height)
    if actual_w < args.width or actual_h < args.height or f'present rect {actual_w}x{actual_h}' not in log:
        raise SystemExit(f'DOS did not present its native resolution at 1:1; see {run}')
    if '[DOS VIDEO] full-surface world' in log and f'present rect {actual_w}x{actual_h} at (0,0)' not in log:
        raise SystemExit(f'DOS world is not covering the native surface; see {run}')
    if re.search(r'audio underruns during profile=[1-9]', log):
        raise SystemExit(f'Sound Blaster audio underrun; see {run}')
    if (args.frontend or args.exercise) and 'title, streamed music and keyboard New Game passed' not in log:
        raise SystemExit(f'DOS frontend traversal missing; see {run}')
    if args.exercise:
        if 'native INDEX8 SDL surface' in log:
            for check in ['indexed palette roundtrip, 255 water/tint samples and weapon cache refresh passed',
                          'indexed UI caches preserve current palette and exact cutouts']:
                if check not in log:
                    raise SystemExit(f'DOS indexed diagnostic missing: {check}; see {run}')
        for check in ['wall texture divisions match original fixed-point expression',
                      'pixel tag buffers absent (0 bytes)',
                      'scaled weapon frames: SDL pixels match original draws',
                      'level-point projection cache cases match uncached results',
                      'frame zone-order cases match linear search',
                      'four keys/underlines, 9999 FPS, crosshair and map fit 320x200/240',
                      'main/options/autosaves/load/new-game/pause menus fit 320x200/240',
                      'all 16 level texts fit VGA 80x25 on one screen; exact glyphs, single-key dismissal and graphics restoration passed']:
            if check not in log:
                raise SystemExit(f'DOS diagnostic check missing: {check}; see {run}')
        if 'opaque grey UI images preserve exact cutout masks' not in log:
            raise SystemExit(f'DOS disabled-key rendering check failed; see {run}')
        if (run / 'INTRO0.BMP').read_bytes() != (run / 'INTRO2.BMP').read_bytes():
            raise SystemExit(f'DOS briefing changed the graphics image; see {run}')
        if struct.unpack_from('<ii', (run / 'INTRO1.BMP').read_bytes(), 18) != (720, 400):
            raise SystemExit(f'DOS briefing did not capture the VGA ROM-font screen; see {run}')
        for level in range(1, 17):
            cells = (run / f'BLURB{level:02}.BIN').read_bytes()
            if len(cells) != 4000 or any(value != 7 for value in cells[1::2]):
                raise SystemExit(f'DOS level {level} briefing has invalid text memory; see {run}')
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
    if '[DOS VIDEO] full-surface world' in log and struct.unpack_from('<ii', image, 18) != (actual_w, actual_h):
        raise SystemExit(f'DOS world dimensions differ from the physical screen; see {run}')
    if len(set(image[54:])) < 16:
        raise SystemExit(f'DOS capture is blank; see {run}')
    for line in log.splitlines():
        if any(tag in line for tag in ['[DOSMEM]', '[DOSPROFILE]', '[DOSPACING]', '[DOSBENCH]', '[DOS AUDIO]', '[DOSCHECK]', '[RPROF]']):
            print(line)
    if args.min_fps is not None:
        measurements = re.findall(r'\[DOSPROFILE\] n=(\d+) fps=([\d.]+)', log)
        if not measurements or int(measurements[-1][0]) != args.frames:
            raise SystemExit(f'Missing final DOS performance result; see {run}')
        fps = float(measurements[-1][1])
        if fps < args.min_fps:
            raise SystemExit(f'DOS FPS target not met: {fps:.2f} < {args.min_fps:.2f}; see {run}')
    print(f'PASS: {case}, {len(image)}-byte nonblank capture; logs in {run}')


if __name__ == '__main__':
    main()
