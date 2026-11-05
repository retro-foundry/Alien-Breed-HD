#!/usr/bin/env python3
"""Exercise the actual DOS SDL capture and game input/INI paths in 4 MB."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def exercise_host_mouse(process, case):
    """Send real host motion only while this test's DOSBox owns focus."""
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.mouse_event.argtypes = [wintypes.DWORD] * 4 + [ctypes.c_size_t]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    windows = []

    @callback_type
    def find_window(window, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
        if owner.value == process.pid and user32.IsWindowVisible(window):
            windows.append(window)
        return True

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        log = (case / 'MOUSE.LOG').read_text(errors='replace') if (case / 'MOUSE.LOG').exists() else ''
        if 'READY: native DOS driver motion' in log:
            user32.EnumWindows(find_window, 0)
            if windows:
                break
        if process.poll() is not None:
            raise RuntimeError('DOS mouse test exited before native motion was ready')
        time.sleep(0.05)
    if not windows:
        raise RuntimeError('No ready DOSBox mouse-test window')
    window = windows[0]
    user32.SetForegroundWindow(window)
    time.sleep(0.3)
    # Small individual moves must survive; the old sensitivity division made
    # these disappear in the game's integer SDL2-facing input path.
    for dx, dy in [(4, 3), (-4, -3)] * 4:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(owner))
        if owner.value != process.pid:
            raise RuntimeError('Mouse test lost focus; refusing to send input to another application')
        user32.mouse_event(1, dx & 0xffffffff, dy & 0xffffffff, 0, 0)
        time.sleep(0.1)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        log = (case / 'MOUSE.LOG').read_text(errors='replace')
        if 'READY: native DOS button gap' in log:
            break
        if process.poll() is not None:
            raise RuntimeError('DOS mouse test exited before button test')
        time.sleep(0.02)
    else:
        raise RuntimeError('DOS mouse test did not reach its button gap')
    owner = wintypes.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(owner))
    if owner.value != process.pid:
        raise RuntimeError('Mouse test lost focus; refusing to click another application')
    user32.mouse_event(2, 0, 0, 0, 0)
    time.sleep(0.03)
    user32.mouse_event(4, 0, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', required=True, type=Path)
    parser.add_argument('--physical', action='store_true', help='Also exercise real host motion in a Windows DOSBox window')
    args = parser.parse_args()
    if args.physical and os.name != 'nt':
        parser.error('--physical requires Windows')
    root = Path(__file__).resolve().parent.parent
    build = root / 'build/dos-mouse-regression'
    build.mkdir(parents=True, exist_ok=True)
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    compiler = Path(deps['djgpp']) / 'bin/i586-pc-msdosdjgpp-gcc.exe'
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ['PATH'])
    sdl = Path(deps['sdl'])
    events = (sdl / 'SDL/src/video/dos/SDL_dosevents.c').read_text()
    start = events.index('static void DOSVESA_SendButton(')
    end = events.index('\nvoid DOSVESA_PumpEvents(', start)
    mouse_header = (sdl / 'SDL/src/events/SDL_mouse_c.h').read_text()
    default_id = next(line for line in mouse_header.splitlines()
                      if line.startswith('#define SDL_DEFAULT_MOUSE_ID'))
    (build / 'dos_mouse_buttons.h').write_text(default_id + '\n' + events[start:end])
    executable = build / 'MOUSE.EXE'
    subprocess.run([
        str(compiler), '-march=i486', '-mtune=i486', '-O2', '-flto', '-std=gnu11',
        '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
        '-DAB3D_DOS', '-DAB3D_NATIVE_RGB565', '-DAB3D_NO_THREADS', '-DSDL_MAIN_HANDLED',
        '-Isrc/platform/dos', '-Isrc', '-I' + str(sdl / 'SDL/include'), '-I' + str(build),
        'tests/dos_mouse.c', 'src/logging.c', str(sdl / 'SDL-build/libSDL3.a'),
        '-lm', '-o', str(executable),
    ], cwd=root, env=env, check=True)
    subprocess.run([str(Path(deps['djgpp']) / 'i586-pc-msdosdjgpp/bin/stubedit.exe'),
                    str(executable), 'minstack=128K'], env=env, check=True)
    for driver, physical in [(True, False), (False, False)] + ([(True, True)] if args.physical else []):
        case = build / ('physical-motion' if physical else 'driver-on' if driver else 'driver-off')
        case.mkdir(exist_ok=True)
        shutil.copyfile(executable, case / 'MOUSE.EXE')
        shutil.copyfile(Path(deps['dpmi']) / 'bin/CWSDPMI.EXE', case / 'CWSDPMI.EXE')
        for name in ['MOUSE.LOG', 'PASS.TXT']:
            (case / name).unlink(missing_ok=True)
        config = case / 'mouse.conf'
        config.write_text(f'''[sdl]
fullscreen=false
windowresolution=640x480
output=texture
[mouse]
mouse_capture={'onstart' if physical else 'nomouse'}
dos_mouse_driver={str(driver).lower()}
[mixer]
nosound=true
[dosbox]
machine=svga_s3
memsize=4
[cpu]
core=normal
cputype=486_slow
cycles=fixed 40000
[dos]
xms=true
ems=false
umb=false
[autoexec]
mount c "{case}"
c:
{'set AB3D_TEST_NO_MOUSE=1' if not driver else ''}
{'set AB3D_TEST_LIVE_MOUSE=1' if physical else ''}
CWSDPMI.EXE -s-
MOUSE.EXE >MOUSE.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
''', encoding='ascii')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        startup = None
        if os.name == 'nt' and not physical:
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
        with (case / 'host.log').open('w') as host:
            process = subprocess.Popen([str(args.dosbox.resolve()), '-conf', str(config)], cwd=case,
                                       stdout=host, stderr=host, creationflags=flags,
                                       startupinfo=startup)
            try:
                if physical:
                    exercise_host_mouse(process, case)
                if process.wait(timeout=60):
                    raise RuntimeError(f'DOSBox failed; see {case}')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        log = (case / 'MOUSE.LOG').read_text(errors='replace')
        print(log, end='')
        if not (case / 'PASS.TXT').exists() or 'DOS mouse regression passed' not in log:
            raise SystemExit(f'DOS mouse regression failed; see {case}')


if __name__ == '__main__':
    main()
