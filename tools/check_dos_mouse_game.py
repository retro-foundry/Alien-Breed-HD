#!/usr/bin/env python3
"""Check default mouse startup in the real DOS game; optional real host input."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from profile_dos import command_arguments


def physical_input(process,case):
    user32=ctypes.WinDLL('user32',use_last_error=True)
    user32.GetForegroundWindow.restype=wintypes.HWND
    user32.SetForegroundWindow.argtypes=[wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes=[wintypes.HWND]
    user32.mouse_event.argtypes=[wintypes.DWORD]*4+[ctypes.c_size_t]
    callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    user32.EnumWindows.argtypes=[callback_type,wintypes.LPARAM]

    def wait(marker):
        deadline=time.monotonic()+25
        while time.monotonic()<deadline:
            log=case/'AB3D.LOG'
            if log.exists() and marker in log.read_text(errors='replace'):return
            if process.poll() is not None:raise RuntimeError(f'Game exited waiting for {marker}; see {case}')
            time.sleep(0.02)
        raise RuntimeError(f'Game did not reach {marker}; see {case}')

    def check_focus():
        owner=wintypes.DWORD()
        user32.GetWindowThreadProcessId(user32.GetForegroundWindow(),ctypes.byref(owner))
        if owner.value!=process.pid:raise RuntimeError('Mouse check lost focus; input was not sent')

    windows=[]
    @callback_type
    def find(window,_):
        owner=wintypes.DWORD();user32.GetWindowThreadProcessId(window,ctypes.byref(owner))
        if owner.value==process.pid and user32.IsWindowVisible(window):windows.append(window)
        return True

    wait('READY: game mouse axes 0')
    user32.EnumWindows(find,0)
    if not windows:raise RuntimeError('No visible mouse-test DOSBox window')
    user32.SetForegroundWindow(windows[0]);time.sleep(0.3)
    for phase in range(2):
        wait(f'READY: game mouse axes {phase}')
        for dx,dy in [(12,8),(-12,-8)]*4:
            check_focus();user32.mouse_event(1,dx&0xffffffff,dy&0xffffffff,0,0);time.sleep(0.1)
        for button,down,up in [(0,2,4),(1,8,16)]:
            wait(f'READY: game mouse button {phase} {button}')
            check_focus();user32.mouse_event(down,0,0,0,0)
            try:time.sleep(0.03)
            finally:user32.mouse_event(up,0,0,0,0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=Path('build/dos'))
    parser.add_argument('--dosbox',type=Path,required=True)
    parser.add_argument('--physical',action='store_true')
    args=parser.parse_args()
    if args.physical and os.name!='nt':parser.error('--physical requires Windows')
    root=Path(__file__).resolve().parent.parent;build=args.build_dir.resolve()
    work=build/'mouse-game-check';work.mkdir(exist_ok=True)
    deps=json.loads((root/'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    entries=json.loads((build/'compile_commands.json').read_text())
    main_entry=next(value for value in entries if Path(value['file']).name=='main.c')
    command=command_arguments(main_entry);flags=[];at=1
    while at<len(command):
        flag=command[at]
        if flag in ('-o','-c','-dumpbase','-dumpbase-ext','-dumpdir'):at+=2;continue
        if flag!='-DNDEBUG' and not flag.startswith(('-fprofile-','-Werror=coverage-mismatch','-Werror=missing-profile')):
            flags.append(flag)
        at+=1
    env=dict(os.environ,PATH=str(Path(command[0]).parent)+os.pathsep+os.environ['PATH'])
    obj=work/'mouse.o';exe=work/'MOUSE.EXE'
    with (work/'build.log').open('w') as output:
        subprocess.run([command[0],*flags,'-c',str(root/'tests/dos_mouse_game.c'),'-o',str(obj)],
                       cwd=main_entry['directory'],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        objects=[str(Path(value['directory'])/value['output']) for value in entries if value is not main_entry]
        subprocess.run([command[0],'-march=i486','-mtune=i486','-O3','-flto','-fwhole-program','-Wl,--gc-sections',
                        str(obj),*objects,str(Path(deps['sdl'])/'SDL-build/libSDL3.a'),'-lm','-o',str(exe)],
                       env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(Path(deps['djgpp'])/'i586-pc-msdosdjgpp/bin/stubedit.exe'),str(exe),'minstack=128K'],
                   env=env,stdout=subprocess.DEVNULL,check=True)
    results=[]
    cases=[('default',True,'',False),('explicit',True,'1',False),
           ('disabled',True,'0',False),('no-driver',False,'',False)]
    if args.physical:cases.append(('physical',True,'',True))
    for name,driver,setting,physical in cases:
        case=work/name;case.mkdir(exist_ok=True)
        shutil.copyfile(exe,case/'MOUSE.EXE')
        shutil.copyfile(Path(deps['dpmi'])/'bin/CWSDPMI.EXE',case/'CWSDPMI.EXE')
        for asset in json.loads((build/'generated/assets.json').read_text()).values():
            target=case/asset;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(build/'game'/asset,target)
        ini=(build/'generated/default.ini').read_text()
        ini=re.sub(r'^start_level=.*$','start_level=1',ini,flags=re.M)
        ini=re.sub(r'^mouse_look=.*$','mouse_look=1',ini,flags=re.M)
        (case/'AB3D.INI').write_text(ini,encoding='ascii')
        for path in case.glob('*.BIN'):path.unlink()
        for filename in ['AB3D.LOG','PASS.TXT']:(case/filename).unlink(missing_ok=True)
        disabled=setting=='0' or not driver
        config=f'''[sdl]
fullscreen=false
windowresolution=960x720
output=texture
[mouse]
mouse_sensitivity=25
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
cycles=fixed 200000
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
set AB3D_DOS_MOUSE={setting}
set AB3D_TEST_MOUSE_DISABLED={'1' if disabled else ''}
set AB3D_TEST_MOUSE_LIVE={'1' if physical else ''}
CWSDPMI.EXE -s-
MOUSE.EXE --dos-bench 1024 --dos-fixed-ticks >BOOT.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
'''
        (case/'mouse.conf').write_text(config,encoding='ascii')
        startup=None
        if os.name=='nt' and not physical:
            startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        with (case/'host.log').open('w') as output:
            process=subprocess.Popen([str(args.dosbox.resolve()),'-conf',str(case/'mouse.conf'),'--exit'],cwd=case,
                stdout=output,stderr=subprocess.STDOUT,startupinfo=startup,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                if physical:physical_input(process,case)
                if process.wait(timeout=90):raise RuntimeError(f'DOSBox failed; see {case}')
            finally:
                if process.poll() is None:process.kill();process.wait()
        log=(case/'AB3D.LOG').read_text()
        if not (case/'PASS.TXT').exists() or '[MOUSEGAME] PASS real New Game/briefing/menu capture lifecycle' not in log:
            raise RuntimeError(f'Mouse startup failed; see {case}')
        if physical:
            for phase in range(2):
                if f'phase={phase} camera directions=15' not in log or f'phase={phase} real host fire/operate taps passed' not in log:
                    raise RuntimeError(f'Physical game mouse input failed; see {case}')
        results.append(dict(case=name,driver=driver,environment=setting,capture=not disabled,real_host_input=physical))
        print(f'PASS 4MB real-game mouse: {name}',flush=True)
    report=dict(engine_sha256=hashlib.sha256((build/'game/ABHD.EXE').read_bytes()).hexdigest(),memory_mb=4,cases=results)
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
