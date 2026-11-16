#!/usr/bin/env python3
"""Test video-mode selection and actual DOS tint primitives in all 16-bit layouts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    work=root/'build/dos-video-regression';work.mkdir(exist_ok=True)
    deps=json.loads((root/'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp=Path(deps['djgpp']);sdl=Path(deps['sdl'])
    env=dict(os.environ,PATH=str(djgpp/'bin')+os.pathsep+os.environ['PATH'])
    with (work/'build.log').open('w') as output:
        subprocess.run([str(djgpp/'bin/i586-pc-msdosdjgpp-gcc.exe'),'-O2','-flto','-fwhole-program',
            '-march=i486','-mtune=i486','-ffunction-sections','-fdata-sections','-Wl,--gc-sections',
            '-DAB3D_DOS','-DAB3D_NATIVE_RGB565','-DSDL_MAIN_HANDLED','-Isrc',
            '-I'+str(sdl/'SDL/include'),'tests/dos_video.c','src/platform/dos/ui_surface_dos.c','src/platform/dos/video_dos.c',
            str(sdl/'SDL-build/libSDL3.a'),'-lm','-lemu','-o',str(work/'VIDEO.EXE')],
            cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(djgpp/'i586-pc-msdosdjgpp/bin/stubedit.exe'),str(work/'VIDEO.EXE'),'minstack=128K'],check=True)
    shutil.copy2(Path(deps['dpmi'])/'bin/CWSDPMI.EXE',work/'CWSDPMI.EXE')
    (work/'VIDEO.LOG').unlink(missing_ok=True)
    config=work/'check.conf'
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
[autoexec]
mount c "{work}"
c:
CWSDPMI.EXE -s-
VIDEO.EXE >VIDEO.LOG
exit
''',encoding='ascii')
    startup=None
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    with (work/'host.log').open('w') as output:
        subprocess.run([str(args.dosbox.resolve()),'-exit','-conf',str(config)],cwd=work,
            stdout=output,stderr=subprocess.STDOUT,startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,timeout=120,check=True)
    log=(work/'VIDEO.LOG').read_text(errors='replace')
    if '3538944 clipped tint pixels' not in log or 'all 4096 colors preserved' not in log:
        raise RuntimeError(f'DOS video checks failed: {log}; see {work}')
    print(log.strip())


if __name__=='__main__':
    main()
