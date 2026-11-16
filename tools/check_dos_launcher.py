#!/usr/bin/env python3
"""Verify the real DOS batch preserves failures and switches only for status 86."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from package_dos import launcher_text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    work=root/'build/dos-launcher-regression';work.mkdir(exist_ok=True)
    deps=json.loads((root/'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp=Path(deps['djgpp']);env=dict(os.environ,PATH=str(djgpp/'bin')+os.pathsep+os.environ['PATH'])
    subprocess.run([str(djgpp/'bin/i586-pc-msdosdjgpp-gcc.exe'),'-O2','-march=i486','-mtune=i486',
        str(root/'tests/dos_video_status.c'),'-lemu','-o',str(work/'ABHD.EXE')],env=env,check=True)
    shutil.copy2(work/'ABHD.EXE',work/'ABVGA.EXE')
    shutil.copy2(Path(deps['dpmi'])/'bin/CWSDPMI.EXE',work/'CWSDPMI.EXE')
    (work/'ABHD.BAT').write_text(launcher_text(False),encoding='ascii')
    startup=None
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    for status in [0,1,85,86,87,255]:
        for name in ['VGA.OK','PASS.TXT']:(work/name).unlink(missing_ok=True)
        expected=0 if status==86 else status
        checks=f'if not errorlevel {expected} goto failed\n' if expected else ''
        if expected<255:checks+=f'if errorlevel {expected+1} goto failed\n'
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
set TESTEXIT={status}
call ABHD.BAT --test 123 --camera 17 --a 2 --b 3 --last
{checks}echo PASS>PASS.TXT
:failed
exit
''',encoding='ascii')
        with (work/f'host-{status}.log').open('w') as output:
            subprocess.run([str(args.dosbox.resolve()),'-exit','-conf',str(config)],cwd=work,
                stdout=output,stderr=subprocess.STDOUT,startupinfo=startup,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,timeout=60,check=True)
        if not (work/'PASS.TXT').exists() or (work/'VGA.OK').exists()!=(status==86):
            raise RuntimeError(f'Incorrect DOS handoff/status for {status}; see {work}')
        if status==86 and (work/'VGA.OK').read_text().splitlines()!=['--dos-vga','--test','123','--camera','17','--a','2','--b','3','--last']:
            raise RuntimeError(f'VGA handoff changed arguments; see {work}')
        print(f'PASS: native status {status}, VGA={status==86}, result={expected}',flush=True)


if __name__=='__main__':
    main()
