#!/usr/bin/env python3
"""Run actual 486 indexed world/sprite kernels against independent pixel oracles."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    run=root/'build/dos-indexed-raster-regression';run.mkdir(parents=True,exist_ok=True)
    deps=json.loads((root/'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    tool=Path(deps['djgpp']);headers=Path(deps['sdl'])/'SDL/include'
    if not (headers/'SDL3/SDL.h').exists():raise SystemExit(f'SDL headers missing: {headers}')
    env=dict(os.environ,PATH=str(tool/'bin')+os.pathsep+os.environ['PATH'])
    subprocess.run([str(tool/'bin/i586-pc-msdosdjgpp-gcc.exe'),'-march=i486','-mtune=i486',
        '-O2','-std=gnu11','-flto','-fwhole-program','-fomit-frame-pointer','-fno-strict-aliasing',
        '-DAB3D_DOS','-DAB3D_DOS_INDEXED','-ffunction-sections','-fdata-sections','-Wl,--gc-sections',
        '-Isrc','-Isrc/platform/dos','-I'+str(headers),'-include','src/platform/dos/math_compat.h',
        'tests/dos_raster_indexed.c','src/platform/dos/raster_dos.c','src/platform/dos/palette_dos.c',
        '-o',str(run/'INDEXED.EXE')],cwd=root,env=env,check=True)
    subprocess.run([str(tool/'i586-pc-msdosdjgpp/bin/stubedit.exe'),str(run/'INDEXED.EXE'),
                    'minstack=128K'],env=env,check=True)
    shutil.copy2(Path(deps['dpmi'])/'bin/CWSDPMI.EXE',run/'CWSDPMI.EXE')
    for name in ['INDEXED.LOG','INDEXED.ERR','PASS.TXT']:(run/name).unlink(missing_ok=True)
    config=run/'profile.conf'
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
INDEXED.EXE >INDEXED.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
''',encoding='ascii')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (run/'host.log').open('w') as output:
        subprocess.run([str(args.dosbox.resolve()),'-conf',str(config),'--exit'],cwd=run,
            stdout=output,stderr=output,creationflags=flags,timeout=60,check=True)
    log=(run/'INDEXED.LOG').read_text(errors='replace');print(log,end='')
    if (run/'INDEXED.ERR').exists():print((run/'INDEXED.ERR').read_text(errors='replace'),end='')
    if not (run/'PASS.TXT').exists() or 'DOS indexed raster regression passed' not in log:
        raise SystemExit(f'DOS indexed raster regression failed; see {run}')


if __name__=='__main__':main()
