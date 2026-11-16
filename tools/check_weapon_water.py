#!/usr/bin/env python3
"""Render authored underwater weapons through native threaded, SDL and GL paths."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-build',type=Path,default=Path('build'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    native=(root/args.native_build).resolve();work=root/'build/weapon-water/native';work.mkdir(parents=True,exist_ok=True)
    def run(command,name,cwd=root):
        log=work/(name+'.log')
        with log.open('w') as output:result=subprocess.run(command,cwd=cwd,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see {log}\n{log.read_text(errors="replace")[-2500:]}')
    run(['cmake','--build',str(native),'--config','Release','--target','ab3d1','--parallel','4'],'desktop-build')
    data=native/'Release/data'
    if not data.is_dir():data=native/'data'
    command=['cmake','-S',str(root/'tests/render'),'-B',str(work),f'-DAB3D_TEST_DATA={data.as_posix()}']
    cached_sdl=native/'_deps/sdl2-src'
    if cached_sdl.is_dir():command.append(f'-DFETCHCONTENT_SOURCE_DIR_SDL2={cached_sdl.as_posix()}')
    run(command,'configure')
    run(['cmake','--build',str(work),'--config','Release','--target','weapon_water_cpu','weapon_water_display','--parallel','4'],'build')
    exe=lambda name:work/'Release'/(name+'.exe') if (work/'Release').is_dir() else work/name
    for name,command in [('cpu',[str(exe('weapon_water_cpu'))]),
                         ('sdl',[str(exe('weapon_water_display')),'--software']),
                         ('gl',[str(exe('weapon_water_display')),'--gl'])]:
        run(command,name)
        text=(work/(name+'.log')).read_text(errors='replace')
        if 'PASS:' not in text:raise RuntimeError(f'Missing completion marker in {name}')
        print('\n'.join(line for line in text.splitlines() if line.startswith('PASS:')),flush=True)


if __name__=='__main__':
    main()
