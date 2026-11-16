#!/usr/bin/env python3
"""Replay underwater weapons with prepared native/indexed DOS assets at 4 MB."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from profile_dos import command_arguments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--dosbox',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    build=(root/args.build_dir).resolve()
    work=build/'weapon-water-check'
    (work/'game').mkdir(parents=True,exist_ok=True)
    (work/'generated').mkdir(exist_ok=True)
    deps=json.loads((root/'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp=Path(deps['djgpp'])
    env=dict(os.environ,PATH=str(djgpp/'bin')+os.pathsep+os.environ['PATH'])
    entries=json.loads((build/'compile_commands.json').read_text())
    entry=next(v for v in entries if Path(v['file']).name=='main.c')
    command=command_arguments(entry)
    indexed='-DAB3D_DOS_INDEXED' in command

    def run(cmd,name,cwd=root):
        log=work/(name+'.log')
        with log.open('w') as output:
            result=subprocess.run(cmd,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see {log}\n{log.read_text(errors="replace")[-2200:]}')

    flags=[]
    i=1
    while i<len(command):
        arg=command[i]
        if arg in ('-o','-c','-dumpbase','-dumpbase-ext','-dumpdir'):
            i+=2;continue
        if not arg.startswith(('-fprofile-','-Werror=coverage-mismatch','-Werror=missing-profile')):flags.append(arg)
        i+=1
    obj=work/'scene.obj'
    run([command[0],*flags,'-c',str(root/'tests/dos_weapon_water_scene.c'),'-o',str(obj)],'compile')
    objects=[str(Path(v['directory'])/v['output']) for v in entries
             if Path(v['file']).name not in ('main.c','renderer.c')]
    exe=work/'game/ABHD.EXE'
    run([command[0],'-O3','-flto','-fwhole-program','-Wl,--gc-sections',str(obj),*objects,
         str(Path(deps['sdl'])/'SDL-build/libSDL3.a'),'-lm','-lemu','-o',str(exe)],'link')
    run([str(djgpp/'i586-pc-msdosdjgpp/bin/stubedit.exe'),str(exe),'minstack=128K'],'stack')
    shutil.copy2(exe,work/'ABHD.exe')
    mapping=json.loads((build/'generated/assets.json').read_text())
    for name in ['CWSDPMI.EXE',*mapping.values()]:
        target=work/'game'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(build/'game'/name,target)
    for name in ['assets.json','default.ini']:shutil.copy2(build/'generated'/name,work/'generated'/name)
    run([sys.executable,str(root/'tools/check_dos.py'),'--build-dir',str(work),
         '--dosbox',str(args.dosbox.resolve()),'--cycles','200000','--memory','4','--height','200',
         '--frames','64','--sound','off','--fixed-ticks',*(['--video','vga'] if indexed else [])],'dos')
    case=work/'checks'/('200000-4m-320x200-l1-off-fixed'+('-vga' if indexed else ''))
    lines=re.findall(r'^\[WEAPONWATER\] frame=(\d+) samples=(\d+) height=(-?\d+) passed$',
                     (case/'AB3D.LOG').read_text(),re.M)
    if len(lines)!=64 or [int(v[0]) for v in lines]!=list(range(64)):
        raise RuntimeError(f'Missing weapon/water checks; see {case}')
    print(f'PASS: DOS {"indexed VGA" if indexed else "native RGB565"}, 4 MB, 64 weapon/water cases, '
          f'{sum(int(v[1]) for v in lines)} source-color samples; both players and all six guns')


if __name__=='__main__':
    main()
