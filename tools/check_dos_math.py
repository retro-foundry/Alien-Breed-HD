#!/usr/bin/env python3
"""Compare production integer maths with the original expressions on DJGPP/x87."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from dos_math import write_math_tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox',type=Path,required=True)
    parser.add_argument('--build-dir',type=Path,default=Path('build/dos-math-regression'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    work=(root/args.build_dir).resolve();work.mkdir(parents=True,exist_ok=True)
    write_math_tables(work)
    deps=json.loads((root/'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    djgpp=Path(deps['djgpp']);env=dict(os.environ,PATH=str(djgpp/'bin')+os.pathsep+os.environ['PATH'])
    shutil.copy2(Path(deps['dpmi'])/'bin/CWSDPMI.EXE',work/'CWSDPMI.EXE')
    cases=[('MATH',['tests/dos_math.c','src/math_tables.c','src/audio_spatial.c'],'DOS maths:'),
           ('TRIANGLE',['tests/dos_triangle.c'],'DOS triangles:')]
    startup=None
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    for name,sources,marker in cases:
        command=[str(djgpp/'bin/i586-pc-msdosdjgpp-gcc.exe'),'-O2','-std=gnu11','-march=i486','-mtune=i486',
                 '-DAB3D_DOS','-Isrc','-I'+str(work),*sources,'-lm','-lemu','-o',str(work/(name+'.EXE'))]
        subprocess.run(command,cwd=root,env=env,check=True)
        log=work/(name+'.LOG');log.unlink(missing_ok=True)
        config=work/(name+'.conf')
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
[dos]
xms=true
ems=false
umb=false
[autoexec]
mount c "{work}"
c:
CWSDPMI.EXE -s-
{name}.EXE >{name}.LOG
exit
''',encoding='ascii')
        with (work/(name+'-host.log')).open('w') as output:
            subprocess.run([str(args.dosbox.resolve()),'-exit','-conf',str(config)],cwd=work,
                           stdout=output,stderr=subprocess.STDOUT,startupinfo=startup,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                           timeout=180,check=True)
        result=log.read_text(errors='replace') if log.exists() else ''
        if marker not in result or 'passed' not in result or 'failure' in result or 'failed' in result:
            raise RuntimeError(f'{name} oracle failed; see {work}\n{result[-1500:]}')
        print(result.strip(),flush=True)


if __name__=='__main__':
    main()
