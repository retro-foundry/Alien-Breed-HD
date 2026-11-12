#!/usr/bin/env python3
"""Check the real briefing/load/dismiss/graphics sequence at 4 MB in DOSBox."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
from profile_dos import command_arguments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=Path('build/dos'))
    parser.add_argument('--dosbox',type=Path,required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[1,4,15])
    args=parser.parse_args()
    if any(level<1 or level>16 for level in args.levels):parser.error('levels must be 1..16')
    root=Path(__file__).resolve().parent.parent
    build=args.build_dir.resolve();work=build/'briefing-check';work.mkdir(exist_ok=True)
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
    test_object=work/'briefing.o';test_exe=work/'BLURB.EXE'
    with (work/'build.log').open('w') as output:
        subprocess.run([command[0],*flags,'-c',str(root/'tests/dos_briefing_scene.c'),'-o',str(test_object)],
                       cwd=main_entry['directory'],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        objects=[str(Path(value['directory'])/value['output']) for value in entries if value is not main_entry]
        subprocess.run([command[0],'-march=i486','-mtune=i486','-O3','-flto','-fwhole-program','-Wl,--gc-sections',
                        str(test_object),*objects,str(Path(deps['sdl'])/'SDL-build/libSDL3.a'),'-lm','-o',str(test_exe)],
                       env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(Path(deps['djgpp'])/'i586-pc-msdosdjgpp/bin/stubedit.exe'),str(test_exe),'minstack=128K'],
                   env=env,stdout=subprocess.DEVNULL,check=True)
    assets=json.loads((build/'generated/assets.json').read_text()).values()
    default=(build/'generated/default.ini').read_text()

    def run(level,variant):
        case=work/f'l{level}-{variant}';case.mkdir(exist_ok=True)
        shutil.copyfile(test_exe,case/'BLURB.EXE')
        shutil.copyfile(Path(deps['dpmi'])/'bin/CWSDPMI.EXE',case/'CWSDPMI.EXE')
        for name in assets:
            target=case/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(build/'game'/name,target)
        import re
        ini=re.sub(r'^start_level=.*$',f'start_level={level}',default,flags=re.M)
        (case/'AB3D.INI').write_text(ini,encoding='ascii')
        for name in ['PASS.TXT','AB3D.LOG','LOADING.BMP','LOADED.BMP','FRAME.BMP']:(case/name).unlink(missing_ok=True)
        extra=' --without-briefing' if variant=='without' else ' --quit-in-briefing' if variant=='quit' else ''
        config=f'''[sdl]
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
set AB3D_DOS_MOUSE=1
CWSDPMI.EXE -s-
BLURB.EXE --dos-bench 4 --dos-fixed-ticks{extra} >BOOT.LOG
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
'''
        (case/'profile.conf').write_text(config,encoding='ascii')
        with (case/'host.log').open('w') as output:
            subprocess.run([str(args.dosbox.resolve()),'-conf',str(case/'profile.conf'),'--exit'],cwd=case,
                           stdout=output,stderr=subprocess.STDOUT,timeout=120,check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        log=(case/'AB3D.LOG').read_text()
        if not (case/'PASS.TXT').exists():raise RuntimeError(f'Briefing sequence failed; see {case}')
        if variant!='without':
            before=(case/'LOADING.BMP').read_bytes();after=(case/'LOADED.BMP').read_bytes()
            if before!=after or struct.unpack_from('<ii',before,18)!=(720,400):
                raise RuntimeError(f'Level loading corrupted the VGA briefing; see {case}')
        if variant=='quit':
            if 'quit restored graphics and released the loaded level' not in log:raise RuntimeError(f'Quit check missing: {case}')
            return
        if f'[BRIEFINGCHECK] level={level} draws=4 ' not in log or 'audio underruns during profile=0' not in log:
            raise RuntimeError(f'Gameplay/audio did not resume correctly; see {case}')
        return (case/'FRAME.BMP').read_bytes()

    results=[]
    for level in args.levels:
        before=run(level,'without');after=run(level,'with')
        if before!=after:raise RuntimeError(f'Briefing changed fixed-tick gameplay pixels for level {level}')
        results.append(dict(level=level,exact_gameplay_image=True,loading_preserves_text=True,audio_underruns=0))
        print(f'PASS level {level}: exact loading text and gameplay pixels, one-key dismissal, sound/mouse at 4MB',flush=True)
    run(args.levels[0],'quit')
    (work/'report.json').write_text(json.dumps(dict(memory_mb=4,scenes=results,quit_restores_graphics=True),indent=2)+'\n')
    print('PASS quit during loaded briefing restores graphics before cleanup',flush=True)


if __name__=='__main__':main()
