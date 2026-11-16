#!/usr/bin/env python3
"""Reject x87 instructions and float-math calls in integer DOS game units."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
from profile_dos import command_arguments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=Path('build/dos'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    build=(root/args.build_dir).resolve();work=build/'integer-check';work.mkdir(exist_ok=True)
    entries=json.loads((build/'compile_commands.json').read_text())
    names={'ai.c','objects.c','movement.c','math_tables.c','game_loop.c','input.c',
           'audio_dos.c','ui_dos.c','ui_surface_dos.c','raster_dos.c'}
    selected=[v for v in entries if Path(v['file']).name in names]
    if len(selected)!=len(names):parser.error('a complete DOS build is required')
    deps=json.loads((root/'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    env=dict(os.environ,PATH=str(Path(deps['djgpp'])/'bin')+os.pathsep+os.environ['PATH'])

    def check(entry):
        command=command_arguments(entry);flags=[];i=1
        if '-DAB3D_DOS' not in command:raise RuntimeError('not a DOS compiler command')
        while i<len(command):
            arg=command[i]
            if arg in ('-o','-c','-dumpbase','-dumpbase-ext','-dumpdir'):i+=2;continue
            if not arg.startswith(('-flto','-fprofile-','-Werror=coverage-mismatch','-Werror=missing-profile')):flags.append(arg)
            i+=1
        assembly=work/(Path(entry['file']).stem+'.s')
        result=subprocess.run([command[0],*flags,'-S',entry['file'],'-o',str(assembly)],
                              cwd=entry['directory'],env=env,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(f'{assembly.name} compilation failed:\n{result.stderr[-2500:]}')
        # Assembly mnemonics, not disassembled data bytes or float type names
        # in SDL headers. Compile without LTO so bodies remain inspectable.
        bad=[line.strip() for line in assembly.read_text().splitlines()
             if re.match(r'^\s+f[a-z0-9]+(?:\s|$)',line) or
             re.search(r'\bcall\s+_(?:atan2?|sqrtf?|powf?|sinf?|cosf?|__.*[sd]f[23])\b',line)]
        if bad:raise RuntimeError(f'{assembly.name} retains floating point: {bad[:8]}')
        return Path(entry['file']).name

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checked=sorted(pool.map(check,selected))
    (work/'report.json').write_text(json.dumps({'units':checked,'x87_instructions':0,'float_math_calls':0},indent=2)+'\n')
    print(f'PASS: {len(checked)} DOS game/input/audio/UI/raster units contain no x87 instructions or float-math calls')


if __name__=='__main__':
    main()
