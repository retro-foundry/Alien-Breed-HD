"""Add real barrel-scene feedback for the native DOS raster translation unit."""
import json
from pathlib import Path
import re
import shutil
import subprocess


def train_raster_scenes(root, work, commands, counters, env, run, check, sdl_library):
    main_index = next(i for i, cmd in enumerate(commands)
                      if Path(cmd[cmd.index('-c') + 1]).name == 'main.c')
    selected = {f'S{i:03d}.GCD' for i, cmd in enumerate(commands)
                if Path(cmd[cmd.index('-c') + 1]).resolve().relative_to(root).as_posix()
                == 'src/platform/dos/raster_dos.c'}
    if len(selected) != 1 or not selected <= counters.keys():
        raise RuntimeError('Native raster feedback is missing from ordinary training')
    compiler = Path(commands[main_index][0])
    dump = compiler.parent / 'i586-pc-msdosdjgpp-gcov-dump.exe'
    if not dump.is_file():
        raise RuntimeError(f'DJGPP profile inspector is missing: {dump}')

    def signature(path):
        output = subprocess.check_output([str(dump), '-l', str(path)], env=env, text=True)
        values = re.findall(r'FUNCTION ident=(\d+), lineno_checksum=0x([0-9a-f]+), '
                            r'cfg_checksum=0x([0-9a-f]+)', output)
        if not values:
            raise RuntimeError(f'No compiler function signatures in {path}')
        return values

    flags = []
    command = commands[main_index]
    i = 1
    while i < len(command):
        arg = command[i]
        if arg in ('-o', '-c', '-dumpbase', '-dumpbase-ext', '-dumpdir'):
            i += 2
            continue
        flags.append(arg)
        i += 1
    run([str(compiler), *flags, '-DAB3D_DOS_PROFILE_SCENE',
         '-fprofile-generate=PROF', f'-fprofile-prefix-path={work}',
         '-c', (root / 'tests/dos_model_scene.c').as_posix(), '-o', 'SCENE.o'],
        'scene-instrument')
    objects = [f'S{i:03d}.o' for i in range(len(commands)) if i != main_index]
    run([str(compiler), '-march=i486', '-mtune=i486', '-O3', '-flto',
         '-fwhole-program', '-Wl,--gc-sections', '-o', 'SCENE.EXE', 'SCENE.o',
         *objects, str(sdl_library), '-lm',
         '-fprofile-generate=PROF', f'-fprofile-prefix-path={work}'], 'scene-link')
    run([str(compiler.parent.parent / 'i586-pc-msdosdjgpp/bin/stubedit.exe'),
         str(work / 'SCENE.EXE'), 'minstack=128K'], 'scene-stack')

    scene = work / 'scene-training'
    shutil.copytree(work / 'game', scene / 'game', dirs_exist_ok=True)
    shutil.copytree(work / 'generated', scene / 'generated', dirs_exist_ok=True)
    shutil.copyfile(work / 'SCENE.EXE', scene / 'game/ABHD.EXE')
    shutil.copyfile(work / 'SCENE.EXE', scene / 'ABHD.exe')
    seed = scene / 'seed'
    seed.mkdir(exist_ok=True)
    rolling = dict(counters)
    signatures = {}
    for name, data in rolling.items():
        path = seed / name
        path.write_bytes(data)
        signatures[name] = signature(path)
    main_name = f'S{main_index:03d}.GCD'
    main_profile = counters[main_name]
    records = []
    # The shared fixture calls object_handle_barrel at frame 8 and advances
    # the real game loop. Its authored level-1/6 views exercise BitMapObj.
    for level, turn in ((1, False), (1, True), (6, False), (6, True)):
        suffix = '-turn' if turn else ''
        case = scene / 'checks' / f'40000-16m-320x240-l{level}-on-mouse{suffix}-fixed'
        prof = case / 'PROF'
        prof.mkdir(parents=True, exist_ok=True)
        for i in range(len(commands)):
            name = f'S{i:03d}.GCD'
            if name in rolling:
                (prof / name).write_bytes(rolling[name])
            else:
                (prof / name).unlink(missing_ok=True)
        (prof / 'SCENE.GCD').unlink(missing_ok=True)
        actual = check(level, 64, training=True, turn=turn, directory=scene,
                       group='scene-')
        if actual != case:
            raise RuntimeError(f'Unexpected scene profile directory: {actual}')
        text = (case / 'BOOT.LOG').read_text(errors='replace')
        text += (case / 'AB3D.LOG').read_text(errors='replace')
        if 'profiling:' in text.lower() or 'libgcov' in text.lower():
            raise RuntimeError(f'Scene compiler feedback was not saved cleanly: {case}')
        if (prof / main_name).read_bytes() != main_profile:
            raise RuntimeError(f'Scene changed production-main feedback: {case}')
        for name, expected in signatures.items():
            if name == main_name:
                continue
            path = prof / name
            if not path.is_file() or signature(path) != expected:
                raise RuntimeError(f'Scene engine profile signature mismatch: {path}')
            rolling[name] = path.read_bytes()
        records.append(dict(level=level, turn=turn, frames=64,
                            engine_signatures_exact=True))
        print(f'PGO: trained barrel scene level {level}, turn={turn}', flush=True)
    for name in selected:
        counters[name] = rolling[name]
    report = dict(scenes=records, selected_sources=['src/platform/dos/raster_dos.c'],
                  production_main_profile_preserved=True, training_memory_mb=16,
                  counters_are_not_performance_measurements=True)
    (scene / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
