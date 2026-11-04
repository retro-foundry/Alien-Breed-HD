#!/usr/bin/env python3
"""Train native DOS compiler feedback and verify the resulting 4 MB image."""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def command_arguments(entry):
    if 'arguments' in entry:
        return list(entry['arguments'])
    if os.name != 'nt':
        return shlex.split(entry['command'])
    # CMake emits Windows command lines, including quoted -I paths with spaces.
    count = ctypes.c_int()
    parse = ctypes.windll.shell32.CommandLineToArgvW
    parse.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    parse.restype = ctypes.POINTER(ctypes.c_wchar_p)
    values = parse(entry['command'], ctypes.byref(count))
    if not values:
        raise ctypes.WinError()
    try:
        return [values[i] for i in range(count.value)]
    finally:
        release = ctypes.windll.kernel32.LocalFree
        release.argtypes = [ctypes.c_void_p]
        release.restype = ctypes.c_void_p
        release(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build/dos'))
    parser.add_argument('--dosbox', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = (root / args.build_dir).resolve()
    box = args.dosbox.resolve()
    if not box.is_file():
        parser.error('--dosbox must name an installed DOSBox Staging executable')
    entries = json.loads((build / 'compile_commands.json').read_text())
    commands = [command_arguments(entry) for entry in entries]
    if not commands or any('-DAB3D_NATIVE_RGB565' not in cmd for cmd in commands):
        parser.error('build the native DOS target first with tools/build_dos.ps1')
    if any(any(arg.startswith('-fprofile-') for arg in cmd) for cmd in commands):
        parser.error('start with the ordinary build; tools/build_dos.ps1 -ProfileGuided does this')
    base_exe = (build / 'ABHD.exe').read_bytes()
    if base_exe != (build / 'game/ABHD.EXE').read_bytes():
        raise RuntimeError('ordinary build and staged executable differ')
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text())
    djgpp = Path(deps['djgpp'])
    env = os.environ.copy()
    env['PATH'] = str(djgpp / 'bin') + os.pathsep + env['PATH']
    work = build / 'pgo'
    work.mkdir(exist_ok=True)
    logs = work / 'logs'
    logs.mkdir(exist_ok=True)

    def run(command, name, cwd=work):
        log = logs / (name + '.log')
        with log.open('w', encoding='utf-8') as output:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=output,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}\n' +
                               log.read_text(encoding='utf-8', errors='replace')[-3000:])

    def stack_and_stage(directory):
        run([str(djgpp / 'i586-pc-msdosdjgpp/bin/stubedit.exe'),
             str(directory / 'ABHD.exe'), 'minstack=128K'], 'stack-' + directory.name)
        shutil.copyfile(directory / 'ABHD.exe', directory / 'game/ABHD.EXE')

    def configure(directory, profiles):
        run(['cmake', '-S', str(root), '-B', str(directory), '-G', 'Ninja',
             f'-DCMAKE_TOOLCHAIN_FILE={root / "cmake/djgpp.cmake"}',
             f'-DDJGPP_ROOT={djgpp}', f'-DDOS_SDL_ROOT={deps["sdl"]}',
             '-DAB3D_DOS_INDEXED=OFF', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
             f'-DAB3D_DOS_PROFILE_DIR={profiles.as_posix()}'], 'configure-' + directory.name)
        run(['cmake', '--build', str(directory), '--parallel', '4'], 'build-' + directory.name)

    def check(level, frames, *, training=False, turn=False, exercise=False):
        memory = 16 if training else 4
        sound = 'on' if training or exercise else 'off'
        case = f'40000-{memory}m-320x240-l{level}-{sound}'
        case += '-exercise' if exercise else ''
        case += '-mouse' if training or exercise else ''
        case += '-turn' if turn else ''
        case += '' if exercise and not training else '-fixed'
        command = [sys.executable, str(root / 'tools/check_dos.py'), '--dosbox', str(box),
                   '--build-dir', str(work), '--memory', str(memory), '--height', '240',
                   '--cycles', '40000', '--frames', str(frames), '--level', str(level),
                   '--sound', sound]
        if training or exercise:
            command.append('--mouse')
        if turn:
            command.append('--turn')
        if exercise:
            command.append('--exercise')
        if not exercise or training:
            command.append('--fixed-ticks')
        run(command, ('train-' if training else 'check-') + case, cwd=root)
        return work / 'checks' / case

    # Keep DOS counters separate from the normal build and from user saves.
    shutil.copytree(build / 'game', work / 'game', dirs_exist_ok=True)
    shutil.copytree(build / 'generated', work / 'generated', dirs_exist_ok=True)
    shutil.copyfile(build / 'ABHD.exe', work / 'ABHD.exe')
    (work / 'ordinary.exe').write_bytes(base_exe)
    oracles = work / 'ordinary-captures'
    oracles.mkdir(exist_ok=True)
    captures = [(level, 4, False) for level in range(1, 17)]
    captures += [(level, 48, True) for level in (1, 2, 4, 6, 15, 16)]
    expected = {}
    for level, frames, turn in captures:
        case = check(level, frames, turn=turn)
        expected[level, turn] = (case / 'FRAME.BMP').read_bytes()
        (oracles / f'l{level}{"-turn" if turn else ""}.bmp').write_bytes(expected[level, turn])
    print('PGO: preserved all 22 ordinary 4 MB captures', flush=True)

    # Short object names produce unique DOS 8.3 counter files (S000.GCD).
    # Normalize the source spelling to Ninja's forward slashes: GCC includes
    # it in location checksums even when the two Windows paths name one file.
    def instrument(pair):
        i, original = pair
        command = list(original)
        command[command.index('-o') + 1] = f'S{i:03d}.o'
        source = command.index('-c') + 1
        command[source] = command[source].replace('\\', '/')
        command += ['-fprofile-generate=PROF', f'-fprofile-prefix-path={work}']
        run(command, f'instrument-{i:03d}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(instrument, enumerate(commands)))
    objects = [f'S{i:03d}.o' for i in range(len(commands))]
    run([commands[0][0], '-march=i486', '-mtune=i486', '-O3', '-flto',
         '-fwhole-program', '-Wl,--gc-sections', '-o', 'ABHD.exe', *objects,
         str(Path(deps['sdl']) / 'SDL-build/libSDL3.a'), '-lm',
         '-fprofile-generate=PROF', f'-fprofile-prefix-path={work}'], 'instrument-link')
    stack_and_stage(work)
    training = [(level, False, False) for level in range(1, 17)]
    training += [(level, True, False) for level in (1, 4, 6, 7, 14, 15)]
    training.append((4, False, True))
    counters = {}
    for level, turn, exercise in training:
        suffix = '-exercise' if exercise else ''
        suffix += '-mouse' + ('-turn' if turn else '') + '-fixed'
        prof = work / 'checks' / f'40000-16m-320x240-l{level}-on{suffix}' / 'PROF'
        prof.mkdir(parents=True, exist_ok=True)
        for i in range(len(commands)):
            name = f'S{i:03d}.GCD'
            if name in counters:
                (prof / name).write_bytes(counters[name])
            else:
                (prof / name).unlink(missing_ok=True)
        case = check(level, 32 if exercise else 128, training=True,
                     turn=turn, exercise=exercise)
        boot = (case / 'BOOT.LOG').read_text(errors='replace')
        if 'profiling:' in boot or 'libgcov' in boot:
            raise RuntimeError(f'compiler feedback was not saved cleanly: {case}')
        for i in range(len(commands)):
            name = f'S{i:03d}.GCD'
            if (prof / name).exists():
                counters[name] = (prof / name).read_bytes()
        if not counters:
            raise RuntimeError(f'no compiler feedback written: {case}')
        print(f'PGO: trained level {level}, turn={turn}, exercise={exercise}', flush=True)

    profiles = work / 'compiler-feedback'
    profiles.mkdir(exist_ok=True)
    source_options = ['# Generated by tools/profile_dos.py; preserve GCC static function IDs.']
    for i, command in enumerate(commands):
        source = Path(command[command.index('-c') + 1]).resolve().relative_to(root).as_posix()
        # An empty dump directory keeps aux_base_name equal to the training
        # object's short name. COMPILE_FLAGS retains the explicit empty arg.
        source_options.append(f'set_property(SOURCE "{source}" APPEND_STRING PROPERTY COMPILE_FLAGS '
                              f'" -dumpbase S{i:03d}.c -dumpbase-ext .c -dumpdir \\"\\"")')
        alias = profiles / f'S{i:03d}.gcda'
        data = counters.get(f'S{i:03d}.GCD')
        if data is None:
            alias.unlink(missing_ok=True)  # Translation units containing only data.
        else:
            alias.write_bytes(data)
    (profiles / 'sources.cmake').write_text('\n'.join(source_options) + '\n')
    configure(work, profiles)
    stack_and_stage(work)
    candidate = (work / 'ABHD.exe').read_bytes()
    if b'libgcov profiling error' in candidate:
        raise RuntimeError('instrumentation remains in optimized image')
    for level, frames, turn in captures:
        case = check(level, frames, turn=turn)
        if (case / 'FRAME.BMP').read_bytes() != expected[level, turn]:
            raise RuntimeError(f'compiler feedback changed pixels: level {level}, turn={turn}')
    check(4, 32, exercise=True)
    if (build / 'ABHD.exe').read_bytes() != base_exe:
        raise RuntimeError('ordinary build changed during compiler training')
    configure(build, profiles)
    run([str(djgpp / 'i586-pc-msdosdjgpp/bin/stubedit.exe'),
         str(build / 'ABHD.exe'), 'minstack=128K'], 'stack-final')
    if (build / 'ABHD.exe').read_bytes() != candidate:
        raise RuntimeError('final executable differs from the verified compiler-feedback image')
    shutil.copyfile(build / 'ABHD.exe', build / 'game/ABHD.EXE')
    report = dict(exe_sha256=hashlib.sha256(candidate).hexdigest(),
                  ordinary_exe_sha256=hashlib.sha256(base_exe).hexdigest(),
                  training_runs=len(training), training_memory_mb=16,
                  verified_captures=len(captures), verified_memory_mb=4,
                  realtime_level4_exercise=True, counter_files=len(counters),
                  capture_sha256={f'l{level}{"-turn" if turn else ""}':
                                  hashlib.sha256(data).hexdigest()
                                  for (level, turn), data in expected.items()})
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PGO: all 22 captures match; real-time 4 MB SB16/mouse exercise passed', flush=True)


if __name__ == '__main__':
    main()
