#!/usr/bin/env python3
"""Compare 22 deterministic DOS scenes with x87 enabled and genuinely disabled."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build/dos'))
    parser.add_argument('--reference-build-dir', type=Path,
                        help='Optional unchanged source build for the hardware-FPU captures')
    parser.add_argument('--dosbox', type=Path, required=True, help='DOSBox-X executable')
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = args.build_dir.resolve()
    reference = (args.reference_build_dir or args.build_dir).resolve()
    work = build / 'no-fpu-check'
    work.mkdir(exist_ok=True)
    scenes = [(level, 4, False) for level in range(1, 17)]
    scenes += [(level, 48, True) for level in (1, 2, 4, 6, 15, 16)]

    def check_scene(scene):
        level, frames, turn = scene
        captures = []
        for software, directory in ((False, reference), (True, build)):
            case = f'200000-4m-320x200-l{level}-on-mouse'
            if turn:
                case += '-turn'
            case += '-fixed' + ('-no-fpu' if software else '')
            command = [sys.executable, str(root / 'tools/check_dos.py'),
                       '--build-dir', str(directory), '--dosbox', str(args.dosbox.resolve()),
                       '--memory', '4', '--height', '200', '--cycles', '200000',
                       '--frames', str(frames), '--level', str(level), '--sound', 'on',
                       '--mouse', '--fixed-ticks']
            if turn:
                command.append('--turn')
            if software:
                command.append('--no-fpu')
            log = work / (case + '.log')
            with log.open('w') as output:
                result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'DOS scene failed; see {log}\n{log.read_text(errors="replace")[-1500:]}')
            captures.append((directory / 'checks' / case / 'FRAME.BMP').read_bytes())
        if captures[0] != captures[1]:
            raise RuntimeError(f'Software FPU changed pixels: level {level}, turn={turn}')
        record = dict(level=level, turn=turn, frames=frames,
                      sha256=hashlib.sha256(captures[1]).hexdigest())
        print(f'PASS: level {level}, turn={turn}; exact hardware/software pixels, traps verified', flush=True)
        return record

    # Isolated emulators; these fixed-tick runs never measure performance.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(check_scene, scenes))
    report = dict(exe_sha256=hashlib.sha256((build / 'game/ABHD.EXE').read_bytes()).hexdigest(),
                  reference_exe_sha256=hashlib.sha256((reference / 'game/ABHD.EXE').read_bytes()).hexdigest(),
                  memory_mb=4, sound='sb16', mouse=True, fixed_ticks=True,
                  negative_control_without_libemu=True, scenes=records)
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: all 22 scenes match with x87 disabled; fixed ticks are correctness evidence only')


if __name__ == '__main__':
    main()
