#!/usr/bin/env python3
"""Read the original Duke3D 1.3d shareware FPS counter in an isolated DOSBox.

This samples stationary level starts, not a timedemo or frame-time distribution.
Supply an installed, unmodified shareware directory; no Duke assets are bundled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import time


SOURCE_HASHES = {
    'DUKE3D.EXE': '6e6f5159f21e795302221622df8be89325ede968c099126d5421dbec589ab3be',
    'DUKE3D.GRP': 'f943d0c2e2a0803a644a2107c81ea897dec87596d9dd1a6a432131ad6f5818d6',
    'GAME.CON': 'b3095352dc2303b5616d6180bffbe24ddd1a853d2105b384dab0e8535814e0e4',
    'DEFS.CON': '20d415b974191ac53ca1d453ced487408462a6c627f8dff409b20e05d5a6d839',
    'USER.CON': '1e872508f1abaa9f5e0504292e275de73a86bc57b29bce35f210c61d29b7255f',
    'DUKE.RTS': '3fbc1d9b221f2a6952825a511ed63756f8eac4f98001eb094a81f17cb0fd05e5',
}


def counter_templates(group):
    # Ken Silverman's ENGINE.C loadtables/printext256: TABLES.DAT contains
    # sine[4096], radar[1280], textfont[1024], smalltextfont[1024].
    if group[:12] != b'KenSilverman':
        raise ValueError('Invalid Duke GRP signature')
    count = struct.unpack_from('<I', group, 12)[0]
    offset = 16 + count * 16
    font = None
    for i in range(count):
        name, size = struct.unpack_from('<12sI', group, 16 + i * 16)
        if name.rstrip(b'\0').upper() == b'TABLES.DAT':
            font = group[offset + 6400:offset + 7424]
        offset += size
    if font is None or len(font) != 1024:
        raise ValueError('Missing original Build small font in TABLES.DAT')
    templates = {}
    for number in range(1, 1000):
        mask = bytearray(256)
        for column, char in enumerate(str(number)):
            for y, row in enumerate(font[ord(char) * 8:ord(char) * 8 + 8]):
                for x in range(4):
                    dest = column * 4 + x - 1
                    if dest >= 0 and row & (1 << (6 - x)):
                        mask[y * 32 + dest] = 1
        templates[bytes(mask)] = number
    return templates


def decode_samples(data, templates):
    if len(data) < 4:
        raise ValueError('FPS sampler did not write its header')
    count, stride = struct.unpack_from('<HH', data)
    if not 0 < count <= 64 or stride not in (260, 264) or len(data) != 4 + count * stride:
        raise ValueError('Incomplete FPS sampler records')
    samples = []
    for i in range(count):
        ticks = struct.unpack_from('<I', data, 4 + i * stride)[0]
        first = 4 + i * stride
        pixels = first + (6 if stride == 264 else 4)
        mask = bytes(int(p == 31) for p in data[pixels:pixels + 256])
        sample = dict(sample=i, bios_ticks=ticks, fps=templates.get(mask))
        if stride == 264:
            saved = tuple(data[first + 4:first + 6])
            restored = tuple(data[first + 262:first + 264])
            if saved != restored:
                raise ValueError('S3 sampler did not restore the interrupted bank/index')
            sample.update(bank=saved[0], crtc_index=saved[1],
                          restored_bank=restored[0], restored_crtc_index=restored[1])
        samples.append(sample)
    # Ignore startup, then take eight early samples. S3 bank selection sees
    # DNRATE sooner, so allow eight samples for its history to settle; preserve
    # the established three-sample window for the byte-identical VGA sampler.
    # These precede the scripted quit menu; later retained screen pixels cannot
    # establish that the counter was still being updated.
    first = next((i for i, value in enumerate(samples) if value['fps']), None)
    if first is None:
        raise ValueError('DNRATE counter was never observed')
    warmup = 8 if stride == 264 else 3
    selected = samples[first + warmup:first + warmup + 8]
    if len(selected) != 8 or any(s['fps'] is None for s in selected):
        raise ValueError('Eight complete gameplay FPS samples were not observed')
    return samples, selected


def verified_video_mode(host_log, width, height):
    modes = re.findall(r'DISPLAY: (VGA|VESA) (\d+)x(\d+) 256-colour graphics mode ([0-9a-fA-F]+)h', host_log)
    for backend, observed_width, observed_height, mode in modes:
        if (int(observed_width), int(observed_height)) == (width, height):
            return dict(backend=backend, width=width, height=height, bits_per_pixel=8,
                        bios_mode=int(mode, 16))
    raise RuntimeError(f'DOSBox did not confirm a {width}x{height} 256-colour graphics mode')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dosbox', type=Path, required=True)
    parser.add_argument('--game-dir', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=Path('build/duke-profile'))
    parser.add_argument('--cycles', type=int, default=40000)
    parser.add_argument('--memory', type=int, default=8)
    parser.add_argument('--level', type=int, choices=range(1, 7), default=1)
    parser.add_argument('--sound', choices=('on', 'off'), default='on')
    parser.add_argument('--height', type=int, choices=(200, 240), default=200,
                        help='320x200 VGA or 320x240 VESA; reject video fallback')
    parser.add_argument('--screen-size', type=int, choices=(0, 4, 8), default=8,
                        help='Duke view: 8=status bar, 4=full world with mini HUD, 0=full world without HUD')
    args = parser.parse_args()
    if args.cycles <= 0 or args.memory < 4:
        parser.error('Require positive fixed cycles and at least 4 MB')
    root = Path(__file__).resolve().parent.parent
    suffix = '-320x240-s3' if args.height == 240 else ''
    if args.screen_size != 8:
        suffix += f'-view{args.screen_size}'
    work = args.build_dir.resolve() / f'{args.cycles}-{args.memory}m-l{args.level}-{args.sound}{suffix}'
    work.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_HASHES.items():
        source = args.game_dir / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Expected original Duke3D 1.3d shareware file: {source}')
        shutil.copyfile(source, work / name)
    deps = json.loads((root / 'build/dos-deps/toolchain.json').read_text(encoding='utf-8-sig'))
    tool = Path(deps['djgpp']) / 'bin'
    env = dict(os.environ, PATH=str(tool) + os.pathsep + os.environ['PATH'])
    # The configured svga_s3 machine exposes CR6A. VESA world rendering may
    # leave its A000 window in bank one; restore the bank and index after each read.
    assembler_options = ['--defsym', 'SAMPLE_VESA=1', '--defsym', 'SAMPLE_S3=1'] if args.height == 240 else []
    subprocess.run([str(tool / 'i586-pc-msdosdjgpp-as.exe'), '--32', *assembler_options,
                    str(root / 'tests/dos_duke_sampler.s'), '-o', str(work / 'sample.o')],
                   env=env, check=True)
    subprocess.run([str(tool / 'i586-pc-msdosdjgpp-objcopy.exe'), '-O', 'binary',
                    '-j', '.text', str(work / 'sample.o'), str(work / 'SAMPLE.COM')],
                   env=env, check=True)
    sound = int(args.sound == 'on')
    (work / 'DUKE3D.CFG').write_text(f'''[Setup]
SetupVersion = "1.3D"
[Screen Setup]
ScreenMode = {1 if args.height == 240 else 2}
ScreenWidth = 320
ScreenHeight = {args.height}
ScreenSize = {args.screen_size}
Detail = 1
Shadows = 1
Tilt = 1
[Sound Setup]
FXDevice = {0 if sound else 13}
MusicDevice = 13
SoundToggle = {sound}
MusicToggle = 0
NumVoices = 8
NumChannels = 2
NumBits = 16
MixRate = 8000
BlasterAddress = 544
BlasterType = 6
BlasterInterrupt = 7
BlasterDma8 = 1
BlasterDma16 = 5
BlasterEmu = 0
[Controls]
ControllerType = 0
''', encoding='ascii')
    config = work / 'profile.conf'
    config.write_text(f'''[sdl]
fullscreen=false
windowresolution=original
output=texture
[render]
glshader=none
[mouse]
mouse_capture=nomouse
[mixer]
nosound=true
[dosbox]
machine=svga_s3
memsize={args.memory}
[cpu]
core=normal
cputype=486_slow
cycles=fixed {args.cycles}
[dos]
xms=true
ems=false
umb=false
[sblaster]
sbtype=sb16
sbbase=220
irq=7
dma=1
hdma=5
[autoexec]
mount c "{work}"
c:
set BLASTER=A220 I7 D1 H5 T6
SAMPLE.COM
autotype -w 12 -p 1 d n r a t e {' '.join([','] * 25)} f10 y enter enter
DUKE3D.EXE /V1 /L{args.level} /S1 {'/NS' if not sound else ''} /NM >DUKE.LOG
SAMPLE.COM D
if errorlevel 1 goto failed
echo PASS>PASS.TXT
:failed
exit
''', encoding='ascii')
    for name in ('FPS.BIN', 'DUKE.LOG', 'PASS.TXT', 'report.json'):
        (work / name).unlink(missing_ok=True)
    options = {}
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        options.update(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
    with (work / 'host.log').open('w') as out:
        process = subprocess.Popen([str(args.dosbox.resolve()), '-conf', str(config)],
                                   cwd=work, stdout=out, stderr=out, **options)
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None:
                log = work / 'DUKE.LOG'
                text = log.read_text(errors='replace') if log.exists() else ''
                if 'not enough memory' in text.lower() or 'not responding' in text.lower():
                    raise RuntimeError(f'Duke startup failed: {text[-1000:].strip()}; see {work}')
                if time.monotonic() > deadline:
                    raise RuntimeError(f'Duke did not finish; inspect {work / "host.log"}')
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f'DOSBox exited {process.returncode}; inspect {work / "host.log"}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if not (work / 'PASS.TXT').exists():
        raise RuntimeError(f'Duke/sampler failed; inspect {work / "DUKE.LOG"}')
    # Duke writes the fallback VGA settings back to its CFG on a normal exit.
    observed = {}
    saved_cfg = (work / 'DUKE3D.CFG').read_text()
    for key in ('ScreenMode', 'ScreenWidth', 'ScreenHeight', 'ScreenSize'):
        match = re.search(rf'^{key}\s*=\s*(\d+)', saved_cfg, re.MULTILINE)
        if not match:
            raise RuntimeError(f'Missing saved video setting {key}; see {work}')
        observed[key] = int(match[1])
    expected_video = dict(ScreenMode=1 if args.height == 240 else 2,
                          ScreenWidth=320, ScreenHeight=args.height,
                          ScreenSize=args.screen_size)
    if observed != expected_video or 'not found/supported' in (work / 'DUKE.LOG').read_text(errors='replace'):
        raise RuntimeError(f'Duke video fallback: requested {expected_video}, saved {observed}; see {work}')
    video_mode = verified_video_mode((work / 'host.log').read_text(errors='replace'), 320, args.height)
    samples, selected = decode_samples((work / 'FPS.BIN').read_bytes(),
                                       counter_templates((work / 'DUKE3D.GRP').read_bytes()))
    values = [s['fps'] for s in selected]
    report = dict(source_sha256=SOURCE_HASHES, cycles=args.cycles, memory_mb=args.memory,
                  level=args.level, sound=args.sound, music=False, width=320, height=args.height,
                  saved_video=observed,
                  observed_video=video_mode,
                  detail='full', screen_size=args.screen_size,
                  sampler_sha256=hashlib.sha256((work / 'SAMPLE.COM').read_bytes()).hexdigest(),
                  samples=samples, selected=selected, median_fps=statistics.median(values),
                  min_fps=min(values), max_fps=max(values),
                  scope='Original integer DNRATE counter at stationary level start; not a timedemo or frame-time distribution')
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Duke E1L{args.level}: median {report["median_fps"]:g} FPS, '
          f'{min(values)}..{max(values)}; eight counter samples; {work}')


if __name__ == '__main__':
    main()
