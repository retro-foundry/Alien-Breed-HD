#!/usr/bin/env python3
"""Stage source-backed assets under deterministic DOS 8.3 names and emit the map."""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', default='build/dos')
    parser.add_argument('--archive', action='store_true', help='Create the runnable DOS zip after compilation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = (root / args.build_dir).resolve()
    game = build / 'game'
    data = game / 'DATA'
    generated = build / 'generated'
    data.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    sources = {}
    for source, logical in (
        ('amiga/vectorobjects', 'data/vectorobjects'),
        ('amiga/texturemaps', 'data/texturemaps'),
        ('amiga/pal', 'data/pal'),
        ('amiga/data/pal', 'data/pal'),
        ('amiga/data/gfx', 'data/gfx'),
        ('amiga/data/helper', 'data/helper'),
        ('amiga/data/includes', 'data/includes'),
        ('amiga/data/levels', 'data/levels'),
        ('fonts', 'fonts'),
    ):
        folder = root / source
        for file in folder.rglob('*'):
            # Some original binary palettes deliberately have a .s suffix.
            # Match the native staging tree rather than classifying by suffix.
            if file.is_file():
                key = (logical + '/' + file.relative_to(folder).as_posix()).lower()
                sources[key] = file
    for file in (root / 'amiga/sounds').iterdir():
        if file.is_file():
            sources['data/sounds/' + file.name.lower()] = file
    # Native io_load_floor parses this assembly palette at startup. Verify its
    # supplied binary is byte-identical, then use that binary on slow DOS CPUs.
    pal = root / 'amiga/data/pal/FloorPalScaled'
    assembly = pal.with_suffix('.s').read_text()
    parsed = bytes(int(x, 16) for pair in re.findall(
        r'dc\.b\s+\$([0-9a-fA-F]+),\$([0-9a-fA-F]+)', assembly) for x in pair)
    if parsed != pal.read_bytes():
        raise SystemExit('FloorPalScaled binary no longer matches the original assembly')
    sources['data/includes/floorpalscaled'] = pal
    # Keep the same generated sprite palettes as the desktop build, including
    # amiga/data/pal overriding amiga/pal. Write the source header only on change.
    palette_dir = generated / 'pal'
    palette_dir.mkdir(exist_ok=True)
    palettes = {Path(key).name: source for key, source in sources.items()
                if key.startswith('data/pal/') and key.endswith('.pal')}
    for old in palette_dir.glob('*.pal'):
        if old.name not in palettes:
            old.unlink()
    for name, source in palettes.items():
        shutil.copy2(source, palette_dir / name)
    palette_header = generated / 'sprite_palettes_data.h'
    subprocess.run([sys.executable, str(root / 'tools/pal_to_header.py'),
                    '--pal-dir', str(palette_dir), '--output', str(palette_header)], check=True)
    source_header = root / 'src/sprite_palettes_data.h'
    if not source_header.exists() or source_header.read_bytes() != palette_header.read_bytes():
        shutil.copy2(palette_header, source_header)
    unpacker=root/'build/dos-deps/asset-tools/bin/sb_unpack.exe'
    if not unpacker.exists():raise SystemExit('Run tools/setup_dos.ps1 to build the host level unpacker')
    for index,(logical,source) in enumerate(sorted(sources.items())):
        if not logical.startswith('data/levels/') or source.read_bytes()[:4]!=b'=SB=':
            continue
        unpacked=generated/f'level{index:06d}.bin'
        if not unpacked.exists() or unpacked.stat().st_mtime<max(source.stat().st_mtime,unpacker.stat().st_mtime):
            subprocess.run([str(unpacker),str(source),str(unpacked)],check=True)
        expected=int.from_bytes(source.read_bytes()[4:8],'big')
        if unpacked.stat().st_size!=expected:raise SystemExit(f'Incomplete unpacked level: {source}')
        sources[logical]=unpacked
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('ffmpeg is required to render DOS music at the native 8007 Hz mixer rate')
    # Preserve the authored PNG pixels/masks, but decode into native UI surfaces
    # offline: the DOS executable needs neither a PNG decoder nor RGBA copies.
    for index,(logical,source) in enumerate(sorted(sources.items())):
        if not logical.startswith('fonts/') or source.suffix.lower()!='.png':
            continue
        native=generated/f'ui{index:06d}.bin'
        if not native.exists() or native.stat().st_mtime<max(source.stat().st_mtime,Path(__file__).stat().st_mtime):
            png=source.read_bytes()
            if png[:8]!=b'\x89PNG\r\n\x1a\n':raise SystemExit(f'Invalid UI PNG: {source}')
            width,height=struct.unpack_from('>II',png,16)
            rgba=subprocess.check_output([ffmpeg,'-hide_banner','-loglevel','error','-i',str(source),
                                          '-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'])
            if len(rgba)!=width*height*4:raise SystemExit(f'UI decode size mismatch: {source}')
            if logical=='fonts/ascii_font/ascii_font_green_black_atlas_v2.png':
                if (width,height)!=(192,102):raise SystemExit('ASCII source atlas dimensions changed')
                # Same 8x13 authored glyph samples as display.c, without the
                # unused 2-pixel margins of each 12x17 source cell.
                packed=bytearray(128*78*4)
                for cell_y in range(6):
                    for cell_x in range(16):
                        for row in range(13):
                            start=((cell_y*17+2+row)*width+cell_x*12+2)*4
                            dest=((cell_y*13+row)*128+cell_x*8)*4
                            packed[dest:dest+32]=rgba[start:start+32]
                rgba=bytes(packed);width,height=128,78

            colors=[((r>>3)<<11)|((g>>2)<<5)|(b>>3) for r,g,b in zip(rgba[0::4],rgba[1::4],rgba[2::4])]
            used={color for color,alpha in zip(colors,rgba[3::4]) if alpha}
            masked=0 in rgba[3::4]
            key=next((i for i in range(1,65536) if i not in used),None) if masked else 0
            if key is None:raise SystemExit(f'No unused UI mask color: {source}')
            pixels=[color if alpha else key for color,alpha in zip(colors,rgba[3::4])]
            native.write_bytes(struct.pack('<4sHHHH',b'ABUI',width,height,int(masked),key)+struct.pack('<'+'H'*len(pixels),*pixels))
        sources[logical]=native
    for kind in ('mt', 'med'):
        for module in (root / 'amiga/sounds' / kind).iterdir():
            if module.suffix.lower() not in ('.mt', '.med'):
                continue
            wav = generated / (kind + '_' + module.stem.lower() + '.wav')
            if not wav.exists() or wav.stat().st_mtime < module.stat().st_mtime:
                subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                                '-i', str(module), '-ac', '1', '-ar', '8007',
                                '-c:a', 'pcm_u8', str(wav)], check=True)
            sources['data/sounds/' + kind + '/' + module.stem.lower() + '.wav'] = wav
    mapping = {}
    for index, (logical, source) in enumerate(sorted(sources.items())):
        alias = f'DATA/A{index:06d}.BIN'
        shutil.copy2(source, game / alias)
        mapping[logical] = alias
    header = ['/* Generated by tools/package_dos.py; original asset formats plus offline UI/music derivatives. */',
              'static const struct { const char *logical, *short_path; } dos_assets[] = {']
    for key, value in mapping.items():
        header.append('    {' + json.dumps(key) + ',' + json.dumps(value) + '},')
    header.append('};')
    (generated / 'assets_dos.h').write_text('\n'.join(header) + '\n')
    (generated / 'assets.json').write_text(json.dumps(mapping, indent=2) + '\n')
    ini = '''# DOS / 486: true 320x200 rendering, RGB565 VESA output, single thread.
start_level=0
infinite_health=0
infinite_ammo=0
all_weapons=0
all_keys=0
flamethrower_weapon=1
marine_hitscan_projectiles=0
mouse_look=0
render_width=320
render_height=200
supersampling=1
render_threads=0
display_mode=windowed
weapon_post_gl=0
volume=100
run_speed=100
hud_bars=1
quicksave_load=1
'''
    if not (game / 'AB3D.INI').exists():
        (game / 'AB3D.INI').write_text(ini, encoding='ascii')
    (generated / 'default.ini').write_text(ini, encoding='ascii')
    (game / 'ABHD.BAT').write_text('@ECHO OFF\nIF "%BLASTER%"=="" SET BLASTER=A220 I7 D1 H5 T6\nCWSDPMI.EXE -s-\nABHD.EXE %1 %2 %3 %4 %5 %6 %7 %8 %9\n', encoding='ascii')
    (game / 'README.TXT').write_bytes((root / 'docs/dos-readme.txt').read_text().replace('\n', '\r\n').encode('ascii'))
    print(f'Staged {len(mapping)} source assets under DOS 8.3 names in {game}')
    if args.archive:
        names = ['ABHD.EXE', 'ABHD.BAT', 'AB3D.INI', 'CWSDPMI.EXE', 'CWSDPMI.DOC',
                 'SDL-LIC.TXT', 'README.TXT', *mapping.values()]
        archive = build / 'ABHD-DOS.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
            for name in names:
                if any(not re.fullmatch(r'[A-Z0-9_-]{1,8}(\.[A-Z0-9_-]{1,3})?', part)
                       for part in name.split('/')):
                    raise SystemExit(f'Not a DOS 8.3 filename: {name}')
                source = generated / 'default.ini' if name == 'AB3D.INI' else game / name
                if source.suffix.upper() in ('.TXT', '.DOC', '.INI', '.BAT'):
                    output.writestr(name, source.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
                else:
                    output.write(source, name)
        print(f'Packaged {archive} ({archive.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
