"""Record immutable editor fixtures using the independent C =SB= decompressor."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unpacker', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('editor/tests/fixtures/original-levels.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    unpacker = args.unpacker.resolve()
    levels = []
    with tempfile.TemporaryDirectory(prefix='ab3d-editor-oracle-') as temporary:
        for number in range(16):
            files = {}
            for name in ('twolev.bin', 'twolev.graph.bin', 'twolev.clips'):
                source = root / 'amiga/data/levels' / f'level_{chr(97 + number)}' / name
                target = Path(temporary) / name
                subprocess.run([str(unpacker), str(source), str(target)], check=True, capture_output=True)
                raw = target.read_bytes()
                files[name] = {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                if name == 'twolev.bin':
                    files[name]['header'] = list(struct.unpack_from('>11h8i', raw))
                elif name == 'twolev.graph.bin':
                    files[name]['header'] = list(struct.unpack_from('>4i', raw))
            levels.append({'level': number + 1, 'files': files})
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'authority': 'src/sb_decompress.c via tools/unpack_sb.c', 'levels': levels}, indent=2) + '\n', encoding='utf-8')
    print(f'Recorded {len(levels)} level triplets with independent C decoder hashes.')


if __name__ == '__main__':
    main()
