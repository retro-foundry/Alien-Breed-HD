"""Generate small independent PNG parsing fixtures using Python's zlib/CRC."""
import base64
import binascii
import json
from pathlib import Path
import struct
import zlib


def chunk(kind, payload):
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', binascii.crc32(kind + payload))


def png(width=7, height=5, animated=False, alpha=False):
    header = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    pixels = (b''.join(b'\0' + b''.join(bytes([0, 255, 0, [0, 128, 255][(x + y) % 3]]) for x in range(width)) for y in range(height))
        if alpha else (b'\0' + bytes([0, 255, 0, 255]) * width) * height)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + (chunk(b'acTL', struct.pack('>II', 1, 0)) if animated else b'') + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    data = {'authority': 'W3C PNG chunk layout, Python zlib and binascii.crc32',
        'rgba7x5': base64.b64encode(png()).decode(),
        'alpha7x5': base64.b64encode(png(alpha=True)).decode(),
        'rgba4k': base64.b64encode(png(3840, 2160)).decode(),
        'tooWide': base64.b64encode(png(5000, 1)).decode(),
        'animated': base64.b64encode(png(animated=True)).decode()}
    (root / 'editor/tests/fixtures/png-images.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
