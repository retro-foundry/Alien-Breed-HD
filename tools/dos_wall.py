"""Prepare WallRoutine3's packed five-bit streams without changing word order."""
import struct
from pal_to_header import rgb565_words


def prepare_wall(data: bytes, indexed: bool = False) -> bytes:
    if len(data) <= 2048 or len(data) % 2:
        raise ValueError(f'Invalid original wall layout: {len(data)} bytes')
    words = (len(data) - 2048) // 2
    texels = bytearray(words * 3)
    for i, (word,) in enumerate(struct.iter_unpack('>H', data[2048:])):
        for plane in range(3):
            texels[plane * words + i] = (word >> (plane * 5)) & 31
    palette = data[:2048] if indexed else rgb565_words(data[:2048])
    # ABW1, native-color flag, original shade palette, plane stride, planes.
    # VALSHIFT is selected by each level record, rather than inferred here.
    return b'ABW1' + struct.pack('<I', not indexed) + palette + struct.pack('<I', words) + texels
