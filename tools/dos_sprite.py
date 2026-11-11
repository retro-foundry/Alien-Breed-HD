"""Prepare ObjDraw3 packed sprite words for the 486 without expanding them."""
import struct


def prepare_sprite(words: bytes, pointers: bytes) -> bytes:
    if not words or len(words) % 2:
        raise ValueError('Sprite WAD must contain complete nonempty 16-bit words')
    if len(pointers) % 4:
        raise ValueError('Sprite PTR must contain complete four-byte columns')
    for start in range(0, len(pointers), 4):
        mode = pointers[start]
        offset = int.from_bytes(pointers[start + 1:start + 4], 'big')
        if (mode or offset) and offset < len(words) and offset % 2:
            raise ValueError(f'Sprite column {start // 4} has an odd word offset: {offset}')
    native = bytearray(len(words))
    native[0::2] = words[1::2]
    native[1::2] = words[0::2]
    return struct.pack('<4sI', b'ABS1', len(native)) + native
