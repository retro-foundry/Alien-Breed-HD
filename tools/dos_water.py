"""Transport AB3DI.s WaterFile phase planes without duplicating them in RAM."""
import struct


def prepare_water(source: bytes) -> bytes:
    if len(source) != 65536:
        raise ValueError('WaterFile must contain exactly 65536 source bytes')
    # AB3DI.s waterpt: d5=(v<<8)|u; address=(d5<<2)+phase_offset.
    # Each even source byte occurs in exactly one of these eight planes.
    offsets = (0, 2, 256, 258, 512, 514, 768, 770)
    planes = bytes(source[((v << 8) | u) * 4 + offset]
                   for offset in offsets for v in range(64) for u in range(64))
    # Preserve the odd bytes in original word order for scalar source reads.
    payload = planes + source[1::2]
    return struct.pack('<4sI', b'ABW1', len(payload)) + payload
