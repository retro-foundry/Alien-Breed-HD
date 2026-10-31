"""Execute the original editor's embedded 68020 geometry bank for TS fixtures.

Development-only dependency: Unicorn 2.1.4. No emulator ships with the React app.
Bank extraction follows AMOS's attached AmBs/AmBk format; the original file and
payload hashes are pinned so different editor versions cannot update fixtures
silently. Addresses below are relative to the Asm bank's payload (Start(9)).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import struct

from unicorn import Uc, UC_ARCH_M68K, UC_MODE_BIG_ENDIAN
from unicorn.m68k_const import UC_CPU_M68K_M68020, UC_M68K_REG_A7, UC_M68K_REG_PC

ROOT = Path(__file__).resolve().parents[1]
BASE, STOP = 0x10000, 0xf0000


def original_bank() -> bytes:
    original = (ROOT / 'amiga/archive/amos/leved.amos').read_bytes()
    assert hashlib.sha256(original).hexdigest() == '26b7f6fa8255515509a3850e92f263ceebcec7e8b70952d2fa0b107e0459b779'
    offset = 20 + struct.unpack_from('>I', original, 16)[0]
    assert original[offset:offset + 4] == b'AmBs'
    offset += 6
    assert original[offset:offset + 6] == b'AmBk\0\x09'
    length = (struct.unpack_from('>I', original, offset + 8)[0] & 0xfffffff) + 12
    bank = original[offset + 20:offset + length]
    assert hashlib.sha256(bank).hexdigest() == 'b73b5c1984829910cfc3fbc888c56a24e9fb26857604289ac16860d0840458f9'
    return bank


def emulator(bank: bytes) -> Uc:
    machine = Uc(UC_ARCH_M68K, UC_MODE_BIG_ENDIAN)
    machine.ctl_set_cpu_model(UC_CPU_M68K_M68020)
    machine.mem_map(BASE, 0x100000)
    machine.mem_write(BASE, bank)
    write(machine, 0xe0000, 'I', STOP)
    machine.reg_write(UC_M68K_REG_A7, 0xe0000)
    return machine


def write(machine: Uc, address: int, fmt: str, *values: int) -> None:
    machine.mem_write(address, struct.pack('>' + fmt, *values))


def run(machine: Uc) -> None:
    machine.emu_start(BASE + 54, STOP, count=500000)
    assert machine.reg_read(UC_M68K_REG_PC) == STOP, 'Original routine exceeded its instruction bound'


def order_mask(bank: bytes, source: list, target: list, walls: list) -> int:
    points = source + target
    machine = emulator(bank)
    write(machine, 0x20000, f'{len(points)}i', *(p[0] for p in points))
    write(machine, 0x22000, f'{len(points)}i', *(p[1] for p in points))
    write(machine, 0x24000, f'{len(source) + 1}i', *range(len(source)), 0)
    write(machine, 0x26000, f'{len(target) + 1}i', *range(len(source), len(points)), len(source))
    write(machine, 0x28000, f'{len(walls)}i', *walls)
    for offset, value in [(4, 0x20000), (8, 0x22000), (12, 0x28000), (42, 0x24000), (46, 0x26000)]:
        write(machine, BASE + offset, 'I', value)
    write(machine, BASE + 40, 'H', 1)
    write(machine, BASE + 50, 'HH', len(source) - 1, len(target) - 1)
    run(machine)
    return struct.unpack('>I', machine.mem_read(BASE + 50, 4))[0]


def portal_points(bank: bytes, portal: list, points: list, used: list, mode: int) -> list:
    machine = emulator(bank)
    write(machine, 0x20000, f'{len(points)}i', *(p[0] for p in points))
    write(machine, 0x22000, f'{len(points)}i', *(p[1] for p in points))
    write(machine, 0x24000, f'{len(used)}i', *used)
    for offset, value in [(0, 0x28000), (4, 0x20000), (8, 0x22000), (30, 0x24000), (34, 0x26000)]:
        write(machine, BASE + offset, 'I', value)
    write(machine, BASE + 12, '8h', *(coord for p in portal for coord in p))
    write(machine, BASE + 28, 'H', len(points) - 1)
    write(machine, BASE + 38, 'HH', mode, 0)
    run(machine)
    return list(struct.unpack(f'>{len(points)}i', machine.mem_read(0x26000, len(points) * 4)))


def separates(bank: bytes, source: list, target: list, left: list, right: list) -> bool:
    points = source + target + left + right
    machine = emulator(bank)
    write(machine, 0x20000, f'{len(points)}i', *(p[0] for p in points))
    write(machine, 0x22000, f'{len(points)}i', *(p[1] for p in points))
    write(machine, 0x24000, f'{len(source)}i', *range(len(source)))
    write(machine, 0x26000, f'{len(target)}i', *range(len(source), len(source) + len(target)))
    offset = len(source) + len(target)
    write(machine, 0x28000, f'{len(left)}i', *range(offset, offset + len(left)))
    write(machine, 0x2a000, f'{len(right)}i', *range(offset + len(left), len(points)))
    for field, pointer in [(4, 0x20000), (8, 0x22000), (12, 0x28000), (16, 0x2a000), (42, 0x24000), (46, 0x26000)]:
        write(machine, BASE + field, 'I', pointer)
    write(machine, BASE + 20, 'HH', len(left) - 1, len(right) - 1)
    write(machine, BASE + 40, 'H', 2)
    write(machine, BASE + 50, 'HH', len(source) - 1, len(target) - 1)
    run(machine)
    return bool(struct.unpack('>H', machine.mem_read(BASE + 40, 2))[0])


def main() -> None:
    bank = original_bank()
    rng = random.Random(348)
    masks, classifications, separations = [], [], []
    for _ in range(120):
        def room():
            x, z = rng.randint(-25000, 25000), rng.randint(-25000, 25000)
            width, depth = rng.randint(1, 4000), rng.randint(1, 4000)
            return [[x, z], [x, z + depth], [x + width, z + depth], [x + width, z]]
        source, target = room(), room()
        walls = [rng.randint(0, 2) for _ in target]
        masks.append(dict(source=source, target=target, walls=walls, mask=order_mask(bank, source, target, walls)))
    for mode in range(3):
        for _ in range(30):
            portal = [[rng.randint(-16000, 16000), rng.randint(-16000, 16000)] for _ in range(4)]
            if mode == 1: portal[0] = portal[3].copy()
            if mode == 2: portal[1] = portal[2].copy()
            points = [[rng.randint(-32768, 32767), rng.randint(-32768, 32767)] for _ in range(24)] + portal
            used = [rng.choice([0, 1, 1]) for _ in points]
            classifications.append(dict(portal=portal, points=points, used=used, mode=mode,
                                        result=portal_points(bank, portal, points, used, mode)))
    for _ in range(120):
        source, target = room(), room()
        left = [[rng.randint(-32000, 32000), rng.randint(-32000, 32000)] for _ in range(rng.randint(1, 4))]
        right = [[rng.randint(-32000, 32000), rng.randint(-32000, 32000)] for _ in range(rng.randint(1, 4))]
        separations.append(dict(source=source, target=target, left=left, right=right,
                                result=separates(bank, source, target, left, right)))
    result = dict(authority='leved.amos bank 9, 68020 entry +54, emulated with Unicorn 2.1.4',
                  bankSha256=hashlib.sha256(bank).hexdigest(), masks=masks, classifications=classifications, separations=separations)
    target = ROOT / 'editor/tests/fixtures/original-geometry.json'
    target.write_text(json.dumps(result, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'Wrote {len(masks)} ordering masks, {len(classifications)} classifications and {len(separations)} separation checks')


if __name__ == '__main__':
    main()
