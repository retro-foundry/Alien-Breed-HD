"""Precompute conservative index-zero row masks from the original sprite data."""
import ast
import hashlib
import re


ALPHA_ROWS = 64  # Hint capacity; higher source rows use the ordinary sampler.


def source_frames(source, name):
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.S)
    match = re.search(r'static const SpriteFrame frames_' + name +
                      r'\[\]\s*=\s*\{(.*?)\};', source, re.S)
    if not match:
        raise ValueError(f'Original renderer frame array is missing: {name}')
    pairs = re.findall(r'\{([^{}]+),([^{}]+)\}', match[1])
    remaining = re.sub(r'\{[^{}]+,[^{}]+\}', '', match[1])
    if not pairs or remaining.strip(' \t\r\n,'):
        raise ValueError(f'Unsupported original frame table: {name}')

    def integer(expr):
        def value(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
                a, b = value(node.left), value(node.right)
                return a + b if isinstance(node.op, ast.Add) else a * b
            raise ValueError(f'Unsupported original frame expression: {name}: {expr}')
        return value(ast.parse(expr.strip(), mode='eval').body)

    result = [(integer(a), integer(b)) for a, b in pairs]
    if any(not (0 <= offset <= 65535 and 0 <= down <= 255) for offset, down in result):
        raise ValueError(f'Frame fields exceed the native alpha key: {name}')
    return result


def alpha_sources(root):
    source = (root / 'src/renderer.c').read_text()
    # These supported widths select optimizations, not game dimensions. The
    # exact PTR offset, strip and width must match before a hint can be used.
    for sprite_type, name, frame_name, widths in (
        (0, 'alien2', 'alien', (16, 64)),
        (8, 'explosion', 'explosion', (64,)),
    ):
        wad = (root / 'amiga/data/includes' / (name + '.wad')).read_bytes()
        ptr = (root / 'amiga/data/includes' / (name + '.ptr')).read_bytes()
        if not wad or len(wad) % 2 or len(ptr) % 4:
            raise ValueError(f'Incomplete original sprite words/pointers: {name}')
        queries = sorted({(offset, down, columns)
                          for offset, down in source_frames(source, frame_name)
                          for columns in widths})
        yield dict(sprite_type=sprite_type, name=name, wad=wad, ptr=ptr, queries=queries)


def prepare_sprite_alpha(root):
    banks = []
    for asset in alpha_sources(root):
        wad, ptr = asset['wad'], asset['ptr']
        records = []
        for offset, down, columns in asset['queries']:
            mask = [0, 0]
            available = max(0, (len(ptr) - offset) // 4)
            for row in range(ALPHA_ROWS):
                for col in range(min(columns, available)):
                    entry = ptr[offset + col * 4:offset + col * 4 + 4]
                    mode, start = entry[0], int.from_bytes(entry[1:], 'big')
                    if (not mode and not start) or start >= len(wad):
                        continue
                    words = (len(wad) - start) // 2
                    if words <= down or row > words - 1 - down:
                        continue
                    at = start + 2 * (down + row)
                    word = int.from_bytes(wad[at:at + 2], 'big')
                    if (word >> (0 if mode == 0 else 5 if mode == 1 else 10)) & 31:
                        mask[row >> 5] |= 1 << (row & 31)
                        break
            # An all-set mask cannot skip any row and needs no resident entry.
            if mask != [0xffffffff, 0xffffffff]:
                records.append(((offset << 16) | (down << 8) | columns, *mask))
        banks.append(dict(sprite_type=asset['sprite_type'], records=sorted(records),
                          wad_sha256=hashlib.sha256(wad).hexdigest(),
                          ptr_sha256=hashlib.sha256(ptr).hexdigest()))
    return banks


def write_sprite_alpha(root, output):
    banks = prepare_sprite_alpha(root)
    lines = ['/* Generated from renderer frame arrays and original PTR/WAD bytes. */',
             f'_Static_assert(DOS_SPRITE_ALPHA_ROWS=={ALPHA_ROWS},"Prepared sprite alpha row capacity");']
    for bank in banks:
        name = f'sprite_alpha_{bank["sprite_type"]}'
        lines.append(f'static const uint32_t {name}[][3]={{')
        lines.extend(f'    {{0x{key:08x}u,0x{a:08x}u,0x{b:08x}u}},'
                     for key, a, b in bank['records'])
        lines.append('};')
    lines.append('static const DosSpriteAlphaBank sprite_alpha_banks[]={')
    for bank in banks:
        name = f'sprite_alpha_{bank["sprite_type"]}'
        lines.append(f'    {{{bank["sprite_type"]},sizeof({name})/sizeof(*{name}),{name}}},')
    lines.append('};')
    text = '\n'.join(lines) + '\n'
    if not output.exists() or output.read_text() != text:
        output.write_text(text)
    return banks
