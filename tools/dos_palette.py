"""Offline DOS palettes from original 0RGB assets and level wall references.

Graph layout/entry lengths match level.c and renderer.c zone_gfx_entry_data_skip;
wall order is read from io.c's WallChunk.s-derived wall_texture_table.
No rendered screenshots or generic color cube are used.
"""
from collections import Counter
import hashlib
import re
import struct


def wall_counts(graph, wall_count, num_zones):
    header = struct.unpack_from('>4i', graph)
    zone_graph = header[3]
    end = min([len(graph), *(x for x in header[:3] if x > zone_graph)])
    counts = Counter()
    seen = set()
    limit = min(num_zones, (zone_graph-16)//4, (end-zone_graph)//8)
    for at in range(zone_graph, zone_graph + limit*8, 8):
        for offset in struct.unpack_from('>2i', graph, at):
            if offset <= 0 or offset in seen:
                continue
            seen.add(offset)
            at = offset + 2
            for _ in range(500):
                t = struct.unpack_from('>h', graph, at)[0]
                at += 2
                if t < 0:
                    break
                if t in (0, 13):
                    wall = struct.unpack_from('>h', graph, at + 12)[0]
                    if 0 <= wall < wall_count:
                        counts[wall] += 1
                    size = 28
                elif t == 5:
                    size = 28
                elif t in (3, 12):
                    size = 0
                elif t == 4:
                    size = 2
                elif t == 6:
                    size = 4
                elif t in (1, 2, 7, 8, 9, 10, 11):
                    sides = max(0, min(100, struct.unpack_from('>h', graph, at + 2)[0] + 1))
                    size = 12 + 2 * sides
                else:
                    raise ValueError(f'Unsupported original zone graphics entry {t} at {at-2}')
                at += size
            else:
                raise ValueError('Unterminated original zone graphics stream')
    return counts


def words(data):
    return [x[0] & 4095 for x in struct.iter_unpack('>H', data[:len(data) & ~1])]


def make_palette(histogram):
    # Reserve exact neutral/UI primaries; all remaining representatives are
    # actual source colors, selected by weighted median cut in 4-bit RGB.
    fixed = list(dict.fromkeys([i * 0x111 for i in range(16)] + [0xf00, 0x0f0, 0x00f, 0xff0, 0x0ff, 0xf0f]))
    points = [(c, w, ((c >> 8) & 15, (c >> 4) & 15, c & 15))
              for c, w in sorted(histogram.items()) if w and c not in fixed]
    boxes = [points]
    while len(boxes) < 255 - len(fixed):
        candidates = [(sum(x[1] for x in box) * max(max(x[2][a] for x in box)-min(x[2][a] for x in box)
                         for a in range(3)), i) for i, box in enumerate(boxes) if len(box) > 1]
        if not candidates:
            break
        _, i = max(candidates)
        box = boxes.pop(i)
        axis = max(range(3), key=lambda a: max(x[2][a] for x in box)-min(x[2][a] for x in box))
        box.sort(key=lambda x: (x[2][axis], x[0]))
        half = sum(x[1] for x in box) / 2
        total = 0
        split = 1
        for split, item in enumerate(box[:-1], 1):
            total += item[1]
            if total >= half:
                break
        boxes.extend((box[:split], box[split:]))
    colors = fixed[:]
    for box in boxes:
        weight = sum(x[1] for x in box)
        mean = [sum(x[2][a]*x[1] for x in box)/weight for a in range(3)]
        colors.append(min(box, key=lambda x: sum((x[2][a]-mean[a])**2 for a in range(3)))[0])
    colors.extend([0] * (256 - len(colors)))  # Index 255 is reserved for cutouts.
    components = [((c >> 8) & 15, (c >> 4) & 15, c & 15) for c in colors[:255]]
    lookup = bytearray(4096)
    for c in range(4096):
        r, g, b = (c >> 8) & 15, (c >> 4) & 15, c & 15
        lookup[c] = min(range(255), key=lambda i: (r-components[i][0])**2 +
                       (g-components[i][1])**2 + (b-components[i][2])**2)
    return b'ABP8' + struct.pack('<256H', *colors) + lookup


def stage_palettes(root, generated, sources):
    table = (root / 'src/io.c').read_text().split('wall_texture_table[] = {', 1)[1].split('};', 1)[0]
    names = re.findall(r'\{\s*"([^"]+)"', table)
    wall_pals = [sources['data/includes/walls/' + name.lower()].read_bytes()[:2048] for name in names]
    common = Counter()
    inputs = []
    for name, path in sorted(sources.items()):
        if ((name.startswith('data/pal/') and
             ((name.endswith('.pal') and name!='data/pal/256palette.pal') or name.rsplit('/',1)[-1] in ('floorpalscaled','bumppalscaled','smoothbumppalscaled')))
                or name in ('data/includes/floorpalscaled', 'data/gfx/backfile', 'data/texturemaps/oldtexturepalscaled')):
            data = path.read_bytes()
            common.update(words(data))
            inputs.append(data)
    graphs = [sources[f'data/levels/level_{chr(97+i)}/twolev.graph.bin'].read_bytes() for i in range(16)]
    levels = [sources[f'data/levels/level_{chr(97+i)}/twolev.bin'].read_bytes() for i in range(16)]
    digest = hashlib.sha256(b''.join(inputs + wall_pals + graphs + levels) +
                            (root / 'tools/dos_palette.py').read_bytes()).hexdigest()
    stamp = generated / 'palette.sha256'
    paths = [generated / f'palette{i:02d}.bin' for i in range(17)]
    if not stamp.exists() or stamp.read_text() != digest or not all(p.exists() for p in paths):
        for level, output in enumerate(paths):
            hist = common.copy()
            if level:
                data = sources[f'data/levels/level_{chr(96+level)}/twolev.bin'].read_bytes()
                counts = wall_counts(graphs[level-1], len(names), struct.unpack_from('>h', data, 16)[0])
            else:
                counts = Counter({i: 1 for i in range(len(names))})
            for wall, count in counts.items():
                for color in words(wall_pals[wall]):
                    hist[color] += count * 16
            output.write_bytes(make_palette(hist))
        stamp.write_text(digest)
    for i, path in enumerate(paths):
        sources[f'data/indexed/palette{i:02d}.bin'] = path
