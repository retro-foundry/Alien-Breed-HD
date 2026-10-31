import { describe, expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { buildGraphicsStreams, importedGraphicsStreams } from '../src/model/graphicsBuild';
import { openLevel, parseLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { encodePolygonSurface } from '../src/model/surfaces';
import { buildAuthoredRoom } from './authoredRoom';

function expectWaterTargets(document: ReturnType<typeof openLevel>, rebuilt: ReturnType<typeof openLevel>, a: BinaryReader, b: BinaryReader) {
  const waterStart = (level: typeof document, reader: BinaryReader) => {
    const last = level.mechanisms.filter(item => item.kind === 'lift').at(-1);
    return last ? last.offset + 18 + last.walls.length * 10 + 4 : reader.i32(4) + 2;
  };
  let oldWater = waterStart(document, a), newWater = waterStart(rebuilt, b);
  for (let group = 0; group < 21; group++) {
    expect(b.bytes.subarray(newWater, newWater + 14)).toEqual(a.bytes.subarray(oldWater, oldWater + 14));
    oldWater += 14; newWater += 14;
    while (a.i16(oldWater) !== -1) {
      expect(b.i16(newWater)).toBe(a.i16(oldWater));
      const oldTarget = a.i32(oldWater + 2), newTarget = b.i32(newWater + 2);
      expect(b.bytes.subarray(newTarget, newTarget + 6)).toEqual(a.bytes.subarray(oldTarget, oldTarget + 6));
      oldWater += 6; newWater += 6;
    }
    expect(b.i16(newWater)).toBe(-1); oldWater += 2; newWater += 2;
  }
}

describe('graphics stream assembly and relocation', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('assembles complete level %i record lists and relocates animation targets after record reordering', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level)), streams = importedGraphicsStreams(document);
    expect(buildGraphicsStreams(document, undefined, undefined, { streams })).toEqual(document.files.graphics);
    const reordered = new Map([...streams].map(([id, stream]) => [id, { lower: stream.lower.toReversed(),
      ...(stream.upper ? { upper: stream.upper.toReversed() } : {}) }]));
    const graphics = buildGraphicsStreams(document, document.zones.map(zone => zone.id).toReversed(), undefined, { streams: reordered });
    const rebuilt = openLevel('complete streams', { ...document.files, graphics });
    const a = new BinaryReader(document.files.graphics, 'original'), b = new BinaryReader(graphics, 'assembled');
    for (const zone of document.zones) {
      const target = rebuilt.zones[zone.id]!;
      for (const [original, generated] of [[zone.lower, target.lower], [zone.upper, target.upper]] as const)
        expect(generated.map(record => b.bytes.slice(record.offset, record.offset + record.size)))
          .toEqual(original.toReversed().map(record => a.bytes.slice(record.offset, record.offset + record.size)));
    }
    for (const mechanism of document.mechanisms) {
      const changed = rebuilt.mechanisms.find(item => item.kind === mechanism.kind && item.id === mechanism.id)!;
      const originalTarget = a.i32(mechanism.offset + 8), target = b.i32(changed.offset + 8);
      expect(b.bytes.slice(target, target + 4)).toEqual(a.bytes.slice(originalTarget, originalTarget + 4));
      mechanism.walls.forEach((wall, i) => expect(b.bytes.slice(changed.walls[i]!.graphic, changed.walls[i]!.graphic + 30))
        .toEqual(a.bytes.slice(wall.graphic, wall.graphic + 30)));
    }
    document.switches.forEach((item, i) => { if (item.zone >= 0)
      expect(b.bytes.slice(rebuilt.switches[i]!.graphic, rebuilt.switches[i]!.graphic + 6)).toEqual(a.bytes.slice(item.graphic, item.graphic + 6));
    });
    expectWaterTargets(document, rebuilt, a, b);
  });

  it('adds an upper stream and inserts records while removing an unreferenced original wall', () => {
    const document = openLevel('original', originalFiles(1)), streams = importedGraphicsStreams(document), original = streams.get(0)!;
    const targets = document.mechanisms.flatMap(item => [new BinaryReader(document.files.graphics, 'graphics').i32(item.offset + 8), ...item.walls.map(wall => wall.graphic)])
      .concat(document.switches.filter(item => item.zone >= 0).map(item => item.graphic));
    const removed = document.zones[0]!.lower.find(record => record.wall && !targets.some(pointer => pointer >= record.offset && pointer < record.offset + record.size))!;
    const floor = original.lower.find(item => document.zones[0]!.lower.find(record => record.offset === item.sourceOffset)!.polygon)!;
    streams.set(0, { lower: [...original.lower.filter(item => item.sourceOffset !== removed.offset), { bytes: floor.bytes }], upper: [{ bytes: floor.bytes }] });
    const rebuilt = openLevel('assembled records', { ...document.files, graphics: buildGraphicsStreams(document, undefined, undefined, { streams }) });
    expect(rebuilt.zones[0]!.lower.filter(record => record.wall)).toHaveLength(document.zones[0]!.lower.filter(record => record.wall).length - 1);
    expect(rebuilt.zones[0]!.upper).toHaveLength(1);
    expect(rebuilt.zones[0]!.lower.at(-1)!.polygon).toEqual(rebuilt.zones[0]!.upper[0]!.polygon);
  });

  it('rejects removed animation targets, ambiguous identities and conflicting stream edits', () => {
    const document = openLevel('original', originalFiles(1)), before = document.files.graphics.slice(), streams = importedGraphicsStreams(document);
    const target = document.mechanisms.find(item => item.walls.length)!.walls[0]!.graphic;
    for (const [id, stream] of streams) streams.set(id, { lower: stream.lower.filter(record => record.sourceOffset !== target),
      ...(stream.upper ? { upper: stream.upper.filter(record => record.sourceOffset !== target) } : {}) });
    expect(() => buildGraphicsStreams(document, undefined, undefined, { streams })).toThrow(/graphics pointer.*outside/);
    const original = importedGraphicsStreams(document).get(0)!, first = original.lower[0]!;
    expect(() => buildGraphicsStreams(document, undefined, undefined, { streams: new Map([[0, { lower: [first, first] }]]) })).toThrow(/duplicate source/);
    expect(() => buildGraphicsStreams(document, undefined, undefined, { streams: new Map([[0, { lower: [{ bytes: first.bytes, sourceOffset: 999999 }] }]]) })).toThrow(/missing.*source/);
    expect(() => buildGraphicsStreams(document, undefined, undefined, { streams: new Map([[0, original]]), replacements: new Map([[first.sourceOffset!, first.bytes]]) })).toThrow(/not both/);
    const floor = document.zones[0]!.lower.find(record => record.type === 1)!;
    expect(() => buildGraphicsStreams(document, undefined, undefined, { streams: new Map([[0, { lower: [{ sourceOffset: floor.offset,
      bytes: encodePolygonSurface({ ...floor.polygon!, type: 2 }) }] }]]) })).toThrow(/source graphics opcode/);
    expect(() => buildGraphicsStreams(document, undefined, undefined, { streams: new Map([[0, { upper: [] } as never]]) })).toThrow(/explicit lower/);
    expect(() => buildGraphicsStreams(document, undefined, undefined, { streams: new Map([[0, { ...original, upper: null } as never]]) })).toThrow(/record list/);
    expect(document.files.graphics).toEqual(before);
  });
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('resizes level %i polygon records and retains the actual animation targets', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    const polygons = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).filter(record => record.polygon);
    const replacements = new Map(polygons.map(record => [record.offset, encodePolygonSurface({ ...record.polygon!, points: [...record.points, record.points[0]!] })]));
    const graphics = buildGraphicsStreams(document, undefined, undefined, { replacements });
    const changedStreams = new Map([...importedGraphicsStreams(document)].map(([id, stream]) => {
      const resize = (records: typeof stream.lower) => records.map(record => ({ ...record, bytes: replacements.get(record.sourceOffset!) ?? record.bytes }));
      return [id, { lower: resize(stream.lower), ...(stream.upper ? { upper: resize(stream.upper) } : {}) }];
    }));
    expect(buildGraphicsStreams(document, undefined, undefined, { streams: changedStreams })).toEqual(graphics);
    expect(graphics.length).toBe(document.files.graphics.length + polygons.length * 2);
    const rebuilt = parseLevel('resized polygons', { ...document.files, graphics });
    const a = new BinaryReader(document.files.graphics, 'original'), b = new BinaryReader(graphics, 'resized');
    document.mechanisms.forEach((mechanism, i) => {
      const changed = rebuilt.mechanisms[i]!, oldTarget = a.i32(mechanism.offset + 8), newTarget = b.i32(changed.offset + 8);
      expect(b.bytes.subarray(newTarget, newTarget + 4)).toEqual(a.bytes.subarray(oldTarget, oldTarget + 4));
      mechanism.walls.forEach((wall, j) => expect(b.bytes.subarray(changed.walls[j]!.graphic, changed.walls[j]!.graphic + 30)).toEqual(a.bytes.subarray(wall.graphic, wall.graphic + 30)));
    });
    document.switches.forEach((item, i) => { if (item.zone >= 0) {
      const changed = rebuilt.switches[i]!;
      expect(b.bytes.subarray(changed.graphic, changed.graphic + 6)).toEqual(a.bytes.subarray(item.graphic, item.graphic + 6));
    } });
    for (const zone of rebuilt.zones) for (const record of [...zone.lower, ...zone.upper]) if (record.polygon)
      expect(record.points.at(-1)).toBe(record.points[0]);
  });
  it('assembles a new coordinate set, collision boundary, room stream and player spawn using the shared builders', () => {
    const original = openLevel('original', originalFiles(1)), compiled = buildAuthoredRoom(original), id = original.zones.length;
    expect(compiled.zones).toHaveLength(id + 1); expect(compiled.points.length).toBe(original.points.length + 4);
    expect(compiled.lines.length).toBe(original.lines.length + 4);
    expect(compiled.starts[0]).toEqual({ player: 1, x: 20064, z: 20064, zone: id });
    expect(compiled.zones[id]!.lower.filter(record => record.wall)).toHaveLength(4);
    expect(compiled.objects).toHaveLength(original.objects.length);
    const reader = new BinaryReader(compiled.files.clips, 'authored clipping');
    expect(reader.bytes.length - compiled.connectTableOffset).toBe(compiled.points.length * 4);
  });
  it('rejects malformed replacements, changed opcodes and duplicate new stream identities', () => {
    const document = openLevel('original', originalFiles(1)), polygon = document.zones[0]!.lower.find(record => record.polygon)!;
    expect(() => buildGraphicsStreams(document, undefined, undefined, { replacements: new Map([[999999, new Uint8Array(2)]]) })).toThrow(/no longer exists/);
    expect(() => buildGraphicsStreams(document, undefined, undefined, { replacements: new Map([[polygon.offset, new Uint8Array(2)]]) })).toThrow(/record extends/);
    const encoded = encodePolygonSurface({ ...polygon.polygon!, type: 2 });
    expect(() => buildGraphicsStreams(document, undefined, undefined, { replacements: new Map([[polygon.offset, encoded]]) })).toThrow(/retain its graphics opcode/);
    expect(() => buildGraphicsStreams(document, undefined, undefined, { additions: [{ id: 0, lower: [encoded] }] })).toThrow(/unique new zone identities/);
  });
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('reassembles level %i and relocates reversed streams without losing animation references', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    expect(buildGraphicsStreams(document)).toEqual(document.files.graphics);
    const graphics = buildGraphicsStreams(document, document.zones.map(zone => zone.id).toReversed());
    expect(graphics).not.toEqual(document.files.graphics);
    const rebuilt = parseLevel(document.name, { ...document.files, graphics });
    const a = new BinaryReader(document.files.graphics, 'original graphics'), b = new BinaryReader(graphics, 'rebuilt graphics');
    for (const zone of document.zones) for (const upper of [false, true]) {
      const original = upper ? zone.upper : zone.lower, changed = upper ? rebuilt.zones[zone.id]!.upper : rebuilt.zones[zone.id]!.lower;
      expect(changed.map(({ offset: _, ...record }) => record)).toEqual(original.map(({ offset: _, ...record }) => record));
    }
    for (const [index, mechanism] of document.mechanisms.entries()) {
      const changed = rebuilt.mechanisms[index]!;
      expect({ ...changed, offset: 0, walls: [] }).toEqual({ ...mechanism, offset: 0, walls: [] });
      // +8 points inside a floor/roof graphics record, never into LEVELDATA.
      expect(b.bytes.subarray(b.i32(changed.offset + 8), b.i32(changed.offset + 8) + 4))
        .toEqual(a.bytes.subarray(a.i32(mechanism.offset + 8), a.i32(mechanism.offset + 8) + 4));
      expect(changed.walls.map(({ graphic: _, ...wall }) => wall)).toEqual(mechanism.walls.map(({ graphic: _, ...wall }) => wall));
      mechanism.walls.forEach((wall, i) => expect(b.bytes.subarray(changed.walls[i]!.graphic, changed.walls[i]!.graphic + 30))
        .toEqual(a.bytes.subarray(wall.graphic, wall.graphic + 30)));
    }
    document.switches.forEach((item, i) => {
      const changed = rebuilt.switches[i]!;
      expect({ ...changed, offset: 0, graphic: 0 }).toEqual({ ...item, offset: 0, graphic: 0 });
      if (item.zone >= 0) expect(b.bytes.subarray(changed.graphic, changed.graphic + 30)).toEqual(a.bytes.subarray(item.graphic, item.graphic + 30));
    });
    expectWaterTargets(document, rebuilt, a, b);
    expect(buildGraphicsStreams(rebuilt)).toEqual(document.files.graphics);
  });
  it('grows the zone prologue and relocates trailing animation tables as well as streams', () => {
    const document = openLevel('Level 1', originalFiles(1)), capacity = document.zones.length + 10;
    const graphics = buildGraphicsStreams(document, undefined, capacity);
    const rebuilt = parseLevel('expanded prologue', { ...document.files, graphics });
    expect(rebuilt.zones).toHaveLength(document.zones.length);
    expect(rebuilt.mechanisms).toHaveLength(document.mechanisms.length);
    expect(rebuilt.switches).toHaveLength(document.switches.length);
    const original = new BinaryReader(document.files.graphics, 'original'), changed = new BinaryReader(graphics, 'changed');
    for (const offset of [0, 4, 8, 12]) expect(changed.i32(offset)).toBe(original.i32(offset) + (offset === 12 ? 40 : 120));
    expect(buildGraphicsStreams(rebuilt)).toEqual(graphics);
  });
  it('rejects incomplete orders, capacity overflow and unaccounted bytes without modifying input', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const before = document.files.graphics.slice();
    expect(() => buildGraphicsStreams(document, [])).toThrow(/each source zone/);
    expect(() => buildGraphicsStreams(document, undefined, 257)).toThrow(/capacity/);
    expect(() => buildGraphicsStreams(document, undefined, 1)).toThrow(/capacity/);
    expect(document.files.graphics).toEqual(before);
    const corrupt = before.slice(), view = new DataView(corrupt.buffer);
    const active = document.switches.find(item => item.zone >= 0)!;
    view.setInt32(active.offset + 6, 999999);
    const broken = parseLevel('invalid switch pointer', { ...document.files, graphics: corrupt });
    expect(() => buildGraphicsStreams(broken)).toThrow(/outside the compiled streams/);
    expect(document.files.graphics).toEqual(before);
  });
});
