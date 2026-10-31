import { describe, expect, it } from 'vitest';
import { importedAnimationTables, buildAnimationTables, remapAnimationRooms } from '../src/model/animationBuild';
import { BinaryReader } from '../src/model/binary';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';

describe('current animation table assembly', () => {
  it('maps live animation rooms and file aliases while retaining graphics, switch slots and source state', () => {
    const original = openLevel('original', originalFiles(1)), data = original.files.data.slice(), last = original.zones.at(-1)!;
    new DataView(data.buffer).setInt16(last.offset, 300);
    const document = openLevel('alias source', { ...original.files, data }), { tables } = importedAnimationTables(document);
    const graphic = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.type === 7)!.offset;
    const source = { ...tables, lifts: tables.lifts!.map((record, i) => i ? record : { ...record, zone: 300 }),
      water: tables.water.map((group, i) => i ? group : { ...group, entries: [{ zone: 300, graphic }] }) };
    const before = structuredClone(source), ids = new Map(document.zones.map(zone => [zone.id, document.zones.length - 1 - zone.id]));
    const mapped = remapAnimationRooms(document, source, ids);
    expect(mapped.lifts![0]!.zone).toBe(0); expect(mapped.water[0]!.entries).toEqual([{ zone: 0, graphic }]);
    source.doors!.forEach((record, i) => expect(mapped.doors![i]).toEqual({ ...record, zone: ids.get(record.zone) }));
    source.switches!.forEach((record, i) => expect(mapped.switches![i]).toEqual(record.zone < 0 ? record : { ...record, zone: ids.get(record.zone) }));
    expect(source).toEqual(before);
    expect(() => remapAnimationRooms(document, source, new Map())).toThrow(/no generated room mapping/);
  });
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('reassembles level %i animation tables exactly and rebases section addresses', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level)), { start, tables } = importedAnimationTables(document);
    const built = buildAnimationTables(tables, start, pointer => pointer), reader = new BinaryReader(document.files.graphics, 'original');
    expect(built.bytes).toEqual(document.files.graphics.subarray(start));
    expect([built.doors, built.lifts, built.switches]).toEqual([reader.i32(0), reader.i32(4), reader.i32(8)]);
    const moved = buildAnimationTables(tables, start + 128, pointer => pointer + 256);
    expect([moved.doors, moved.lifts, moved.switches]).toEqual([built.doors + 128, built.lifts + 128, built.switches + 128]);
    const after = new BinaryReader(moved.bytes, 'relocated tables'), before = new BinaryReader(built.bytes, 'original tables');
    for (const item of document.mechanisms) {
      const offset = item.offset - start;
      expect(after.i32(offset + 8)).toBe(before.i32(offset + 8) + 256);
      expect(after.i16(offset + 12)).toBe(item.zone);
      item.walls.forEach((wall, i) => {
        const entry = offset + 18 + i * 10;
        expect(after.i16(entry)).toBe(wall.line); expect(after.i32(entry + 2)).toBe(wall.graphic + 256); expect(after.i32(entry + 6)).toBe(wall.base);
      });
    }
    document.switches.forEach(item => expect(after.i32(item.offset - start + 6)).toBe(item.zone < 0 ? item.graphic : item.graphic + 256));
  });

  it('writes explicit room, collision-line and water relationships without overwriting rule or texture-phase bytes', () => {
    const document = openLevel('Level 1', originalFiles(1)), { start, tables } = importedAnimationTables(document);
    const changed = { ...tables, doors: tables.doors!.map(item => ({ ...item, zone: 200,
      walls: item.walls.map(wall => ({ ...wall, line: 1234 })) })), water: tables.water.map(group => ({ ...group,
      entries: [...group.entries, { zone: 200, graphic: document.zones[0]!.lower.find(record => record.polygon)!.offset + 2 }] })) };
    const built = buildAnimationTables(changed, start, pointer => pointer), reader = new BinaryReader(built.bytes, 'changed tables');
    let cursor = 0;
    for (const item of changed.doors!) {
      expect(reader.i16(cursor + 12)).toBe(200);
      expect(reader.bytes.slice(cursor, cursor + 12)).toEqual(item.header.slice(0, 12));
      expect(reader.bytes.slice(cursor + 14, cursor + 18)).toEqual(item.header.slice(14));
      item.walls.forEach((wall, i) => { expect(reader.i16(cursor + 18 + i * 10)).toBe(1234); expect(reader.i32(cursor + 24 + i * 10)).toBe(wall.base); });
      cursor += 18 + item.walls.length * 10 + 2;
    }
    expect(reader.i16(cursor)).toBe(999);
    expect(built.switches).toBe(new BinaryReader(document.files.graphics, 'original').i32(8) + 21 * 6);
  });

  it('rejects malformed table shapes, reserved headers and unresolved graphics without changing input', () => {
    const document = openLevel('Level 1', originalFiles(1)), { start, tables } = importedAnimationTables(document), before = document.files.graphics.slice();
    expect(() => buildAnimationTables({ ...tables, water: [] }, start, pointer => pointer)).toThrow(/21 water/);
    expect(() => buildAnimationTables({ ...tables, switches: [] }, start, pointer => pointer)).toThrow(/Switch slot count/);
    const header = tables.doors![0]!.header.slice(); new DataView(header.buffer).setInt16(0, 999);
    expect(() => buildAnimationTables({ ...tables, doors: [{ ...tables.doors![0]!, header }] }, start, pointer => pointer)).toThrow(/reserved 999/);
    expect(() => buildAnimationTables(tables, start, () => -1)).toThrow(/graphics pointer/);
    expect(() => buildAnimationTables(tables, 16 * 1024 * 1024, pointer => pointer)).toThrow(/Graphics file size/);
    expect(document.files.graphics).toEqual(before);
  });

  it('reads the final permitted moving-room and wall-list entries before their terminators', () => {
    const document = openLevel('capacity', originalFiles(1)), { tables } = importedAnimationTables(document), template = tables.doors![0]!;
    const doors = Array.from({ length: 256 }, (_, i) => ({ ...template,
      walls: i ? [] : Array.from({ length: 64 }, () => ({ ...template.walls[0]! })) }));
    const graphics = buildGraphicsStreams(document, undefined, undefined, { animations: { ...tables, doors } });
    const rebuilt = openLevel('capacity', { ...document.files, graphics });
    expect(rebuilt.mechanisms.filter(item => item.kind === 'door')).toHaveLength(256);
    expect(rebuilt.mechanisms[0]!.walls).toHaveLength(64);
    expect(() => buildAnimationTables({ ...tables, doors: [...doors, template] }, 16, pointer => pointer)).toThrow(/Moving room count/);
    expect(() => buildAnimationTables({ ...tables, doors: [{ ...template, walls: Array.from({ length: 65 }, () => template.walls[0]!) }] }, 16, pointer => pointer)).toThrow(/Moving wall count/);
  });

  it('distinguishes absent animation tables from empty present tables', () => {
    const document = openLevel('optional tables', originalFiles(1)), { tables } = importedAnimationTables(document);
    const absent = { doors: null, lifts: null, water: [], switches: null, trailing: new Uint8Array() };
    const disabled = openLevel('absent tables', { ...document.files, graphics: buildGraphicsStreams(document, undefined, undefined, { animations: absent }) });
    expect(disabled.mechanisms).toEqual([]); expect(disabled.switches).toEqual([]);
    expect(buildGraphicsStreams(disabled)).toEqual(disabled.files.graphics);
    const empty = { ...tables, doors: [], lifts: [], water: tables.water.map(group => ({ ...group, entries: [] })) };
    const graphics = buildGraphicsStreams(document, undefined, undefined, { animations: empty });
    const reader = new BinaryReader(graphics, 'empty present tables');
    expect(reader.i16(reader.i32(0))).toBe(999); expect(reader.i16(reader.i32(4))).toBe(999);
    expect(importedAnimationTables(openLevel('empty tables', { ...document.files, graphics })).tables.water).toHaveLength(21);
    const inactive = buildAnimationTables({ ...absent, switches: tables.switches!.map((slot, i) => ({ ...slot, zone: -1, graphic: 123 + i, point: -i - 1 })) },
      64, () => { throw new Error('An inactive switch must not resolve a live graphics target.'); });
    const slots = new BinaryReader(inactive.bytes, 'inactive slots');
    for (let i = 0; i < 8; i++) { expect(slots.i16(i * 14)).toBe(-1); expect(slots.i16(i * 14 + 4)).toBe(-i - 1); expect(slots.i32(i * 14 + 6)).toBe(123 + i); }
  });
});
