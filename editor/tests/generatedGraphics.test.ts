import { expect, it } from 'vitest';
import { importedAnimationTables } from '../src/model/animationBuild';
import { BinaryReader } from '../src/model/binary';
import { buildGeneratedGraphics } from '../src/model/generatedGraphics';
import { importedGraphicsStreams } from '../src/model/graphicsBuild';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('assembles level %i from an explicit generated room set without changing its bytes', level => {
  const document = openLevel('original', originalFiles(level)), streams = importedGraphicsStreams(document);
  const reader = new BinaryReader(document.files.graphics, 'source');
  const rooms = document.zones.map(zone => ({ id: zone.id, dataOffset: reader.i32(16 + zone.id * 4), ...streams.get(zone.id)! }));
  expect(buildGeneratedGraphics(document, rooms, importedAnimationTables(document).tables)).toEqual(document.files.graphics);
});

it('assembles a reduced room set and keeps absent and empty upper streams distinct', () => {
  const document = openLevel('original', originalFiles(1)), stream = importedGraphicsStreams(document).get(0)!;
  const animations = { doors: null, lifts: null, water: [], switches: null, trailing: new Uint8Array() };
  const bytes = buildGeneratedGraphics(document, [{ id: 0, dataOffset: 100, lower: stream.lower, upper: [] }], animations);
  const reader = new BinaryReader(bytes, 'generated');
  expect(reader.i32(12)).toBe(20); expect(reader.i32(16)).toBe(100);
  const upper = reader.i32(24); expect(upper).toBeGreaterThan(0);
  expect(reader.i16(upper)).toBe(0); expect(reader.i32(upper + 2)).toBe(-1);
  const absent = new BinaryReader(buildGeneratedGraphics(document, [{ id: 0, dataOffset: 100, lower: [] }], animations), 'absent');
  expect(absent.i32(24)).toBe(0);
  expect(() => buildGeneratedGraphics(document, [{ id: 1, dataOffset: 100, lower: [] }], animations)).toThrow(/dense/);
});

it('requires an explicit animation attachment when one source record has several generated copies', () => {
  const document = openLevel('original', originalFiles(1)), before = document.files.graphics.slice();
  const original = importedAnimationTables(document).tables.doors![0]!;
  const record = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === original.graphic)!;
  const item = { sourceOffset: record.offset, bytes: document.files.graphics.slice(record.offset, record.offset + record.size) };
  const rooms = [0, 1].map(id => ({ id, dataOffset: 100 + id * 48, lower: [item] }));
  const animations = { doors: [0, 1].map(zone => ({ ...original, zone, walls: [] })), lifts: null, water: [], switches: null, trailing: new Uint8Array() };
  expect(() => buildGeneratedGraphics(document, rooms, animations)).toThrow(/multiple generated copies/);
  expect(() => buildGeneratedGraphics(document, rooms, animations, () => ({ room: 9, upper: false, index: 0 }))).toThrow(/outside its generated copies/);
  const bytes = buildGeneratedGraphics(document, rooms, animations, (_pointer, context, candidates) => {
    expect(context.kind).toBe('door'); return candidates.find(candidate => candidate.room === context.zone)!;
  });
  const reader = new BinaryReader(bytes, 'generated'), start = reader.i32(0);
  const a = reader.i32(start + 8), b = reader.i32(start + 20 + 8);
  expect(a).not.toBe(b);
  expect(bytes.slice(a, a + record.size)).toEqual(item.bytes);
  expect(bytes.slice(b, b + record.size)).toEqual(item.bytes);
  expect(document.files.graphics).toEqual(before);
});

it('passes wall, lift, water and switch attachment context to the explicit copy selector', () => {
  const document = openLevel('original', originalFiles(1)), tables = importedAnimationTables(document).tables;
  const sourceRecords = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]);
  const water = sourceRecords.find(record => record.type === 7)!;
  const items = sourceRecords.map(record => ({ sourceOffset: record.offset,
    bytes: document.files.graphics.slice(record.offset, record.offset + record.size) }));
  // Repeat the complete source record set in two generated rooms so every
  // live attachment requires a deliberate choice, including interior switch pointers.
  const rooms = [0, 1].map(id => ({ id, dataOffset: 100 + id * 48, lower: items }));
  const animations = { ...tables,
    doors: tables.doors!.map(record => ({ ...record, zone: 0 })),
    lifts: tables.lifts!.map(record => ({ ...record, zone: 1 })),
    water: tables.water.map((group, index) => ({ ...group, entries: index ? [] : [{ zone: 1, graphic: water.offset }] })),
    switches: tables.switches!.map(record => record.zone < 0 ? record : { ...record, zone: 0 }),
  };
  const seen = new Set<string>();
  const bytes = buildGeneratedGraphics(document, rooms, animations, (pointer, context, candidates) => {
    seen.add(context.kind);
    if (context.kind === 'moving-wall') expect(Number.isInteger(context.line)).toBe(true);
    if (context.kind === 'switch') expect(sourceRecords.some(record => record.wall && record.offset + 10 === pointer)).toBe(true);
    return candidates.find(candidate => candidate.room === context.zone)!;
  });
  expect([...seen].sort()).toEqual(['door', 'lift', 'moving-wall', 'switch', 'water']);
  expect(bytes.length).toBeGreaterThan(document.files.graphics.length);
  expect(() => buildGeneratedGraphics(document, rooms, { ...animations, doors: [{ ...animations.doors[0]!, zone: 2 }] })).toThrow(/missing generated room/);
});

it('enforces generated table capacity and rejects missing live attachments', () => {
  const document = openLevel('limits', originalFiles(1));
  const animations = { doors: null, lifts: null, water: [], switches: null, trailing: new Uint8Array() };
  const rooms = Array.from({ length: 256 }, (_, id) => ({ id, dataOffset: 100, lower: [] }));
  const reader = new BinaryReader(buildGeneratedGraphics(document, rooms, animations), 'maximum');
  expect(reader.i32(12)).toBe(16 + 256 * 4);
  const last = reader.i32(reader.i32(12) + 255 * 8);
  expect(reader.i16(last)).toBe(255); expect(reader.i32(last + 2)).toBe(-1);
  expect(() => buildGeneratedGraphics(document, [...rooms, { id: 256, dataOffset: 100, lower: [] }], animations)).toThrow(/room count/);
  const door = importedAnimationTables(document).tables.doors![0]!;
  expect(() => buildGeneratedGraphics(document, rooms, { ...animations, doors: [{ ...door, zone: 0, walls: [] }] })).toThrow(/no generated record/);
  expect(() => buildGeneratedGraphics(document, [{ id: 0, dataOffset: -1, lower: [] }], animations)).toThrow(/data offset/);
});
