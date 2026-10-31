import { describe, expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { buildDataPrefix, buildDataTables, importedDataTables } from '../src/model/dataBuild';
import { openLevel, parseLevel } from '../src/model/level';
import { originalFiles } from './helpers';

describe('current level data-table assembly', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('rebuilds level %i tables exactly and relocates rooms after growing points and collision', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    expect(buildDataTables(document)).toEqual(document.files);
    expect(buildDataPrefix(document, importedDataTables(document), document.zones.length))
      .toEqual(document.files.data.subarray(0, Math.min(...document.zones.map(zone => zone.exitListOffset))));
    const input = importedDataTables(document);
    const expanded = { points: [...input.points, { x: 20000, z: 20000, lowerBrightness: -7, upperBrightness: 19 }],
      collision: [...input.collision, { start: { x: 20000, z: 20000 }, end: { x: 20128, z: 20000 }, neighbour: -1 }] };
    const files = buildDataTables(document, expanded), rebuilt = parseLevel('expanded tables', files);
    expect(rebuilt.points.slice(0, document.points.length).map(({ offset: _, ...point }) => point))
      .toEqual(document.points.map(({ offset: _, ...point }) => point));
    expect(rebuilt.points.at(-1)).toMatchObject({ x: 20000, z: 20000 });
    expect(rebuilt.objects).toEqual(document.objects); expect(rebuilt.starts).toEqual(document.starts);
    expect(rebuilt.controlPoints).toEqual(document.controlPoints);
    expect(rebuilt.mechanisms).toEqual(document.mechanisms); expect(rebuilt.switches).toEqual(document.switches);
    for (const zone of document.zones) {
      const changed = rebuilt.zones.find(item => item.id === zone.id)!;
      expect(changed.offset).toBe(zone.offset + 24);
      expect(files.data.subarray(changed.exitListOffset, changed.dataEnd)).toEqual(document.files.data.subarray(zone.exitListOffset, zone.dataEnd));
    }
    const reader = new BinaryReader(files.data, 'expanded');
    expect(reader.i16(rebuilt.pointBrightnessOffset + document.points.length * 4)).toBe(-7);
    expect(reader.i16(rebuilt.pointBrightnessOffset + document.points.length * 4 + 2)).toBe(19);
    // The appended line is not referenced yet: the reference-based parser
    // correctly leaves it out of the visible line model.
    expect(reader.i32(26)).toBe(new BinaryReader(document.files.data, 'original').i32(26) + 8);
  });
  it('assembles resized tables for a new room count while preserving controls, objects and shot pools', () => {
    const document = openLevel('Level 1', originalFiles(1)), original = new BinaryReader(document.files.data, 'original');
    const points = [{ x: 0, z: 0 }, { x: 0, z: 128 }, { x: 128, z: 128 }, { x: 128, z: 0 }]
      .map((point, i) => ({ ...point, lowerBrightness: i - 2, upperBrightness: i + 3 }));
    const collision = points.map((start, i) => ({ start, end: points[(i + 1) % 4]!, neighbour: -1 }));
    const bytes = buildDataPrefix(document, { points, collision }, 1), reader = new BinaryReader(bytes, 'generated prefix');
    expect(reader.i16(14)).toBe(3); expect(reader.i16(16)).toBe(0); expect(reader.i16(bytes.length - 2)).toBe(1);
    expect(reader.i32(26)).toBe(original.i32(22) + 32);
    expect(bytes.length).toBe(original.i32(22) + 32 + 64 + 2);
    expect(bytes.subarray(54, original.i32(22))).toEqual(original.bytes.subarray(54, original.i32(22)));
    for (const offset of [22, 30, 34, 38, 42, 46, 50]) expect(reader.i32(offset)).toBe(original.i32(offset));
    points.forEach((point, i) => {
      expect(reader.i16(reader.i32(22) + i * 4)).toBe(point.x);
      expect(reader.i16(reader.i32(22) + i * 4 + 2)).toBe(point.z);
      expect(reader.i16(reader.i32(22) + 16 + i * 4)).toBe(point.lowerBrightness);
      expect(reader.i16(reader.i32(22) + 18 + i * 4)).toBe(point.upperBrightness);
    });
    expect(() => buildDataPrefix(document, { points, collision }, 0)).toThrow(/room count/);
    expect(() => buildDataPrefix(document, { points, collision }, 257)).toThrow(/room count/);
    expect(() => buildDataPrefix(document, { points, collision: [{ ...collision[0]!, neighbour: 1 }] }, 1)).toThrow(/missing room/);
    expect(() => buildDataPrefix(document, { points: [], collision }, 1)).toThrow(/Point table size/);
    expect(document.files.data).toEqual(openLevel('original', originalFiles(1)).files.data);
  });
  it('refuses to silently discard referenced tables, invalid brightness or unexplained data', () => {
    const document = openLevel('Level 1', originalFiles(1)), input = importedDataTables(document), before = document.files.data.slice();
    expect(() => buildDataTables(document, { ...input, points: input.points.slice(1) })).toThrow(/Point table size/);
    expect(() => buildDataTables(document, { ...input, collision: input.collision.slice(1) })).toThrow(/Collision table size/);
    expect(() => buildDataTables(document, { ...input, points: input.points.map((point, i) => i ? point : { ...point, lowerBrightness: 32768 }) })).toThrow(/brightness/);
    const damaged = before.slice(); new DataView(damaged.buffer).setInt16(document.zones[0]!.exitListOffset - 2, 0);
    expect(() => buildDataTables(parseLevel('invalid count', { ...document.files, data: damaged }))).toThrow(/zone count/);
    expect(document.files.data).toEqual(before);
  });
});
