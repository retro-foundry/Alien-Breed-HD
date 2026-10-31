import { describe, expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { buildDataTables, importedDataTables } from '../src/model/dataBuild';
import { openLevel, parseLevel } from '../src/model/level';
import { originalFiles } from './helpers';

describe('current level data-table assembly', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('rebuilds level %i tables exactly and relocates rooms after growing points and collision', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    expect(buildDataTables(document)).toEqual(document.files);
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
