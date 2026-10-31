import { describe, expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { openLevel } from '../src/model/level';
import { encodePolygonSurface, encodeWallSurface, polygonForCell, readPolygonSurface, readWallSurface } from '../src/model/surfaces';
import { originalFiles } from './helpers';

describe('current engine surface records', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('rebuilds level %i walls and polygons byte for byte with authored material settings', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level)), reader = new BinaryReader(document.files.graphics, 'graphics');
    let walls = 0, polygons = 0;
    for (const zone of document.zones) for (const graphic of [...zone.lower, ...zone.upper]) {
      if (graphic.wall) {
        const surface = readWallSurface(reader, graphic.offset);
        expect(surface).toEqual(graphic.wall);
        expect(encodeWallSurface(surface)).toEqual(document.files.graphics.subarray(graphic.offset, graphic.offset + graphic.size));
        walls++;
      }
      if (graphic.polygon) {
        const surface = readPolygonSurface(reader, graphic.offset);
        expect(surface).toEqual(graphic.polygon);
        expect(encodePolygonSurface(surface)).toEqual(document.files.graphics.subarray(graphic.offset, graphic.offset + graphic.size));
        polygons++;
      }
    }
    expect(walls).toBeGreaterThan(0); expect(polygons).toBeGreaterThan(0);
  });
  it('binds a material to new cell corners while preserving its scale, tile and brightness', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const template = document.zones.flatMap(zone => zone.lower).find(graphic => graphic.polygon)!.polygon!;
    const before = structuredClone(template), vertices = [100, 102, 101];
    const generated = polygonForCell(template, vertices, 64.25);
    expect(generated).toEqual({ ...template, points: vertices, closingPoint: 100, heightQuarter: 257 });
    const bytes = encodePolygonSurface(generated);
    expect(readPolygonSurface(new BinaryReader(bytes, 'generated polygon'), 0)).toEqual(generated);
    expect(template).toEqual(before);
    expect(() => polygonForCell(template, [0, 1], 0)).toThrow(/corner count/);
    expect(() => polygonForCell(template, [0, 1, 2], 0.1)).toThrow(/quarter units/);
    expect(() => polygonForCell(template, [0, 1, 2048], 0)).toThrow(/point index/);
  });
  it('retains wall phase and sampling independently when endpoints or heights change', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const wall = document.zones.flatMap(zone => zone.lower).find(graphic => graphic.wall)!.wall!;
    const changed = { ...wall, points: [100, 101] as [number, number], topFixed: -12345, bottomFixed: 67890 };
    expect(readWallSurface(new BinaryReader(encodeWallSurface(changed), 'generated wall'), 0)).toEqual(changed);
    expect(() => encodeWallSurface({ ...wall, verticalMask: 256 })).toThrow(/Byte/);
    expect(() => encodeWallSurface({ ...wall, topFixed: 2147483648 })).toThrow(/Wall top/);
    expect(() => encodeWallSurface({ ...wall, points: [-1, 1] })).toThrow(/point index/);
  });
  it('rejects truncated and incorrectly typed records before reading material fields', () => {
    const bytes = new Uint8Array(30), reader = new BinaryReader(bytes, 'invalid graphic');
    expect(() => readWallSurface(new BinaryReader(bytes.subarray(0, 29), 'truncated'), 0)).toThrow(/record extends/);
    expect(() => readPolygonSurface(reader, 0)).toThrow(/not a floor/);
    bytes[1] = 1;
    expect(() => readWallSurface(reader, 0)).toThrow(/not a straight wall/);
    bytes[4] = 127;
    expect(() => readPolygonSurface(reader, 0)).toThrow(/corner count/);
  });
});
