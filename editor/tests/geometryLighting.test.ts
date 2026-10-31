import { expect, it } from 'vitest';
import { buildCellGraphics } from '../src/model/cellGraphics';
import { buildGeometryLighting } from '../src/model/geometryLighting';
import { openLevel } from '../src/model/level';
import { compileRoomTopology } from '../src/model/roomTopology';
import { originalFiles } from './helpers';
import { importedDataTables } from '../src/model/dataBuild';
import { BinaryReader } from '../src/model/binary';
import { readGraphicRecord } from '../src/model/graphics';
import { interpolatePointBrightness } from '../src/model/pointLighting';
import { buildSourceVisibility } from '../src/model/visibilityTables';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('retains level %i lighting and assigns translated geometry from its source identities', level => {
  const document = openLevel('original', originalFiles(level)), topology = compileRoomTopology(document, document.roomSource);
  const original = buildCellGraphics(document, document.roomSource, topology);
  const untouched = buildGeometryLighting(document, document.roomSource, original).points;
  expect(untouched).toEqual(importedDataTables(document).points);
  const source = { ...document.roomSource, rooms: document.roomSource.rooms.map(room => ({ ...room,
    vertices: room.vertices.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) })) };
  const translated = buildCellGraphics(document, source, compileRoomTopology(document, source));
  const { bindings, points, graphics } = buildGeometryLighting(document, source, translated);
  expect(points.length).toBeGreaterThan(document.points.length);
  expect(points.slice(0, document.points.length)).toEqual(importedDataTables(document).points);
  for (const [id, binding] of bindings) if (id >= document.points.length) {
    expect(points[id]!.x).toBe(graphics.topology.points[id]!.x);
    expect(points[id]!.z).toBe(graphics.topology.points[id]!.z);
    expect(binding.kind === 'copy' || binding.kind === 'segment').toBe(true);
  }
  expect(graphics.topology.cells).toEqual(translated.topology.cells);
  if (level === 1) expect(graphics.topology.points.length).toBeGreaterThan(translated.topology.points.length);
  if (level === 1) {
    const before = buildSourceVisibility(translated.topology, translated.pointReferences);
    const after = buildSourceVisibility(graphics.topology, graphics.pointReferences);
    expect(after.connectTableOffset).toBe(before.connectTableOffset);
    expect(after.clips.slice(0, before.clips.length)).toEqual(before.clips);
    expect(after.clips.slice(before.clips.length).every(byte => byte === 255)).toBe(true);
    for (const [id, room] of before.rooms) expect(after.rooms.get(id)!.visibility).toEqual(room.visibility);
  }
  const originals = new Map(document.zones.flatMap(zone => [...zone.lower, ...zone.upper].map(record => [record.offset, record] as const)));
  const data = importedDataTables(document).points;
  for (const stream of graphics.streams.values()) for (const item of [...stream.lower, ...(stream.upper ?? [])]) {
    const original = originals.get(item.sourceOffset!)!, record = readGraphicRecord(new BinaryReader(item.bytes, 'lit material'), 0, points.length);
    for (const id of record.points) {
      const position = { x: points[id]!.x - 8192, z: points[id]!.z - 8192 };
      const originalId = original.points.find(id => document.points[id]!.x === position.x && document.points[id]!.z === position.z);
      if (originalId !== undefined) {
        expect(points[id]!.lowerBrightness).toBe(data[originalId]!.lowerBrightness);
        expect(points[id]!.upperBrightness).toBe(data[originalId]!.upperBrightness);
      } else {
        expect(original.wall).toBeDefined();
        const [a, b] = original.points.map(id => data[id]!), dx = b!.x - a!.x, dz = b!.z - a!.z;
        const n = (position.x - a!.x) * dx + (position.z - a!.z) * dz, d = dx * dx + dz * dz;
        expect(points[id]!.lowerBrightness).toBe(interpolatePointBrightness(a!.lowerBrightness, b!.lowerBrightness, n, d));
        expect(points[id]!.upperBrightness).toBe(interpolatePointBrightness(a!.upperBrightness, b!.upperBrightness, n, d));
      }
    }
  }
});

it('interpolates static and animated channels automatically at an inserted boundary point', () => {
  const original = openLevel('split lighting', originalFiles(1)), room = original.roomSource.rooms[0]!;
  const a = room.vertices[0]!, b = room.vertices[1]!, midpoint = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 };
  const data = original.files.data.slice(), view = new DataView(data.buffer);
  for (const point of original.points) {
    if (point.x === a.x && point.z === a.z) { view.setInt16(original.pointBrightnessOffset + point.id * 4, -10); view.setInt16(original.pointBrightnessOffset + point.id * 4 + 2, 0x7102); }
    if (point.x === b.x && point.z === b.z) { view.setInt16(original.pointBrightnessOffset + point.id * 4, 10); view.setInt16(original.pointBrightnessOffset + point.id * 4 + 2, 0x7112); }
  }
  const document = openLevel('split lighting', { ...original.files, data });
  const source = { ...document.roomSource, rooms: document.roomSource.rooms.map(item => item.id !== room.id ? item : ({ ...room, vertices: [a, midpoint, ...room.vertices.slice(1)],
    wallSources: [room.wallSources[0]!, ...room.wallSources] })) };
  const graphics = buildCellGraphics(document, source, compileRoomTopology(document, source));
  const built = buildGeometryLighting(document, source, graphics);
  let samples = 0;
  for (const stream of [built.graphics.streams.get(0)!]) for (const item of [...stream.lower, ...(stream.upper ?? [])]) {
    const record = readGraphicRecord(new BinaryReader(item.bytes, 'split lighting'), 0, built.points.length);
    for (const id of record.points) if (built.points[id]!.x === midpoint.x && built.points[id]!.z === midpoint.z) {
      expect(built.points[id]!.lowerBrightness).toBe(0); expect(built.points[id]!.upperBrightness).toBe(0x710a); samples++;
    }
  }
  expect(samples).toBeGreaterThan(1);
});

it('fails atomically if distinct lighting identities would exceed the point capacity', () => {
  const document = openLevel('capacity', originalFiles(1));
  const source = { ...document.roomSource, rooms: document.roomSource.rooms.map(room => ({ ...room,
    vertices: room.vertices.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) })) };
  const graphics = buildCellGraphics(document, source, compileRoomTopology(document, source));
  graphics.topology.points.push(...Array.from({ length: 2048 - graphics.topology.points.length }, () => ({ x: 30000, z: 30000 })));
  const before = structuredClone(graphics);
  expect(() => buildGeometryLighting(document, source, graphics)).toThrow(/separate lighting identities beyond.*2,048/);
  expect(graphics).toEqual(before);
});
