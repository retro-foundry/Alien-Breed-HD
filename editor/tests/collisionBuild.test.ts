import { describe, expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { buildCornerWallLists, compileCollisionGeometry, encodeCollisionSegments } from '../src/model/collisionBuild';
import { importRoomSource } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { BinaryReader } from '../src/model/binary';

describe('original collision line encoding', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('reproduces every collision byte in level %i independently', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    const bytes = encodeCollisionSegments(document.lines.map(line => ({ start: { x: line.x, z: line.z },
      end: { x: line.x + line.dx, z: line.z + line.dz }, neighbour: line.connect })));
    const expected = new Uint8Array(document.lines.length * 16);
    for (const line of document.lines) expected.set(document.files.data.subarray(line.offset, line.offset + 16), line.id * 16);
    expect(bytes).toEqual(expected);
    // Independently rebuild the secondary corner lists from primary geometry,
    // not from the original clip/connect tables or stored secondary lists.
    const coordinates = new Map<string, number>();
    for (const point of document.points) {
      const key = `${point.x},${point.z}`; if (!coordinates.has(key)) coordinates.set(key, point.id);
    }
    const corners = buildCornerWallLists(document.zones.map(zone => ({
      points: zone.boundaryIds.map(id => { const line = document.lines[id]!; return coordinates.get(`${line.x},${line.z}`)!; }),
      lines: zone.boundaryIds, neighbours: zone.boundaryIds.map(id => document.lines[id]!.connect),
    })));
    expect(corners).toEqual(document.zones.map(zone => zone.wallIds));
  });
  it('emits the generated cell boundaries with matching endpoints and destination IDs', () => {
    const document = openLevel('Level 1', originalFiles(1)), topology = compileRoomTopology(document, importRoomSource(document));
    const collision = compileCollisionGeometry(topology), reader = new BinaryReader(collision.bytes, 'generated collision');
    expect(collision.boundaries).toHaveLength(topology.cells.length);
    expect(collision.cornerWalls).toHaveLength(topology.cells.length);
    for (const cell of topology.cells) cell.vertices.forEach((point, index) => {
      const offset = collision.boundaries[cell.id]![index]! * 16, start = topology.points[point]!;
      const end = topology.points[cell.vertices[(index + 1) % cell.vertices.length]!]!;
      expect([reader.i16(offset), reader.i16(offset + 2)]).toEqual([start.x, start.z]);
      expect([reader.i16(offset + 4), reader.i16(offset + 6)]).toEqual([end.x - start.x, end.z - start.z]);
      expect(reader.i16(offset + 8)).toBe(cell.neighbours[index]);
      expect(reader.i16(offset + 14)).toBe(0);
    });
  });
  it('rejects zero-length, overflowing and invalid segments rather than wrapping them', () => {
    const start = { x: 0, z: 0 }, valid = { start, end: { x: 128, z: 0 }, neighbour: -1 };
    expect(() => encodeCollisionSegments([{ ...valid, end: start }])).toThrow(/Wall length/);
    expect(() => encodeCollisionSegments([{ ...valid, start: { x: -32768, z: 0 }, end: { x: 32767, z: 0 } }])).toThrow(/Wall X span/);
    expect(() => encodeCollisionSegments([{ ...valid, end: { x: 30000, z: 30000 } }])).toThrow(/Wall length/);
    expect(() => encodeCollisionSegments([{ ...valid, neighbour: 256 }])).toThrow(/Connected room/);
    expect(() => encodeCollisionSegments([{ ...valid, end: { x: NaN, z: 0 } }])).toThrow(/Collision X/);
    expect(() => encodeCollisionSegments([{ ...valid, end: { x: 1.5, z: 0 } }])).toThrow(/Collision X/);
  });
});
