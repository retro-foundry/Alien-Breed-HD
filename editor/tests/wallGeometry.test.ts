import { expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { openLevel } from '../src/model/level';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildCellWallGeometry } from '../src/model/wallGeometry';
import { readWallSurface } from '../src/model/surfaces';
import { originalFiles } from './helpers';
import { normaliseRoomOutline } from '../src/model/geometry';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../src/model/roomSource';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('binds original level %i wall materials to runtime cell boundaries', level => {
  const document = openLevel('original', originalFiles(level)), topology = compileRoomTopology(document, document.roomSource);
  const result = buildCellWallGeometry(document, document.roomSource, topology);
  expect(result.walls.size).toBe(topology.cells.length);
  for (const cell of topology.cells) {
    const template = document.zones[cell.recordZone]!;
    const original = normaliseRoomOutline(template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })));
    const unchanged = JSON.stringify(normaliseRoomOutline(cell.vertices.map(id => topology.points[id]!))) === JSON.stringify(original);
    for (const [records, bound] of [[template.lower, result.walls.get(cell.id)!.lower], [template.upper, result.walls.get(cell.id)!.upper]] as const) {
      if (unchanged) expect(bound.map(item => item.bytes)).toEqual(records.filter(record => record.wall).map(record => document.files.graphics.slice(record.offset, record.offset + record.size)));
      for (const item of bound) {
        const wall = readWallSurface(new BinaryReader(item.bytes, 'bound wall'), 0);
        const original = records.find(record => record.offset === item.sourceOffset)!.wall!;
        expect({ ...wall, points: [], stripStart: 0, stripEnd: 0 }).toEqual({ ...original, points: [], stripStart: 0, stripEnd: 0 });
      }
    }
  }
});

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('regenerates translated level %i wall geometry without changing material fields', level => {
  const document = openLevel('translated', originalFiles(level));
  const source = { ...document.roomSource, rooms: document.roomSource.rooms.map(room => ({ ...room,
    vertices: room.vertices.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) })) };
  const topology = compileRoomTopology(document, source), result = buildCellWallGeometry(document, source, topology);
  for (const zone of document.zones) for (const [upper, records] of [[false, zone.lower], [true, zone.upper]] as const) {
    const bound = topology.cells.filter(cell => cell.recordZone === zone.id).flatMap(cell => result.walls.get(cell.id)![upper ? 'upper' : 'lower']);
    for (const record of records.filter(record => record.wall)) {
      const original = record.wall!, generated = bound.filter(item => item.sourceOffset === record.offset)
        .map(item => readWallSurface(new BinaryReader(item.bytes, 'translated wall'), 0));
      expect(generated.length).toBeGreaterThan(0);
      const endpoints = generated.flatMap(wall => wall.points.map(id => result.topology.points[id]!));
      original.points.forEach(id => expect(endpoints).toContainEqual({ x: document.points[id]!.x + 8192, z: document.points[id]!.z + 8192 }));
      const [a, b] = original.points.map(id => document.points[id]!);
      const length = generated.reduce((sum, wall) => {
        const [a, b] = wall.points.map(id => result.topology.points[id]!); return sum + Math.hypot(b!.x - a!.x, b!.z - a!.z);
      }, 0);
      expect(length).toBeCloseTo(Math.hypot(b!.x - a!.x, b!.z - a!.z), 7);
      expect(generated.reduce((sum, wall) => sum + wall.stripEnd - wall.stripStart, 0)).toBe(original.stripEnd - original.stripStart);
      for (const wall of generated) expect({ ...wall, points: [], stripStart: 0, stripEnd: 0 }).toEqual({ ...original, points: [], stripStart: 0, stripEnd: 0 });
    }
  }
});

it('retains wall phases and strip density when an existing boundary is lengthened', () => {
  const document = openLevel('deform', originalFiles(1)), room = document.roomSource.rooms[0]!;
  const source = { ...document.roomSource, rooms: [{ ...room, vertices: room.vertices.map(point => ({ x: point.x * 2, z: point.z * 2 })) }] };
  const topology = compileRoomTopology(document, source), result = buildCellWallGeometry(document, source, topology);
  const originalWalls = document.zones[0]!.lower.filter(record => record.wall);
  expect(result.walls.get(0)!.lower).toHaveLength(originalWalls.length);
  result.walls.get(0)!.lower.forEach((item, index) => {
    const wall = readWallSurface(new BinaryReader(item.bytes, 'deformed'), 0), original = originalWalls[index]!.wall!;
    expect(wall.stripEnd - wall.stripStart).toBe(2 * (original.stripEnd - original.stripStart));
    expect(wall.horizontalOffset).toBe(original.horizontalOffset); expect(wall.verticalOffset).toBe(original.verticalOffset);
  });
  expect(result.topology.points.slice(0, topology.points.length)).toEqual(topology.points);
});

it('generates new endpoints for a partial boundary decoration after its room moves', () => {
  const document = openLevel('decoration', originalFiles(1)), room = document.roomSource.rooms.find(room => room.originZone === 90)!;
  const moved = { ...room, vertices: room.vertices.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) };
  const source = { ...document.roomSource, rooms: [moved] }, topology = compileRoomTopology(document, source);
  const result = buildCellWallGeometry(document, source, topology);
  const partial = document.zones[90]!.lower.find(record => record.offset === 15426)!;
  const item = result.walls.get(0)!.lower.find(item => item.sourceOffset === partial.offset)!;
  const wall = readWallSurface(new BinaryReader(item.bytes, 'decoration'), 0);
  expect(wall.points.map(id => result.topology.points[id])).toEqual(partial.points.map(id => ({ x: document.points[id]!.x + 8192, z: document.points[id]!.z + 8192 })));
  expect(wall.stripStart).toBe(partial.wall!.stripStart); expect(wall.stripEnd).toBe(partial.wall!.stripEnd);
  expect(result.topology.points.length).toBeGreaterThan(topology.points.length);
  expect(topology.points.length).toBe(document.points.length + moved.vertices.length);
});

it('translates interior decorations exactly and rejects unsupported reshaping without mutating input', () => {
  const document = openLevel('unattached', originalFiles(2)), room = document.roomSource.rooms.find(room => room.originZone === 150)!;
  const source = { ...document.roomSource, rooms: [{ ...room, vertices: room.vertices.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) }] };
  const translated = buildCellWallGeometry(document, source, compileRoomTopology(document, source));
  const record = document.zones[150]!.lower.find(record => record.offset === 22706)!;
  const item = translated.walls.get(0)!.lower.find(item => item.sourceOffset === record.offset)!;
  const wall = readWallSurface(new BinaryReader(item.bytes, 'interior decoration'), 0);
  expect(wall.points.map(id => translated.topology.points[id])).toEqual(record.points.map(id => ({ x: document.points[id]!.x + 8192, z: document.points[id]!.z + 8192 })));
  const reshaped = { ...source, rooms: source.rooms.map(room => ({ ...room, vertices: room.vertices.map((point, index) => ({ ...point, x: point.x + (index === 1 ? 32 : 0) })) })) };
  const topology = compileRoomTopology(document, reshaped), before = structuredClone(topology);
  expect(() => buildCellWallGeometry(document, reshaped, topology)).toThrow(/not attached to a single boundary/);
  expect(topology).toEqual(before);
  const newSource = { ...source, rooms: source.rooms.map(room => ({ ...room, originZone: null })) };
  expect(() => buildCellWallGeometry(document, newSource, compileRoomTopology(document, newSource))).toThrow(/explicit wall materials/);
});

it('keeps texture columns continuous when one source wall becomes two cell boundary segments', () => {
  const document = openLevel('split', originalFiles(1)), room = document.roomSource.rooms[0]!;
  const a = room.vertices[0]!, b = room.vertices[1]!, midpoint = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 };
  const split = { ...room, vertices: [a, midpoint, ...room.vertices.slice(1)], wallSources: [room.wallSources[0]!, ...room.wallSources] };
  const source = { ...document.roomSource, rooms: [split] }, topology = compileRoomTopology(document, source);
  const result = buildCellWallGeometry(document, source, topology);
  const original = document.zones[0]!.lower.find(record => record.wall && record.points.every(id => {
    const point = document.points[id]!; return (point.x === a.x && point.z === a.z) || (point.x === b.x && point.z === b.z);
  }))!;
  const bound = result.walls.get(0)!.lower.filter(item => item.sourceOffset === original.offset)
    .map(item => readWallSurface(new BinaryReader(item.bytes, 'split'), 0)).sort((a, b) => a.stripStart - b.stripStart);
  expect(bound).toHaveLength(2);
  expect(bound[0]!.stripStart).toBe(original.wall!.stripStart);
  expect(bound[0]!.stripEnd).toBe(bound[1]!.stripStart);
  expect(bound[1]!.stripEnd).toBe(original.wall!.stripEnd);
});

it('builds a concave new room perimeter without placing walls across its internal joins', () => {
  const document = openLevel('new room', originalFiles(1));
  const drawn = commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20064 }, { x: 20064, z: 20064 }, { x: 20064, z: 20256 }, { x: 20000, z: 20256 }])).after;
  const room = drawn.rooms.at(-1)!, source = { ...drawn, rooms: [room] }, topology = compileRoomTopology(document, source);
  const material = document.zones[0]!.lower.find(record => record.wall)!.wall!;
  const result = buildCellWallGeometry(document, source, topology, new Map([[room.id, { lower: [material] }]]));
  expect(topology.cells.length).toBeGreaterThan(1);
  for (const cell of topology.cells) {
    const bound = result.walls.get(cell.id)!.lower.map(item => readWallSurface(new BinaryReader(item.bytes, 'new room'), 0));
    expect(bound).toHaveLength(cell.neighbours.filter(neighbour => neighbour === -1).length);
    cell.neighbours.forEach((neighbour, index) => {
      const a = cell.vertices[index]!, b = cell.vertices[(index + 1) % cell.vertices.length]!;
      expect(bound.some(wall => wall.points[0] === a && wall.points[1] === b)).toBe(neighbour === -1);
    });
  }
});
