import { describe, expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { normaliseRoomOutline } from '../src/model/geometry';
import { openLevel } from '../src/model/level';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildCellPolygons, wallForSpan } from '../src/model/surfaceGeometry';
import { encodePolygonSurface, encodeWallSurface, readPolygonSurface } from '../src/model/surfaces';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { originalFiles } from './helpers';

describe('source cell polygon material binding', () => {
  it('writes new point identities even when the room coordinates themselves are unchanged', () => {
    const document = openLevel('reindexed points', originalFiles(1)), imported = compileRoomTopology(document, document.roomSource), original = imported.cells[0]!;
    const points = original.vertices.map(id => imported.points[id]!), vertices = points.map((_, i) => i);
    const topology = { points, cells: [{ ...original, vertices }], sourceCells: new Map([[original.sourceId, [0]]]) };
    const built = buildCellPolygons(document, topology).get(0)!;
    for (const entry of [...built.lower, ...built.upper]) {
      const polygon = readPolygonSurface(new BinaryReader(entry.bytes, 'reindexed surface'), 0);
      expect(polygon.points).toEqual(vertices); expect(polygon.closingPoint).toBe(0);
    }
  });
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('binds level %i polygons to convex cells and retains unchanged records exactly', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level)), topology = compileRoomTopology(document, document.roomSource);
    const built = buildCellPolygons(document, topology);
    for (const zone of document.zones) for (const record of [...zone.lower, ...zone.upper]) if (record.wall) {
      const [a, b] = record.wall.points.map(id => document.points[id]!);
      const length = Math.hypot(b!.x - a!.x, b!.z - a!.z);
      expect(encodeWallSurface(wallForSpan(record.wall, length, record.wall.points, 0, length)))
        .toEqual(document.files.graphics.subarray(record.offset, record.offset + record.size));
    }
    expect(built.size).toBe(topology.cells.length);
    for (const cell of topology.cells) {
      const zone = document.zones[cell.recordZone]!;
      const original = normaliseRoomOutline(zone.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })));
      const unchanged = JSON.stringify(normaliseRoomOutline(cell.vertices.map(id => topology.points[id]!))) === JSON.stringify(original);
      for (const [records, generated] of [[zone.lower, built.get(cell.id)!.lower], [zone.upper, built.get(cell.id)!.upper]] as const) {
        const polygons = records.filter(record => record.polygon);
        expect(generated).toHaveLength(polygons.length);
        generated.forEach((entry, i) => {
          const record = polygons[i]!, polygon = readPolygonSurface(new BinaryReader(entry.bytes, 'compiled polygon'), 0);
          expect(entry.sourceOffset).toBe(record.offset);
          expect({ ...polygon, points: [], closingPoint: 0 }).toEqual({ ...record.polygon!, points: [], closingPoint: 0 });
          if (unchanged) expect(entry.bytes).toEqual(document.files.graphics.subarray(record.offset, record.offset + record.size));
          else { expect(polygon.points).toEqual(cell.vertices); expect(polygon.closingPoint).toBe(cell.vertices[0]); }
        });
      }
    }
    // Force actual regeneration for every shipped material template, rather
    // than proving only unchanged-record retention on most original rooms.
    const translated = { ...topology, points: topology.points.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) };
    const relocated = buildCellPolygons(document, translated);
    for (const cell of translated.cells) for (const record of [...relocated.get(cell.id)!.lower, ...relocated.get(cell.id)!.upper]) {
      const polygon = readPolygonSurface(new BinaryReader(record.bytes, 'relocated material'), 0);
      expect(polygon.points).toEqual(cell.vertices);
      expect(polygon.closingPoint).toBe(cell.vertices[0]);
    }
  });

  it('keeps split texture coordinates continuous and retains material phases, masks and heights', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const original = document.zones.flatMap(zone => zone.lower).find(record => record.wall)!.wall!;
    const before = structuredClone(original), template = { ...original, stripStart: 17, stripEnd: 83 };
    const first = wallForSpan(template, 128, [100, 101], 0, 64), second = wallForSpan(template, 128, [101, 102], 64, 128);
    expect(first).toEqual({ ...template, points: [100, 101], stripEnd: 50 });
    expect(second).toEqual({ ...template, points: [101, 102], stripStart: 50 });
    expect(wallForSpan(template, 128, [100, 102], 0, 256).stripEnd).toBe(149);
    expect(() => wallForSpan(template, 0, [100, 101], 0, 64)).toThrow(/nonzero/);
    expect(() => wallForSpan(template, 128, [100, 100], 0, 64)).toThrow(/different endpoints/);
    expect(() => wallForSpan(template, 128, [100, 2048], 0, 64)).toThrow(/point index/);
    expect(() => wallForSpan(template, 128, [100, 101], 64, 0)).toThrow(/increasing/);
    expect(() => wallForSpan(template, 128, [100, 101], 0, 100000)).toThrow(/texture column/);
    expect(original).toEqual(before);
  });

  it('retains original partial surfaces but refuses to silently expand them across changed geometry', () => {
    const original = openLevel('partial floor', originalFiles(1)), floor = original.zones[0]!.lower.find(record => record.type === 1)!;
    const vertices = [floor.points[0]!, floor.points[Math.floor(floor.points.length / 2)]!, floor.points.at(-1)!];
    const bytes = encodePolygonSurface({ ...floor.polygon!, points: vertices, closingPoint: vertices[0]! });
    const document = openLevel('partial floor', { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined,
      { replacements: new Map([[floor.offset, bytes]]) }) });
    const imported = compileRoomTopology(document, document.roomSource);
    expect(buildCellPolygons(document, imported).get(0)!.lower.find(record => record.sourceOffset === floor.offset)!.bytes).toEqual(bytes);
    const source = commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0),
      [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }])).after;
    const before = document.files.graphics.slice();
    expect(() => buildCellPolygons(document, compileRoomTopology(document, source))).toThrow(/partial or degenerate footprint/);
    expect(document.files.graphics).toEqual(before);
    expect(() => buildCellPolygons(document, { ...imported, cells: [imported.cells[0]!, imported.cells[0]!] })).toThrow(/duplicate/);
    expect(() => buildCellPolygons(document, { ...imported, cells: [{ ...imported.cells[0]!, recordZone: -1 }] })).toThrow(/settings template/);
    expect(() => buildCellPolygons(document, { ...imported, cells: [{ ...imported.cells[0]!, vertices: [99999, 0, 1] }] })).toThrow(/missing surface corner/);
  });

  it('generates every convex piece of a new concave room from its selected floor and roof records', () => {
    const document = openLevel('Level 1', originalFiles(1)), vertices = [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 },
      { x: 20256, z: 20064 }, { x: 20064, z: 20064 }, { x: 20064, z: 20256 }, { x: 20000, z: 20256 }];
    const source = commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0), vertices)).after;
    const topology = compileRoomTopology(document, source), polygons = buildCellPolygons(document, topology);
    const cells = topology.cells.filter(cell => cell.originZone === null);
    expect(cells.length).toBeGreaterThan(1);
    for (const cell of cells) {
      const surfaces = polygons.get(cell.id)!.lower.map(entry => readPolygonSurface(new BinaryReader(entry.bytes, 'new room'), 0));
      expect(surfaces.some(surface => surface.type === 1)).toBe(true);
      expect(surfaces.some(surface => surface.type === 2)).toBe(true);
      expect(surfaces.every(surface => JSON.stringify(surface.points) === JSON.stringify(cell.vertices))).toBe(true);
    }
  });
});
