import { expect, it } from 'vitest';
import { buildAnimationAttachments } from '../src/model/animationAttachments';
import { buildCellGraphics } from '../src/model/cellGraphics';
import { compileCollisionGeometry } from '../src/model/collisionBuild';
import { buildGeometryLighting } from '../src/model/geometryLighting';
import { openLevel } from '../src/model/level';
import { compileRoomTopology } from '../src/model/roomTopology';
import { originalFiles } from './helpers';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { buildGeneratedGraphics } from '../src/model/generatedGraphics';
import { BinaryReader } from '../src/model/binary';
import { encodePolygonSurface, polygonForCell } from '../src/model/surfaces';
import { readGraphicRecord } from '../src/model/graphics';
import type { CellGraphics } from '../src/model/cellGraphics';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('reattaches level %i animation writes to generated collision and graphics identities', level => {
  const document = openLevel('original', originalFiles(level)), topology = compileRoomTopology(document, document.roomSource);
  const lighting = buildGeometryLighting(document, document.roomSource, buildCellGraphics(document, document.roomSource, topology));
  const collision = compileCollisionGeometry(lighting.graphics.topology);
  const built = buildAnimationAttachments(document, lighting.graphics, collision), original = importedAnimationTables(document).tables;
  expect(built.tables.doors?.length).toBe(original.doors?.length); expect(built.tables.lifts?.length).toBe(original.lifts?.length);
  original.switches?.forEach((record, index) => {
    const generated = built.tables.switches![index]!;
    expect({ ...generated, zone: record.zone }).toEqual(record);
    if (record.zone >= 0) expect(topology.cells.find(cell => cell.id === generated.zone)!.originZone).toBe(record.zone);
    else expect(generated.zone).toBe(record.zone);
  });
  expect(built.addedLighting.size).toBe(0);
  for (const [source, generated] of [[original.doors, built.tables.doors], [original.lifts, built.tables.lifts]] as const) source?.forEach((record, index) => {
    const target = generated![index]!;
    expect(target.header).toEqual(record.header); expect(target.graphic).toBe(record.graphic);
    expect(target.walls).toHaveLength(record.walls.length);
    target.walls.forEach((wall, index) => {
      expect(wall.graphic).toBe(record.walls[index]!.graphic); expect(wall.base).toBe(record.walls[index]!.base);
      const old = document.lines[record.walls[index]!.line]!, view = new DataView(collision.bytes.buffer), offset = wall.line * 16;
      expect([view.getInt16(offset), view.getInt16(offset + 2), view.getInt16(offset + 4), view.getInt16(offset + 6)])
        .toEqual([old.x, old.z, old.dx, old.dz]);
    });
  });
  const source = { ...document.roomSource, rooms: document.roomSource.rooms.map(room => ({ ...room,
    vertices: room.vertices.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) })) };
  const moved = buildGeometryLighting(document, source, buildCellGraphics(document, source, compileRoomTopology(document, source))).graphics;
  const translated = buildAnimationAttachments(document, moved, compileCollisionGeometry(moved.topology));
  expect(translated.addedLighting.size).toBe(2 * (original.switches ?? []).filter(slot => slot.zone >= 0).length);
  original.switches?.forEach((record, index) => {
    const generated = translated.tables.switches![index]!;
    if (record.zone < 0) { expect(generated).toEqual(record); return; }
    expect([generated.point, generated.point + 1].map(id => translated.graphics.topology.points[id]))
      .toEqual([record.point, record.point + 1].map(id => ({ x: document.points[id]!.x + 8192, z: document.points[id]!.z + 8192 })));
    expect(translated.addedLighting.get(generated.point)).toEqual({ kind: 'copy', source: record.point });
  });
});

it('fans out a shared water controller to distinct generated polygon copies', () => {
  const original = openLevel('water', originalFiles(1));
  const selected = original.zones.flatMap(zone => [...zone.lower, ...zone.upper].map(record => ({ zone: zone.id, record }))).find(item => item.record.type === 7)!;
  const tables = importedAnimationTables(original).tables;
  const document = openLevel('water', { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined,
    { animations: { ...tables, doors: null, lifts: [], switches: null, water: tables.water.map((group, index) => ({ ...group,
      entries: index ? [] : [{ zone: selected.zone, graphic: selected.record.offset }] })) } }) });
  const water = document.zones[selected.zone]!.lower.concat(document.zones[selected.zone]!.upper).find(record => record.type === 7)!;
  const imported = compileRoomTopology(document, document.roomSource), template = imported.cells.find(cell => cell.originZone === selected.zone)!;
  const points = [...imported.points, ...[20000, 20256].flatMap(x => [{ x, z: 20000 }, { x, z: 20128 }, { x: x + 128, z: 20128 }, { x: x + 128, z: 20000 }])];
  const cells = [0, 1].map(id => ({ ...template, id, vertices: Array.from({ length: 4 }, (_, index) => imported.points.length + id * 4 + index), neighbours: [-1, -1, -1, -1], sourceLines: [null, null, null, null] }));
  const streams = new Map(cells.map(cell => [cell.id, { lower: [{ sourceOffset: water.offset,
    bytes: encodePolygonSurface(polygonForCell(water.polygon!, cell.vertices, water.height!)) }] }]));
  const graphics: CellGraphics = { topology: { points, cells, sourceCells: new Map([[template.sourceId, [0, 1]]]) }, streams,
    pointReferences: new Map(cells.map(cell => [cell.id, cell.vertices])) };
  const built = buildAnimationAttachments(document, graphics, compileCollisionGeometry(graphics.topology));
  expect(built.tables.water[0]!.entries.map(entry => entry.zone)).toEqual([0, 1]);
  const bytes = buildGeneratedGraphics(document, cells.map(cell => ({ id: cell.id, dataOffset: 100 + cell.id * 48, ...streams.get(cell.id)! })), built.tables, built.selectTarget);
  const reader = new BinaryReader(bytes, 'water output'), entries = reader.i32(4) + 2 + 14;
  expect(reader.i16(entries)).toBe(0); expect(reader.i16(entries + 6)).toBe(1); expect(reader.i16(entries + 12)).toBe(-1);
  const targets = [reader.i32(entries + 2), reader.i32(entries + 8)];
  expect(targets[0]).not.toBe(targets[1]);
  targets.forEach((offset, index) => expect(readGraphicRecord(reader, offset, points.length).points).toEqual(cells[index]!.vertices));
});

it('rejects independent copies of a moving volume rather than changing its controller semantics', () => {
  const document = openLevel('grouping', originalFiles(1)), topology = compileRoomTopology(document, document.roomSource);
  const graphics = buildCellGraphics(document, document.roomSource, topology), door = document.mechanisms.find(item => item.kind === 'door')!;
  const cell = topology.cells.find(cell => cell.originZone === door.zone)!, id = topology.cells.length;
  const duplicated: CellGraphics = { ...graphics, topology: { ...topology, cells: [...topology.cells, { ...cell, id }] },
    streams: new Map([...graphics.streams, [id, graphics.streams.get(cell.id)!]]),
    pointReferences: new Map([...graphics.pointReferences, [id, graphics.pointReferences.get(cell.id)!]]) };
  expect(() => buildAnimationAttachments(document, duplicated, compileCollisionGeometry(duplicated.topology))).toThrow(/Synchronized controller grouping/);
});

it('rotates authored switch activation offsets with their attached surfaces', () => {
  const document = openLevel('rotated switches', originalFiles(1));
  const source = { ...document.roomSource, rooms: document.roomSource.rooms.map(room => ({ ...room,
    vertices: room.vertices.map(point => ({ x: 8192 - point.z, z: 8192 + point.x })) })) };
  const geometry = buildCellGraphics(document, source, compileRoomTopology(document, source));
  const lighting = buildGeometryLighting(document, source, geometry);
  const built = buildAnimationAttachments(document, lighting.graphics, compileCollisionGeometry(lighting.graphics.topology));
  document.switches.forEach((record, index) => {
    if (record.zone < 0) return;
    const generated = built.tables.switches![index]!;
    expect([generated.point, generated.point + 1].map(id => built.graphics.topology.points[id]))
      .toEqual([record.point, record.point + 1].map(id => ({ x: 8192 - document.points[id]!.z, z: 8192 + document.points[id]!.x })));
  });
});
