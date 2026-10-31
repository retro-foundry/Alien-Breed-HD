import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { portalWallSpans } from '../src/model/portalWalls';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { originalFiles } from './helpers';
import { importedPortalFixture } from './portalFixture';
import { BinaryReader } from '../src/model/binary';
import { buildCellGraphics } from '../src/model/cellGraphics';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildGeometryLighting } from '../src/model/geometryLighting';
import { roomAppearanceBuildOptions } from '../src/model/roomAppearance';

it('cuts destination lower air out of the wall, leaving the authored step and lintel fields intact', () => {
  const document = openLevel('bounds', originalFiles(1)), zone = document.zones[0]!, material = zone.lower.find(record => record.wall)!.wall!;
  const full = { ...material, topFixed: -128 * 256, bottomFixed: 0 };
  const spans = portalWallSpans(document, full, { ...zone, ceiling: -96, floor: -32, upperFloor: 0, upperCeiling: 0 });
  expect(spans).toEqual([{ ...full, bottomFixed: -96 * 256 }, { ...full, topFixed: -32 * 256 }]);
  expect(portalWallSpans(document, full, { ...zone, ceiling: -128, floor: 0, upperFloor: 0, upperCeiling: 0 })).toEqual([]);
  expect(portalWallSpans(document, full, { ...zone, ceiling: 0, floor: 0, upperFloor: 0, upperCeiling: 0 })).toEqual([full]);
});

it('retains the middle band between real upper and lower layers and ignores absent upper streams', () => {
  const document = openLevel('two floors', originalFiles(1));
  const reader = new BinaryReader(document.files.graphics, 'layers'), table = reader.i32(12);
  const zone = document.zones.find(zone => reader.i32(table + zone.id * 8 + 4) > 0)!;
  expect(zone).toBeDefined();
  const original = document.zones[0]!.lower.find(record => record.wall)!.wall!;
  const material = { ...original, topFixed: -256 * 256, bottomFixed: 0 };
  const target = { ...zone, ceiling: -96, floor: 0, upperCeiling: -256, upperFloor: -160 };
  expect(portalWallSpans(document, material, target)).toEqual([{ ...material, topFixed: -160 * 256, bottomFixed: -96 * 256 }]);
  const graphics = document.files.graphics.slice(); new DataView(graphics.buffer).setInt32(table + zone.id * 8 + 4, 0);
  expect(portalWallSpans({ ...document, files: { ...document.files, graphics } }, material, target))
    .toEqual([{ ...material, bottomFixed: -96 * 256 }]);
});

it('opens a partial shipped solid wall, preserves horizontal texture phase and compiles valid lighting/references', () => {
  const fixture = importedPortalFixture(), before = structuredClone(fixture.document.files);
  const settings = roomAppearanceBuildOptions(fixture.document), topology = compileRoomTopology(fixture.document, fixture.document.roomSource);
  const graphics = buildCellGraphics(fixture.document, fixture.document.roomSource, topology, settings.newRoomGraphics);
  const owner = topology.sourceCells.get(fixture.room.id)![0]!, inputs = graphics.streams.get(owner)!.lower.filter(item => item.sourceOffset === fixture.material.offset);
  expect(inputs.length).toBeGreaterThanOrEqual(2);
  for (const item of inputs) expect(item.lightingSpan).toBeDefined();
  const walls = inputs.map(item => new BinaryReader(item.bytes, 'wall'));
  const points = walls.map(wall => [graphics.topology.points[wall.i16(2)]!, graphics.topology.points[wall.i16(4)]!] as const);
  const target = fixture.original.zones[0]!;
  const middle = fixture.axis === 'x' ? fixture.x : fixture.z;
  expect(points.every((pair, index) => Math.max(pair[0][fixture.axis], pair[1][fixture.axis]) <= middle - 64 || Math.min(pair[0][fixture.axis], pair[1][fixture.axis]) >= middle + 64
    || walls[index]!.i32(22) <= target.ceiling * 256 || walls[index]!.i32(18) >= target.floor * 256)).toBe(true);
  const material = fixture.material.wall!, a = fixture.original.points[material.points[0]]!, b = fixture.original.points[material.points[1]]!;
  const dx = b.x - a.x, dz = b.z - a.z;
  walls.forEach((wall, index) => points[index]!.forEach((point, end) => {
    expect(wall.i16(end ? 8 : 6)).toBe(material.stripStart + Math.trunc(((point.x - a.x) * dx + (point.z - a.z) * dz) / (dx * dx + dz * dz) * (material.stripEnd - material.stripStart)));
  }));
  expect(() => buildGeometryLighting(fixture.document, fixture.document.roomSource, graphics, settings.newRoomLighting)).not.toThrow();
  const built = buildPlayableLevel(fixture.document, -1);
  expect(built.document.zones.length).toBe(built.topology!.cells.length);
  expect(built.topology!.cells.some(cell => cell.sourceId === fixture.room.id && cell.neighbours.some(id => id >= 0
    && built.topology!.cells.find(target => target.id === id)!.sourceId === fixture.added.id))).toBe(true);
  expect(fixture.document.files).toEqual(before);
});
