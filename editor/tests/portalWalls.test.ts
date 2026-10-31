import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { closedPortalEdges, hasUpperLayer, portalWallSpans } from '../src/model/portalWalls';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { originalFiles } from './helpers';
import { closedPortalFixture, importedPortalFixture } from './portalFixture';
import { BinaryReader } from '../src/model/binary';
import { buildCellGraphics } from '../src/model/cellGraphics';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildGeometryLighting } from '../src/model/geometryLighting';
import { roomAppearanceBuildOptions, roomAppearanceChoices } from '../src/model/roomAppearance';
import { applyCommand, editRoomAppearance, execute, redo, undo } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';
import { readWallSurface } from '../src/model/surfaces';
import { importedDataTables } from '../src/model/dataBuild';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { removeSourceRooms } from '../src/model/roomSource';

it('cuts destination lower air out of the wall, leaving the authored step and lintel fields intact', () => {
  const document = openLevel('bounds', originalFiles(1)), zone = document.zones[0]!, material = zone.lower.find(record => record.wall)!.wall!;
  const full = { ...material, topFixed: -128 * 256, bottomFixed: 0 };
  const spans = portalWallSpans(document, full, { ...zone, ceiling: -96, floor: -32, upperFloor: 0, upperCeiling: 0 });
  expect(spans).toEqual([{ ...full, bottomFixed: -96 * 256 }, { ...full, topFixed: -32 * 256 }]);
  expect(portalWallSpans(document, full, { ...zone, ceiling: -128, floor: 0, upperFloor: 0, upperCeiling: 0 })).toEqual([]);
  expect(portalWallSpans(document, full, { ...zone, ceiling: 0, floor: 0, upperFloor: 0, upperCeiling: 0 })).toEqual([full]);
});

it('keeps closure texture columns continuous across an inserted corner on the old portal', () => {
  const fixture = closedPortalFixture(), topology = compileRoomTopology(fixture.document, fixture.document.roomSource);
  const cell = topology.cells.find(cell => cell.sourceId === fixture.owner.id)!, edge = closedPortalEdges(fixture.document, cell)[0]!;
  const a = topology.points[cell.vertices[edge]!]!, b = topology.points[cell.vertices[(edge + 1) % cell.vertices.length]!]!;
  const midpoint = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 };
  const index = fixture.owner.vertices.findIndex(point => point.x === a.x && point.z === a.z);
  expect(index).toBeGreaterThanOrEqual(0);
  const room = { ...fixture.owner, vertices: [...fixture.owner.vertices.slice(0, index + 1), midpoint, ...fixture.owner.vertices.slice(index + 1)],
    wallSources: [...fixture.owner.wallSources.slice(0, index + 1), fixture.owner.wallSources[index]!, ...fixture.owner.wallSources.slice(index + 1)] };
  const source = { ...fixture.document, roomSource: { ...fixture.document.roomSource,
    rooms: fixture.document.roomSource.rooms.map(item => item.id === room.id ? room : item) } };
  const document = applyCommand(source, editRoomAppearance(source, fixture.appearance)), settings = roomAppearanceBuildOptions(document);
  const graphics = buildCellGraphics(document, document.roomSource, compileRoomTopology(document, document.roomSource), settings.newRoomGraphics, settings.closingRoomGraphics);
  const fills = graphics.streams.get(cell.id)!.lower.filter(item => item.sourceOffset === undefined)
    .map(item => readWallSurface(new BinaryReader(item.bytes, 'closure segments'), 0));
  expect(fills).toHaveLength(2);
  expect(fills[0]!.stripEnd).toBe(fills[1]!.stripStart);
  expect(fills[0]!.stripStart).not.toBe(fills[1]!.stripStart);
  expect(() => buildPlayableLevel(document, -1)).not.toThrow();
});

it('retains every original surface byte when a closure style is assigned but no portal is closed', () => {
  const fixture = closedPortalFixture(), source = applyCommand(fixture.baseline, editRoomAppearance(fixture.baseline, fixture.appearance));
  const settings = roomAppearanceBuildOptions(source), topology = compileRoomTopology(source, source.roomSource);
  const styled = buildCellGraphics(source, source.roomSource, topology, settings.newRoomGraphics, settings.closingRoomGraphics);
  const original = buildCellGraphics(source, source.roomSource, topology);
  expect(styled).toEqual(original);
});

it('rejects closure of an actual moving volume before losing its animation writes', () => {
  const document = openLevel('moving door closure', originalFiles(1));
  const moving = document.mechanisms.find(item => document.zones[item.zone]?.boundaryIds.some(id => document.lines[id]!.connect >= 0))!;
  expect(moving).toBeDefined();
  const line = document.zones[moving.zone]!.boundaryIds.map(id => document.lines[id]!).find(line => line.connect >= 0)!;
  const target = document.roomSource.rooms.find(room => room.originZone === line.connect)!;
  const source = { ...document, roomSource: removeSourceRooms(document.roomSource, [target.id]).after };
  expect(() => buildSourceLevel(source, { resolvedExit: -1 })).toThrow(/Closing a doorway in a moving door or lift/);
});

it('fills both storeys of a shipped two-floor doorway and rejects an omitted upper style', () => {
  const original = openLevel('upper doorway closure', originalFiles(1));
  const zone = original.zones.find(zone => hasUpperLayer(original, zone) && zone.lower.some(record => record.wall)
    && zone.upper.some(record => record.wall) && zone.boundaryIds.some(id => original.lines[id]!.connect >= 0))!;
  expect(zone).toBeDefined();
  const room = original.roomSource.rooms.find(room => room.originZone === zone.id)!;
  const source = { ...original, roomSource: { ...original.roomSource, rooms: [room] } };
  const choices = roomAppearanceChoices(source, room);
  const document = applyCommand(source, editRoomAppearance(source, { room: room.id, lowerWall: choices.lower[0]!.offset,
    upperWall: choices.upper[0]!.offset, lightingPoint: choices.points[0]! }));
  const settings = roomAppearanceBuildOptions(document), topology = compileRoomTopology(document, document.roomSource);
  const graphics = buildCellGraphics(document, document.roomSource, topology, settings.newRoomGraphics, settings.closingRoomGraphics);
  const cell = topology.cells[0]!, count = closedPortalEdges(document, cell).length, streams = graphics.streams.get(cell.id)!;
  expect(count).toBeGreaterThan(0);
  for (const [upper, records] of [[false, streams.lower], [true, streams.upper!]] as const) {
    const fills = records.filter(item => item.sourceOffset === undefined);
    expect(fills).toHaveLength(count);
    for (const item of fills) {
      const wall = readWallSurface(new BinaryReader(item.bytes, 'two-floor closure'), 0);
      expect(wall.topFixed).toBe((upper ? zone.upperCeiling : zone.ceiling) * 256);
      expect(wall.bottomFixed).toBe((upper ? zone.upperFloor : zone.floor) * 256);
    }
  }
  const selected = settings.closingRoomGraphics.get(room.id)!;
  expect(() => buildCellGraphics(document, document.roomSource, topology, settings.newRoomGraphics,
    new Map([[room.id, { ...selected, walls: { ...selected.walls, upper: [] } }]]))).toThrow(/upper wall style/);
  expect(() => buildCellGraphics(document, document.roomSource, topology, settings.newRoomGraphics,
    new Map([[room.id, { ...selected, walls: { ...selected.walls, lower: [] } }]]))).toThrow(/closed doorway needing a wall style/);
});

it('closes an imported doorway with explicitly selected material and lighting after deleting its neighbour', () => {
  const fixture = closedPortalFixture(), original = structuredClone(fixture.document.files);
  expect(() => buildPlayableLevel(fixture.document, -1)).toThrow(/closed doorway needing a wall style/);
  let history = execute({ document: fixture.document, past: [], future: [] }, editRoomAppearance(fixture.document, fixture.appearance));
  const restored = decodeProject(encodeProject(history.document));
  expect(restored.roomAppearances).toEqual([fixture.appearance]);
  const settings = roomAppearanceBuildOptions(restored), topology = compileRoomTopology(restored, restored.roomSource);
  const cell = topology.cells.find(cell => cell.sourceId === fixture.owner.id)!;
  expect(closedPortalEdges(restored, cell)).toHaveLength(1);
  const graphics = buildCellGraphics(restored, restored.roomSource, topology, settings.newRoomGraphics, settings.closingRoomGraphics);
  const fills = graphics.streams.get(cell.id)!.lower.filter(item => item.sourceOffset === undefined);
  expect(fills).toHaveLength(1); expect(fills[0]!.lightingPoint).toBe(fixture.appearance.lightingPoint);
  const wall = readWallSurface(new BinaryReader(fills[0]!.bytes, 'closing wall'), 0), template = restored.zones[cell.recordZone]!;
  expect(wall.topFixed).toBe(template.ceiling * 256); expect(wall.bottomFixed).toBe(template.floor * 256);
  const built = buildPlayableLevel(restored, -1);
  const room = built.document.zones.find(zone => zone.id === built.topology!.cells.find(cell => cell.sourceId === fixture.owner.id)!.id)!;
  const covering = room.lower.filter(record => record.wall && record.points.every(id => {
    const p = built.document.points[id]!;
    return (p.x - fixture.x) * fixture.normal.x + (p.z - fixture.z) * fixture.normal.z === 0;
  }) && Math.min(...record.points.map(id => built.document.points[id]![fixture.axis])) < (fixture.axis === 'x' ? fixture.x : fixture.z)
    && Math.max(...record.points.map(id => built.document.points[id]![fixture.axis])) > (fixture.axis === 'x' ? fixture.x : fixture.z));
  expect(covering).toHaveLength(1);
  expect(covering[0]).toMatchObject({ top: template.ceiling, bottom: template.floor });
  const generatedPoints = importedDataTables(built.document).points, sourcePoints = importedDataTables(restored).points;
  for (const id of covering[0]!.points) {
    expect(generatedPoints[id]!.lowerBrightness).toBe(sourcePoints[fixture.appearance.lightingPoint]!.lowerBrightness);
    expect(generatedPoints[id]!.upperBrightness).toBe(sourcePoints[fixture.appearance.lightingPoint]!.upperBrightness);
  }
  history = undo(history); expect(() => buildPlayableLevel(history.document, -1)).toThrow(/closed doorway needing a wall style/);
  history = redo(history); expect(buildPlayableLevel(history.document, -1).document.files).toEqual(built.document.files);
  expect(fixture.document.files).toEqual(original);
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
    || readWallSurface(walls[index]!, 0).bottomFixed <= target.ceiling * 256 || readWallSurface(walls[index]!, 0).topFixed >= target.floor * 256)).toBe(true);
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
