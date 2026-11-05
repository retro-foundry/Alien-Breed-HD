import { expect, it } from 'vitest';
import { wallRepeatColumns, restoreWallRepeat, wallRepeatWidths } from '../src/model/wallRepeat';
import { setWallSettings } from '../src/model/wallSettings';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { switchRoomFixture } from './sourceSwitchFixture';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { roomSourceSnapshot, restoreRoomSource, insertSourceCorner, moveSourceCorner } from '../src/model/roomSource';
import { encodeProject, decodeProject } from '../src/model/project';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from './roomHeightFixture';
import { setWallMaterial, setSurfaceMaterial, wallMaterialChoices, validateWallMaterialSampling, restoreWallMaterials } from '../src/model/wallMaterials';
import { restoreWallSettings } from '../src/model/wallSettings';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { setRoomControllerCopies } from '../src/model/movingControllerCopies';
import { importedAnimationTables } from '../src/model/animationBuild';
import { createRoomOpening } from '../src/model/roomOpening';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { encodeWallSurface } from '../src/model/surfaces';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { setCornerLight } from '../src/model/cornerLights';

it.each([
  [128, 'natural', 64], [128, 'up', 128], [128, 'down', 64],
  [130, 'natural', 65], [130, 'up', 128], [130, 'down', 64],
  [32, 'natural', 16], [32, 'up', 64], [32, 'down', 64],
] as const)('ports the original 64-column repeat/fit for length %i, %s', (length, fit, expected) => {
  expect(wallRepeatColumns(length, { width: 64, fit })).toBe(expected);
});

it('validates the original repeat widths, all fit choices, field limits and strict source data', () => {
  for (const width of wallRepeatWidths) for (const fit of ['natural', 'up', 'down'] as const) {
    expect(restoreWallRepeat({ width, fit })).toEqual({ width, fit });
    expect(wallRepeatColumns(width * 2, { width, fit })).toBe(fit === 'up' ? width * 2 : width);
  }
  for (const value of [{ width: 3, fit: 'natural' }, { width: 64, fit: 'guess' }, { width: 64, fit: 'up', extra: true }, { width: 64 }]) expect(() => restoreWallRepeat(value)).toThrow();
  expect(() => wallRepeatColumns(0, { width: 64, fit: 'natural' })).toThrow(/nonzero/);
  expect(() => wallRepeatColumns(65536, { width: 64, fit: 'up' })).toThrow(/texture column count/);
});

it.each(wallRepeatWidths)('emits the native mask and full-repeat strip span for width %i', width => {
  const { document, room } = switchRoomFixture();
  const target = { room: room.id, index: 0, upper: false };
  let source = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width, fit: 'up' } })));
  const style = wallMaterialChoices(document).find(item => {
    try { validateWallMaterialSampling(item.wall!, { ...item.wall!, horizontalMask: width - 1 }, 0); return true; } catch { return false; }
  })!;
  source = applyCommand(source, editRoomSource(setSurfaceMaterial(source, target, style.offset)));
  const built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const identity = built.surfaceSources!.find(item => item.room === cell.id && item.boundaryIndex === 0)!;
  const wall = built.document.zones[cell.id]!.lower.find(item => item.offset === identity.offset)!.wall!;
  expect(wall.horizontalMask).toBe(width - 1); expect(wall.stripEnd).toBe(Math.floor(128 / width) * width + width);
});

it.each([['natural', 128], ['up', 192], ['down', 128]] as const)('compiles %s repeat on a whole authored boundary with a source-only transaction', (fit, expected) => {
  const { document, room } = switchRoomFixture(), target = { room: room.id, index: 0, upper: false };
  const history = execute({ document, past: [], future: [] }, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width: 64, fit } })));
  const source = history.document, built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const identity = built.surfaceSources!.find(item => item.room === cell.id && item.boundaryIndex === 0)!;
  expect(built.document.zones[cell.id]!.lower.find(item => item.offset === identity.offset)!.wall).toMatchObject({ stripStart: 0, stripEnd: expected, horizontalMask: 63 });
  const snapshot = roomSourceSnapshot(source.roomSource); expect(snapshot.version).toBe(19);
  expect(() => restoreRoomSource(document, { ...snapshot, version: 18 })).toThrow(/version 19/);
  expect(decodeProject(encodeProject(source)).roomSource).toEqual(source.roomSource); expect(source.files).toEqual(document.files);
  expect(undo(history).document.roomSource).toBe(document.roomSource); expect(redo(undo(history)).document.roomSource).toBe(source.roomSource);
  const pasted = pasteRooms(document, copyRooms(source, [room.id]), 512, 0), copied = applyCommand(document, pasted.command);
  expect(copied.roomSource.rooms.at(-1)!.wallSettings).toEqual(source.roomSource.rooms.at(-1)!.wallSettings);
});

it('shares an exact repeat column at inserted split endpoints instead of refitting every piece', () => {
  const { document, room } = switchRoomFixture(), target = { room: room.id, index: 0, upper: false };
  let source = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width: 64, fit: 'up' } })));
  source = applyCommand(source, editRoomSource(insertSourceCorner(source.roomSource, room.id, 0, { x: 20000, z: 20096 })));
  const built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const walls = built.surfaceSources!.filter(item => item.room === cell.id && item.boundaryIndex !== undefined && [0, 1].includes(item.boundaryIndex))
    .map(item => built.document.zones[cell.id]!.lower.find(record => record.offset === item.offset)!.wall!).toSorted((a, b) => a.stripStart - b.stripStart);
  expect(walls.map(wall => [wall.stripStart, wall.stripEnd])).toEqual([[0, 72], [72, 192]]);
});

it.each([false, true])('binds imported/copy face repeat to its original native span (upper=%s)', upper => {
  const fixture = nativeRoomCopyFixture(upper ? twoStoreyHeightTemplate() : 0), room = fixture.copy;
  let document = fixture.document;
  const original = (upper ? document.zones[room.recordZone]!.upper : document.zones[room.recordZone]!.lower).find(record => record.wall && record.texture !== 11)!;
  const target = { room: room.id, record: original.offset, upper };
  const style = wallMaterialChoices(document).find(item => {
    try { validateWallMaterialSampling(item.wall!, { ...item.wall!, horizontalMask: 63 }, 0); return true; } catch { return false; }
  })!;
  document = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width: 64, fit: 'up' } })));
  document = applyCommand(document, editRoomSource(setWallMaterial(document, target, style.offset)));
  const built = buildPlayableLevel(document, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const identity = built.surfaceSources!.find(item => item.room === cell.id && item.upper === upper && item.sourceOffset === original.offset)!;
  const wall = (upper ? built.document.zones[cell.id]!.upper : built.document.zones[cell.id]!.lower).find(item => item.offset === identity.offset)!.wall!;
  const [a, b] = original.points.map(id => document.points[id]!);
  expect(wall.stripStart).toBe(0); expect(wall.stripEnd).toBe(wallRepeatColumns(Math.hypot(b!.x - a!.x, b!.z - a!.z), { width: 64, fit: 'up' }));
  expect(wall.horizontalMask).toBe(63);
});

it('uses the complete deformed boundary length and retains repeat settings through opening partitions', () => {
  const { document, room } = switchRoomFixture(), target = { room: room.id, index: 0, upper: false };
  let source = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width: 64, fit: 'up' } })));
  source = applyCommand(source, editRoomSource(insertSourceCorner(source.roomSource, room.id, 0, { x: 20000, z: 20096 })));
  source = applyCommand(source, editRoomSource(moveSourceCorner(source.roomSource, room.id, 1, { x: 19968, z: 20096 })));
  const built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const walls = built.surfaceSources!.filter(item => item.room === cell.id && item.boundaryIndex !== undefined && source.roomSource.rooms.at(-1)!.boundaryMaterials!.edges[item.boundaryIndex] === 0)
    .map(item => built.document.zones[cell.id]!.lower.find(record => record.offset === item.offset)!.wall!);
  expect(Math.min(...walls.map(wall => wall.stripStart))).toBe(0); expect(Math.max(...walls.map(wall => wall.stripEnd))).toBe(192);
  // The unbent source keeps a rectangular opening wholly inside its outline.
  const initial = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width: 64, fit: 'up' } })));
  const proposal = createRoomOpening(initial, room.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 }, initial.roomAppearances![0]!, 'sky');
  const opened = applyCommand(initial, proposal.command);
  expect(opened.roomSource.rooms.filter(room => proposal.rooms.includes(room.id)).some(room => room.wallSettings?.some(item => item.repeat))).toBe(true);
  expect(() => buildPlayableLevel(decodeProject(encodeProject(opened)), -1)).not.toThrow();
});

it('fits a retained decorative native face independently of collision boundaries', () => {
  const baseline = openLevel('Decorative repeat', originalFiles(1)), zone = baseline.zones[0]!, floor = zone.lower.find(item => item.type === 1)!;
  const material = zone.lower.find(item => item.wall && item.texture !== 11)!.wall!;
  const records = zone.lower.map(item => ({ sourceOffset: item.offset, bytes: baseline.files.graphics.slice(item.offset, item.offset + item.size) }));
  const decoration = { ...material, points: [floor.points[0]!, floor.points[2]!] as [number, number] };
  const graphics = buildGraphicsStreams(baseline, undefined, undefined, { streams: new Map([[zone.id, { lower: [...records, { bytes: encodeWallSurface(decoration) }] }]]) });
  const fixture = nativeRoomCopyFixture(0, openLevel(baseline.name, { ...baseline.files, graphics }));
  let document = fixture.document;
  const record = document.zones[0]!.lower.at(-1)!, target = { room: fixture.copy.id, upper: false, record: record.offset };
  document = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width: 64, fit: 'up' } })));
  const built = buildPlayableLevel(document, -1), cell = built.topology!.cells.find(cell => cell.sourceId === fixture.copy.id)!;
  const identity = built.surfaceSources!.find(item => item.room === cell.id && item.sourceOffset === record.offset)!;
  const wall = built.document.zones[cell.id]!.lower.find(item => item.offset === identity.offset)!.wall!;
  const [a, b] = record.points.map(id => document.points[id]!);
  expect(wall.stripEnd).toBe(wallRepeatColumns(Math.hypot(b!.x - a!.x, b!.z - a!.z), { width: 64, fit: 'up' }));
});

it('refits a deformed imported face over its emitted physical pieces rather than its old texture-column span', () => {
  const fixture = nativeRoomCopyFixture(), room = fixture.copy;
  const on = (p: { x: number; z: number }, a: { x: number; z: number }, b: { x: number; z: number }) => (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
    && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);
  const candidates = fixture.document.zones[room.recordZone]!.lower.filter(record => record.wall && record.texture !== 11).flatMap(record => {
    const endpoints = record.points.map(id => ({ x: fixture.document.points[id]!.x + fixture.dx, z: fixture.document.points[id]!.z + fixture.dz }));
    const edge = room.vertices.findIndex((a, index) => {
      const b = room.vertices[(index + 1) % room.vertices.length]!;
      return (a.x + b.x) % 2 === 0 && (a.z + b.z) % 2 === 0 && endpoints.every(point => on(point, a, b));
    });
    return edge < 0 ? [] : [{ record, edge }];
  });
  const { record, edge } = candidates[0]!, a = room.vertices[edge]!, b = room.vertices[(edge + 1) % room.vertices.length]!;
  const middle = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 };
  let source = applyCommand(fixture.document, editRoomSource(setWallSettings(fixture.document, { room: room.id, record: record.offset, upper: false }, { horizontalOffset: 0, repeat: { width: 64, fit: 'up' } })));
  source = applyCommand(source, editRoomSource(insertSourceCorner(source.roomSource, room.id, edge, middle)));
  const changed = source.roomSource.rooms.find(item => item.id === room.id)!, index = changed.vertices.findIndex(point => point.x === middle.x && point.z === middle.z);
  // Imported point lighting is explicit when an inserted point leaves the old
  // straight gradient. This is authored data, retained by the move transaction.
  source = applyCommand(source, editRoomSource(setCornerLight(source, room.id, index, null, { lowerBrightness: 0, upperBrightness: 0 })));
  source = applyCommand(source, editRoomSource(moveSourceCorner(source.roomSource, room.id, index, { x: middle.x + (b.z - a.z === 0 ? 0 : -32), z: middle.z + (b.z - a.z === 0 ? 32 : 0) })));
  const built = buildPlayableLevel(source, -1), cells = new Set(built.topology!.cells.filter(cell => cell.sourceId === room.id).map(cell => cell.id));
  const walls = built.surfaceSources!.filter(item => cells.has(item.room) && item.sourceOffset === record.offset).map(item => built.document.zones[item.room]!.lower.find(record => record.offset === item.offset)!.wall!);
  const spans = new Map(walls.map(wall => [wall.points.join(':'), wall.points.map(id => built.document.points[id]!)]));
  const length = [...spans.values()].reduce((sum, [a, b]) => sum + Math.hypot(b!.x - a!.x, b!.z - a!.z), 0);
  expect(Math.min(...walls.map(wall => wall.stripStart))).toBe(0);
  expect(Math.max(...walls.map(wall => wall.stripEnd))).toBe(wallRepeatColumns(length, { width: 64, fit: 'up' }));
});

it.each(['door', 'lift'] as const)('retains %s controller bases and strips when authoring copied moving-face repeat', kind => {
  const fixture = movingControllerCopyFixture(kind);
  let document = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [fixture.copy])));
  const original = importedAnimationTables(document).tables[kind === 'door' ? 'doors' : 'lifts']![fixture.copy.id]!, graphic = original.walls[0]!.graphic;
  const owner = document.zones.find(zone => [...zone.lower, ...zone.upper].some(item => item.offset === graphic))!;
  const target = { room: fixture.copy.graphicsRooms.get(owner.id)!, record: graphic, upper: owner.upper.some(item => item.offset === graphic) };
  document = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0, repeat: { width: 64, fit: 'up' } })));
  const built = buildPlayableLevel(document, -1).document, copied = importedAnimationTables(built).tables[kind === 'door' ? 'doors' : 'lifts']!.at(-1)!;
  expect(copied.walls).toHaveLength(original.walls.length);
  const record = built.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(item => item.offset === copied.walls[0]!.graphic)!;
  expect(record.wall!.horizontalMask).toBe(63); expect(record.wall!.stripEnd % 64).toBe(0);
  expect(kind === 'door' ? copied.walls[0]!.base >>> 12 : copied.walls[0]!.base >> 16).toBe(0);
});

it.each(Array.from({ length: 16 }, (_, index) => index + 1))('binds repeat/fit for every straight non-switch face in original map %i', level => {
  const original = openLevel(`Repeat corpus ${level}`, originalFiles(level));
  const style = wallMaterialChoices(original).find(item => {
    try { validateWallMaterialSampling(item.wall!, { ...item.wall!, horizontalMask: 63 }, 0); return true; } catch { return false; }
  })!.wall!;
  let faces = 0;
  const rooms = original.roomSource.rooms.map(room => {
    const zone = original.zones[room.recordZone]!, values = ([[false, zone.lower], [true, zone.upper]] as const).flatMap(([upper, records]) =>
      records.filter(record => record.wall && record.texture !== 11).map(record => ({ upper, record: record.offset })));
    faces += values.length;
    return { ...room, ...(values.length ? {
      wallSettings: restoreWallSettings(values.map(item => ({ ...item, horizontalOffset: 0, repeat: { width: 64, fit: 'up' } }))),
      wallMaterials: restoreWallMaterials(values.map(item => ({ ...item, texture: style.texture, verticalMask: style.verticalMask, verticalShift: style.verticalShift, horizontalMask: style.horizontalMask }))),
    } : {}) };
  });
  expect(faces).toBeGreaterThan(0);
  const source = { ...original, roomSource: { ...original.roomSource, rooms } }, built = buildPlayableLevel(source, -1);
  expect(built.document.zones.length).toBeGreaterThanOrEqual(original.zones.length);
  expect(built.document.switches.filter(item => item.zone >= 0)).toHaveLength(original.switches.filter(item => item.zone >= 0).length);
});
