import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, undo, redo, numberPatch } from '../src/model/commands';
import { setWallSettings, wallSettingsCurrent, restoreWallSettings } from '../src/model/wallSettings';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from './roomHeightFixture';
import { encodeProject, decodeProject } from '../src/model/project';
import { roomSourceSnapshot, translateSourceRooms, insertSourceCorner } from '../src/model/roomSource';
import { setSurfaceMaterial, setWallMaterial, wallMaterialChoices, validateWallMaterialSampling } from '../src/model/wallMaterials';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { switchRoomFixture } from './sourceSwitchFixture';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { setRoomControllerCopies } from '../src/model/movingControllerCopies';
import { importedAnimationTables } from '../src/model/animationBuild';
import { createRoomOpening } from '../src/model/roomOpening';
import { clearRoomWallSettings } from '../src/model/wallSettings';

for (const upper of [false, true]) it(`independently edits ${upper ? 'upper' : 'lower'} face alignment and brightness, with save and Undo`, () => {
  const fixture = nativeRoomCopyFixture(upper ? twoStoreyHeightTemplate() : 0), room = fixture.copy;
  let document = fixture.document;
  const record = (upper ? document.zones[room.recordZone]!.upper : document.zones[room.recordZone]!.lower).find(record => record.wall && record.texture !== 11)!;
  const target = { room: room.id, record: record.offset, upper };
  const style = wallMaterialChoices(document).find(item => {
    try { validateWallMaterialSampling(item.wall!, item.wall!, 0); return true; } catch { return false; }
  })!;
  document = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 0 })));
  document = applyCommand(document, editRoomSource(setWallMaterial(document, target, style.offset)));
  const history = execute({ document, past: [], future: [] }, editRoomSource(setWallSettings(document, target, { brightness: 18, horizontalOffset: 0, verticalOffset: 7 })));
  const saved = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(saved, -1);
  expect(roomSourceSnapshot(saved.roomSource).version).toBe(18); expect(saved.files).toEqual(document.files);
  const cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const identity = built.surfaceSources!.find(item => item.room === cell.id && item.upper === upper && item.sourceOffset === record.offset)!;
  const wall = (upper ? built.document.zones[cell.id]!.upper : built.document.zones[cell.id]!.lower).find(item => item.offset === identity.offset)!.wall!;
  expect(wall).toMatchObject({ brightness: 18, horizontalOffset: 0, verticalOffset: 7, texture: style.texture });
  expect(undo(history).document.roomSource).toBe(document.roomSource); expect(redo(undo(history)).document.roomSource).toBe(history.document.roomSource);
  const moved = applyCommand(saved, editRoomSource(translateSourceRooms(saved.roomSource, [room.id], 16, 0)));
  expect(wallSettingsCurrent(moved, target)).toEqual(wallSettingsCurrent(saved, target));
  expect(() => buildPlayableLevel(moved, -1)).not.toThrow();
  const pasted = pasteRooms(document, copyRooms(history.document, [room.id]), -8192, 0), copy = applyCommand(document, pasted.command);
  expect(copy.roomSource.rooms.at(-1)!.wallSettings).toEqual(saved.roomSource.rooms.find(item => item.id === room.id)!.wallSettings);
  const reset = applyCommand(saved, editRoomSource(setWallSettings(saved, target, null)));
  expect(reset.roomSource.rooms.find(item => item.id === room.id)!.wallSettings).toBeUndefined();
});

it('keeps boundary settings through inserted corners and independent material resets', () => {
  const fixture = switchRoomFixture(), document = fixture.document, room = fixture.room, target = { room: room.id, index: 0, upper: false };
  let source = applyCommand(document, editRoomSource(setWallSettings(document, target, { brightness: 12, horizontalOffset: 0, verticalOffset: -3 })));
  const style = wallMaterialChoices(document).find(item => item.texture !== 11)!;
  source = applyCommand(source, editRoomSource(setSurfaceMaterial(source, target, style.offset)));
  source = applyCommand(source, editRoomSource(setSurfaceMaterial(source, target, null)));
  source = applyCommand(source, editRoomSource(insertSourceCorner(source.roomSource, room.id, 0, { x: 20000, z: 20128 })));
  const built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const walls = built.surfaceSources!.filter(item => item.room === cell.id && item.boundaryIndex !== undefined && [0, 1].includes(item.boundaryIndex));
  expect(walls.length).toBeGreaterThan(0);
  for (const item of walls) expect(built.document.zones[cell.id]!.lower.find(record => record.offset === item.offset)!.wall).toMatchObject({ brightness: 12, horizontalOffset: 0, verticalOffset: -3 });
  expect(() => restoreWallSettings([{ upper: false, record: 1, brightness: 32768 }])).toThrow();
  expect(() => restoreWallSettings([{ upper: false, record: 1 }])).toThrow();
  expect(() => restoreWallSettings([{ upper: false, record: 1, boundary: 1, brightness: 0 }])).toThrow();
  expect(() => restoreWallSettings([{ upper: false, record: 1, brightness: 0, unknown: 0 }])).toThrow();
});

for (const kind of ['door', 'lift'] as const) it(`synchronizes independently copied ${kind} wall phase bases without changing its rules or collision links`, () => {
  const fixture = movingControllerCopyFixture(kind);
  const document = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [fixture.copy])));
  const tables = importedAnimationTables(document).tables, controller = tables[kind === 'door' ? 'doors' : 'lifts']![fixture.copy.id]!;
  const entry = controller.walls[0]!, owner = document.zones.find(zone => [...zone.lower, ...zone.upper].some(record => record.offset === entry.graphic))!;
  const id = fixture.copy.graphicsRooms.get(owner.id)!, record = [...owner.lower, ...owner.upper].find(record => record.offset === entry.graphic)!;
  const target = { room: id, record: entry.graphic, upper: owner.upper.includes(record) };
  const source = applyCommand(document, editRoomSource(setWallSettings(document, target, { brightness: -4, horizontalOffset: 0, verticalOffset: 9 })));
  const baseline = importedAnimationTables(buildPlayableLevel(document, -1).document).tables[kind === 'door' ? 'doors' : 'lifts']!.at(-1)!;
  const built = buildPlayableLevel(source, -1).document, copied = importedAnimationTables(built).tables[kind === 'door' ? 'doors' : 'lifts']!.at(-1)!;
  expect(copied.header).toEqual(baseline.header);
  const mapping = copied.walls.find(item => {
    const wall = built.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === item.graphic)?.wall;
    return wall?.brightness === -4;
  })!;
  expect(mapping).toBeDefined();
  expect(kind === 'door' ? mapping.base >>> 12 : mapping.base >> 16).toBe(0);
  const current = new DataView(baseline.header.buffer, baseline.header.byteOffset).getInt16(4) / 4;
  expect(kind === 'door' ? (mapping.base + ((-Math.floor(current)) & 255)) & 255 : mapping.base & 65535).toBe(9);
  expect(copied.walls.length).toBe(controller.walls.length);
  expect(built.mechanisms.filter(item => item.kind === kind)).toHaveLength(2);
  expect(importedAnimationTables(built).tables[kind === 'door' ? 'doors' : 'lifts']![0]!.walls.map(item => item.base)).toEqual(controller.walls.map(item => item.base));
});

it('partitions boundary settings onto surviving walls and recovers removed-wall settings from the room inspector', () => {
  const { document, room } = switchRoomFixture(), target = { room: room.id, index: 0, upper: false };
  const source = applyCommand(document, editRoomSource(setWallSettings(document, target, { brightness: 15, verticalOffset: -7 })));
  const proposal = createRoomOpening(source, room.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 }, source.roomAppearances![0]!, 'sky');
  const opened = applyCommand(source, proposal.command);
  expect(opened.roomSource.rooms.filter(room => proposal.rooms.includes(room.id)).some(room => room.wallSettings?.length)).toBe(true);
  expect(() => buildPlayableLevel(decodeProject(encodeProject(opened)), -1)).not.toThrow();
  const damaged = { ...source, roomSource: { ...source.roomSource, rooms: source.roomSource.rooms.map(item => item.id === room.id
    ? { ...item, wallSettings: [{ boundary: 999, upper: false, brightness: 7 }] } : item) } };
  expect(() => buildPlayableLevel(damaged, -1)).toThrow(/reset room wall settings/);
  const clear = editRoomSource(clearRoomWallSettings(damaged, room.id)), repaired = applyCommand(damaged, clear);
  expect(() => buildPlayableLevel(repaired, -1)).not.toThrow(); expect(applyCommand(repaired, clear, true).roomSource).toBe(damaged.roomSource);
});

it('preserves a signed legacy door base tile when editing only the row phase', () => {
  const fixture = movingControllerCopyFixture('door');
  let document = applyCommand(fixture.document, { label: 'Native signed door base fixture', patches: [numberPatch(fixture.document, 'graphics', fixture.mechanism.offset + 24, -16777216, 4, true)] });
  document = applyCommand(document, editRoomSource(setRoomControllerCopies(document, [fixture.copy])));
  const record = importedAnimationTables(document).tables.doors![fixture.copy.id]!, wall = record.walls[0]!;
  const owner = document.zones.find(zone => [...zone.lower, ...zone.upper].some(item => item.offset === wall.graphic))!;
  const target = { room: fixture.copy.graphicsRooms.get(owner.id)!, record: wall.graphic, upper: owner.upper.some(item => item.offset === wall.graphic) };
  const source = applyCommand(document, editRoomSource(setWallSettings(document, target, { verticalOffset: 9 })));
  const copied = importedAnimationTables(buildPlayableLevel(source, -1).document).tables.doors!.at(-1)!, changed = copied.walls[0]!;
  expect(changed.base).toBeLessThan(0); expect(changed.base >>> 12).toBe(wall.base >>> 12);
  const current = new DataView(copied.header.buffer, copied.header.byteOffset).getInt16(4) / 4;
  expect((changed.base + ((-Math.floor(current)) & 255)) & 255).toBe(9);
});
