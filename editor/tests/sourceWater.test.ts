import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, redo, undo } from '../src/model/commands';
import { setRoomWater, setWaterGroup, restoreWaterGroups, restoreWaterSurface, waterGroupSettings, waterMaterialChoices } from '../src/model/sourceWater';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject, encodeGameLevel, openArchive } from '../src/model/project';
import { openLevel } from '../src/model/level';
import { roomSettings } from '../src/model/roomHeights';
import { roomHeightFixture } from './roomHeightFixture';
import { sourceWaterFixture } from './sourceWaterFixture';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { nativeWaterRoomCopyFixture } from './nativeRoomCopyFixture';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, roomSourceSnapshot, setDraftCorners } from '../src/model/roomSource';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';

it('creates synchronized independent water in two new rooms with native plane, collision and animation targets', () => {
  const { document, from, to, group } = sourceWaterFixture(), built = buildPlayableLevel(document, -1);
  const animation = importedAnimationTables(built.document).tables.water[group.id]!;
  expect(animation.entries).toHaveLength(2);
  for (const id of [from.id, to.id]) {
    const cell = built.topology!.cells.find(cell => cell.sourceId === id)!, zone = built.document.zones.find(zone => zone.id === cell.id)!;
    expect(zone.water).toBe(group.current);
    const polygon = zone.lower.find(record => record.type === 7)!;
    expect(polygon.height).toBe(group.current);
    expect(animation.entries).toContainEqual({ zone: cell.id, graphic: polygon.offset });
    expect(zone.lower.filter(record => record.type === 4)).toHaveLength(2);
  }
  expect(new Set(animation.entries.map(entry => entry.graphic)).size).toBe(2);
  expect(decodeProject(encodeProject(document)).roomSource).toEqual(document.roomSource);
  expect(roomSourceSnapshot(document.roomSource).version).toBe(16);
  const game = openArchive(encodeGameLevel(built.document), 'water.zip')[0]!;
  expect(importedAnimationTables(openLevel(game.name, game.files)).tables.water[group.id]).toEqual(animation);
});

it('supports static water, pauses whole groups, detaches rooms and removes both polygon and collision water', () => {
  const { document, from, to, group } = sourceWaterFixture();
  let source = applyCommand(document, setWaterGroup(document, { ...group, enabled: false, current: group.top }));
  let built = buildPlayableLevel(source, -1);
  expect(importedAnimationTables(built.document).tables.water[group.id]!.entries).toHaveLength(0);
  for (const id of [from.id, to.id]) expect(built.document.zones.find(zone => zone.id === built.topology!.cells.find(cell => cell.sourceId === id)!.id)!.water).toBe(group.top);
  source = applyCommand(source, setRoomWater(source, from.id, { material: waterMaterialChoices(source, from)[0]!.offset, height: group.current, group: null }));
  source = applyCommand(source, setRoomWater(source, to.id, false)); built = buildPlayableLevel(source, -1);
  const fromZone = built.document.zones[built.topology!.cells.find(cell => cell.sourceId === from.id)!.id]!, toZone = built.document.zones[built.topology!.cells.find(cell => cell.sourceId === to.id)!.id]!;
  expect(fromZone.water).toBe(group.current); expect(fromZone.lower.some(record => record.type === 7)).toBe(true);
  expect(toZone.water).toBe(toZone.floor + 4); expect(toZone.lower.some(record => record.type === 7)).toBe(false);
  expect(roomSettings(source, source.roomSource.rooms.find(room => room.id === to.id)!).water).toBe(toZone.water);
});

it('reassigns imported animated water without replacing polygon shapes and can stop the shared original/copy group', () => {
  const fixture = nativeWaterRoomCopyFixture(), group = waterGroupSettings(fixture.document, fixture.group);
  const source = applyCommand(fixture.document, setWaterGroup(fixture.document, { ...group, current: group.top, enabled: false }));
  const built = buildPlayableLevel(source, -1), original = built.topology!.cells.find(cell => cell.originZone === fixture.zone)!;
  const native = built.document.zones[original.id]!, template = fixture.document.zones[fixture.zone]!;
  expect(native.lower.filter(record => record.type === 7).map(record => record.points.length)).toEqual(template.lower.filter(record => record.type === 7).map(record => record.points.length));
  expect(native.lower.find(record => record.type === 7)!.height).toBe(group.top); expect(native.water).toBe(group.top);
  expect(importedAnimationTables(built.document).tables.water[fixture.group]!.entries).toHaveLength(0);
  const room = fixture.document.roomSource.rooms.find(room => room.originZone === fixture.zone)!;
  const alternate = { ...group, id: 9, enabled: true };
  let reassigned = applyCommand(fixture.document, setWaterGroup(fixture.document, alternate));
  reassigned = applyCommand(reassigned, setRoomWater(reassigned, room.id, { material: null, height: group.current, group: 9 }));
  const moved = buildPlayableLevel(reassigned, -1), movedTables = importedAnimationTables(moved.document).tables;
  expect(movedTables.water[9]!.entries.some(entry => entry.zone === original.id)).toBe(true);
  expect(movedTables.water[fixture.group]!.entries.some(entry => entry.zone === original.id)).toBe(false);
  expect(movedTables.water[fixture.group]!.entries.length).toBeGreaterThan(0); // copied room keeps its original assignment
});

it('adds the mandatory lift terminator and 21 slots when authoring water in a level with no animation tables', () => {
  const { original } = roomHeightFixture(), tables = importedAnimationTables(original).tables;
  const empty = openLevel('No animation tables', { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined, { animations: { ...tables, doors: null, lifts: null, switches: null, water: [] } }) });
  const fixture = roomHeightFixture(0, empty), settings = roomSettings(fixture.document, fixture.from), group = { id: 20, top: settings.floor - 40, bottom: settings.floor - 8, current: settings.floor - 16, velocity: -128, enabled: true };
  let source = applyCommand(fixture.document, setWaterGroup(fixture.document, group));
  source = applyCommand(source, setRoomWater(source, fixture.from.id, { material: waterMaterialChoices(source, fixture.from)[0]!.offset, height: group.current, group: 20 }));
  const built = importedAnimationTables(buildPlayableLevel(source, -1).document).tables;
  expect(built.lifts).toEqual([]); expect(built.water).toHaveLength(21); expect(built.water[20]!.entries).toHaveLength(1);
});

it('retains water assignments through room copies, concave decomposition, deletion and stale-checked Undo', () => {
  const { document, from, group } = sourceWaterFixture();
  const result = pasteRooms(document, copyRooms(document, [from.id]), -10000, 0), pasted = applyCommand(document, result.command);
  expect(pasted.roomSource.rooms.find(room => room.id === result.ids[0])!.waterSurface).toEqual(document.roomSource.rooms.find(room => room.id === from.id)!.waterSurface);
  expect(importedAnimationTables(buildPlayableLevel(pasted, -1).document).tables.water[group.id]!.entries).toHaveLength(3);
  const changed = applyCommand(pasted, editRoomSource(commitRoomDraft(pasted.roomSource, setDraftCorners(beginRoomDraft(pasted, pasted.roomSource, from.recordZone, from.id), [
    { x: 20000, z: 20000 }, { x: 20400, z: 20000 }, { x: 20400, z: 20128 }, { x: 20128, z: 20128 }, { x: 20128, z: 20400 }, { x: 20000, z: 20400 },
  ]))));
  const concave = buildPlayableLevel(changed, -1), cells = concave.topology!.cells.filter(cell => cell.sourceId === from.id), water = importedAnimationTables(concave.document).tables.water[group.id]!.entries;
  expect(cells.length).toBeGreaterThan(1);
  for (const cell of cells) expect(water.filter(entry => entry.zone === cell.id)).toHaveLength(2);
  const command = setRoomWater(document, from.id, false), history = execute({ document, past: [], future: [] }, command);
  expect(undo(history).document.roomSource).toBe(document.roomSource); expect(redo(undo(history)).document.roomSource).toBe(history.document.roomSource);
  expect(() => applyCommand(history.document, command)).toThrow(/room source changed/);
  const deleted = applyCommand(pasted, editRoomSource(removeSourceRooms(pasted.roomSource, [result.ids[0]!])));
  expect(importedAnimationTables(buildPlayableLevel(deleted, -1).document).tables.water[group.id]!.entries).toHaveLength(2);
});

it('rejects malformed groups, missing materials and unconfigured slot assignments without partial edits', () => {
  const { document, from, group } = sourceWaterFixture(), original = roomHeightFixture();
  expect(() => restoreWaterGroups([{ ...group, id: 21 }])).toThrow(/group/);
  expect(() => restoreWaterGroups([group, group])).toThrow(/Duplicate/);
  expect(() => setWaterGroup(document, { ...group, current: group.bottom + 1 })).toThrow(/inside that range/);
  expect(() => restoreWaterSurface({ material: null, group: 0, height: 1 / 256 })).toThrow(/quarter units/);
  expect(() => restoreWaterSurface({ material: null, group: 0, height: null })).toThrow(/must be a number/);
  expect(() => restoreWaterGroups([{ ...group, top: String(group.top) }])).toThrow(/must be a number/);
  expect(() => setRoomWater(document, from.id, { material: 0xffffff, height: group.current, group: null })).toThrow(/pattern/);
  expect(() => setRoomWater(original.document, original.from.id, { material: waterMaterialChoices(original.document, original.from)[0]!.offset, height: 0, group: 20 })).toThrow(/configure/);
  const missing = { ...document, roomSource: { ...document.roomSource, waterGroups: [] } };
  expect(() => buildPlayableLevel(decodeProject(encodeProject(missing)), -1)).toThrow(/configure water group/);
});

it('retains native 24.8 phase precision and uses arithmetic-shift rounding for water polygon heights', () => {
  const { document, from, group } = sourceWaterFixture(), current = group.current + 1 / 256;
  const source = applyCommand(document, setWaterGroup(document, { ...group, current, velocity: 1 })), built = buildPlayableLevel(source, -1);
  const zone = built.document.zones[built.topology!.cells.find(cell => cell.sourceId === from.id)!.id]!;
  expect(zone.water).toBe(current); expect(zone.lower.find(record => record.type === 7)!.height).toBe(Math.floor(current * 4) / 4);
  const header = importedAnimationTables(built.document).tables.water[group.id]!.header, view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  expect(view.getInt32(8)).toBe(current * 256); expect(view.getInt16(12)).toBe(1);
});

it('restores a captured water group when pasting into the same workspace before its creation', () => {
  const { document, from, group } = sourceWaterFixture();
  const prior = { ...document, roomSource: { ...document.roomSource, waterGroups: undefined, rooms: document.roomSource.rooms.map(room => { const { waterSurface: _water, ...rest } = room; return rest; }) } };
  const result = pasteRooms(prior, copyRooms(document, [from.id]), -10000, 0), pasted = applyCommand(prior, result.command);
  expect(pasted.roomSource.waterGroups).toEqual([group]);
  expect(importedAnimationTables(buildPlayableLevel(pasted, -1).document).tables.water[group.id]!.entries).toHaveLength(1);
});
