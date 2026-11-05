import { pointInRoomOutline } from '../src/model/geometry';
import { expect, it } from 'vitest';
import { switchCopyPlayerFixture } from './switchCopyFixture';
import { switchRoomFixture } from './sourceSwitchFixture';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { reassignCopiedSwitch, moveSourceSwitch, setSourceSwitch, availableSwitchSlots } from '../src/model/sourceSwitches';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { importedAnimationTables } from '../src/model/animationBuild';
import { setRoomSwitchCopies } from '../src/model/switchCopies';
import { encodeProject, decodeProject } from '../src/model/project';
import { roomSourceSnapshot, restoreRoomSource, insertSourceCorner, translateSourceRooms } from '../src/model/roomSource';
import { copyRooms, copyRoomGroup, pasteRooms } from '../src/model/roomClipboard';
import { switchSourceRoom } from '../src/model/nativeSwitches';
import { copiedSwitchAuthoringFixture } from './copiedSwitchAuthoringFixture';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';

it('moves a copied native switch into an authored panel, retaining its condition number, raw settings and independent original', () => {
  const fixture = switchCopyPlayerFixture(), fresh = switchRoomFixture(fixture.document), slot = fixture.switchCopy.slot;
  const before = buildPlayableLevel(fresh.document, -1).document.switches[fixture.source.id]!;
  const history = execute({ document: fresh.document, past: [], future: [] }, editRoomSource(reassignCopiedSwitch(fresh.document, slot, { room: fresh.room.id, edge: 0 })));
  const source = history.document, built = buildPlayableLevel(source, -1), item = built.document.switches[slot]!;
  expect(source.roomSource.switchCopies).toBeUndefined();
  expect(source.roomSource.rooms.find(room => room.id === fixture.switchCopy.graphicsRoom)!.retiredCopiedSwitches).toEqual([fixture.source.id]);
  expect(source.roomSource.rooms.at(-1)!.switches![0]).toMatchObject({ slot, native: fixture.source.id, settings: fixture.document.roomSource.switchCopies![0]!.bytes });
  expect(switchSourceRoom(source, slot)).toBe(fresh.room.id);
  expect(built.topology!.cells.find(cell => cell.id === item.zone)!.sourceId).toBe(fresh.room.id);
  expect(built.surfaceSources!.filter(record => record.sourceOffset === fixture.source.graphic - 10)).toHaveLength(1);
  expect(built.document.switches[fixture.source.id]).toMatchObject({ type: before.type, countdown: before.countdown, state: before.state });
  expect(decodeProject(encodeProject(source)).roomSource).toEqual(source.roomSource);
  expect(roomSourceSnapshot(source.roomSource).version).toBe(23);
  expect(undo(history).document.roomSource).toBe(fresh.document.roomSource);
  expect(redo(undo(history)).document.roomSource).toBe(source.roomSource);
});

it('removes a running copied switch, clears its runtime slot and panel, and permits fresh slot reuse', () => {
  const fixture = switchCopyPlayerFixture(), slot = fixture.switchCopy.slot, original = fixture.source;
  const captured = fixture.document.roomSource.switchCopies![0]!, bytes = [...captured.bytes];
  bytes[2] = 7; bytes[3] = 91; bytes.splice(10, 4, 255, 18, 52, 86);
  const document = applyCommand(fixture.document, editRoomSource(setRoomSwitchCopies(fixture.document, [{ ...fixture.switchCopy, bytes: new Uint8Array(bytes) }])));
  const source = applyCommand(document, editRoomSource(reassignCopiedSwitch(document, slot, null))), built = buildPlayableLevel(source, -1);
  const record = importedAnimationTables(built.document).tables.switches![slot]!;
  expect(record).toMatchObject({ zone: -1, point: 0, graphic: 0 }); expect([...record.bytes.slice(2)]).toEqual(Array(12).fill(0));
  expect(built.surfaceSources!.filter(item => item.sourceOffset === original.graphic - 10)).toHaveLength(1);
  expect(built.document.switches[original.id]!.zone).toBeGreaterThanOrEqual(0);
  expect(availableSwitchSlots(source)).toContain(slot);
  expect(decodeProject(encodeProject(source)).roomSource).toEqual(source.roomSource);
  const fresh = switchRoomFixture(source), replacement = applyCommand(fresh.document, editRoomSource(setSourceSwitch(fresh.document, fresh.room.id, slot, { edge: 0, automatic: false })));
  expect(buildPlayableLevel(replacement, -1).document.switches[slot]).toMatchObject({ type: 0, countdown: 0 });
});

it('keeps captured copy settings independent through reset editing, repeated moves, geometry edits and captured clipboard paste after Undo', () => {
  const fixture = switchCopyPlayerFixture(), fresh = switchRoomFixture(fixture.document), slot = fixture.switchCopy.slot;
  const captured = fixture.document.roomSource.switchCopies![0]!, bytes = [...captured.bytes];
  bytes[2] = 7; bytes[3] = 91; bytes.splice(10, 4, 255, 18, 52, 86);
  let document = applyCommand(fresh.document, editRoomSource(setRoomSwitchCopies(fresh.document, [{ ...fixture.switchCopy, bytes: new Uint8Array(bytes) }])));
  document = applyCommand(document, editRoomSource(reassignCopiedSwitch(document, slot, { room: fresh.room.id, edge: 0 })));
  document = applyCommand(document, editRoomSource(setSourceSwitch(document, fresh.room.id, slot, { edge: 1, automatic: false })));
  const pasted = pasteRooms(document, copyRooms(document, [fresh.room.id]), 512, 0);
  document = applyCommand(document, pasted.command);
  document = applyCommand(document, editRoomSource(moveSourceSwitch(document, slot, { room: pasted.ids[0]!, edge: 2 })));
  const room = document.roomSource.rooms.find(room => room.id === pasted.ids[0])!;
  document = applyCommand(document, editRoomSource(insertSourceCorner(document.roomSource, room.id, 0, { x: 20512, z: 20128 })));
  document = applyCommand(document, editRoomSource(translateSourceRooms(document.roomSource, [room.id], 0, 512)));
  const result = importedAnimationTables(buildPlayableLevel(document, -1).document).tables.switches![slot]!;
  expect([...result.bytes.slice(2, 4)]).toEqual([0, 91]); expect([...result.bytes.slice(10)]).toEqual([255, 18, 52, 86]);
  const clipboard = copyRooms(document, [room.id]), afterUndo = pasteRooms(fresh.document, clipboard, 512, 512), restored = applyCommand(fresh.document, afterUndo.command);
  expect(restored.roomSource.rooms.at(-1)!.switches!.every(item => item.settings?.[2] === 7)).toBe(true);
  expect(() => buildPlayableLevel(restored, -1)).not.toThrow();
  expect(decodeProject(encodeProject(restored)).roomSource).toEqual(restored.roomSource);
});

it('does not resurrect a stripped copied panel when its room is copied, saved or edited', () => {
  const fixture = switchCopyPlayerFixture();
  const source = applyCommand(fixture.document, editRoomSource(reassignCopiedSwitch(fixture.document, fixture.switchCopy.slot, null)));
  const clipboard = copyRooms(source, [fixture.switchCopy.room]);
  for (const destination of [source, fixture.document]) {
    const pasted = pasteRooms(destination, clipboard, -8000, 0), result = applyCommand(destination, pasted.command);
    expect(buildPlayableLevel(result, -1).surfaceSources!.filter(item => item.sourceOffset === fixture.source.graphic - 10)).toHaveLength(destination === source ? 1 : 2);
    expect(result.roomSource.rooms.at(-1)!.retiredCopiedSwitches).toEqual([fixture.source.id]);
    expect(decodeProject(encodeProject(result)).roomSource).toEqual(result.roomSource);
  }
});

it('rejects unsafe retirement, duplicate attachment source, unsupported versions and incomplete captured settings', () => {
  const fixture = switchCopyPlayerFixture();
  const source = applyCommand(fixture.document, editRoomSource(reassignCopiedSwitch(fixture.document, fixture.switchCopy.slot, null))), snapshot = roomSourceSnapshot(source.roomSource);
  expect(() => restoreRoomSource(source, { ...snapshot, version: 22 })).toThrow(/unsupported fields/);
  expect(() => buildPlayableLevel({ ...source, roomSource: { ...source.roomSource, clearedSwitchSlots: [fixture.source.id] } }, -1)).toThrow(/active original switch/);
  expect(() => reassignCopiedSwitch(source, fixture.switchCopy.slot, null)).toThrow(/no longer exists/);
  const bad = { ...source, roomSource: { ...source.roomSource, rooms: source.roomSource.rooms.map(room => room.id === fixture.switchCopy.room ? { ...room, retiredCopiedSwitches: [fixture.source.id, fixture.source.id] } : room) } };
  expect(() => buildPlayableLevel(bad, -1)).toThrow(/repeat/);
  const fresh = switchRoomFixture(fixture.document), moved = applyCommand(fresh.document, editRoomSource(reassignCopiedSwitch(fresh.document, fixture.switchCopy.slot, { room: fresh.room.id, edge: 0 })));
  const saved = roomSourceSnapshot(moved.roomSource);
  expect(() => restoreRoomSource(moved, { ...saved, rooms: saved.rooms.map(room => room.id === fresh.room.id ? { ...room, switches: [{ ...room.switches![0]!, settings: [0] }] } : room) })).toThrow(/complete 14-byte/);
});

it('protects a required copied switch from removal, keeps the copied door linked when moved, and remaps further group copies', () => {
  const fixture = copiedSwitchAuthoringFixture(), slot = fixture.switchCopy.slot;
  expect(() => reassignCopiedSwitch(fixture.document, slot, null)).toThrow(/required by a door or lift/);
  const source = applyCommand(fixture.document, editRoomSource(reassignCopiedSwitch(fixture.document, slot, { room: fixture.room.id, edge: 0 })));
  expect(source.roomSource.controllerCopies).toEqual(fixture.document.roomSource.controllerCopies);
  expect(buildPlayableLevel(source, -1).document.mechanisms.filter(item => item.kind === 'door').at(-1)!.conditions).toBe(1 << (slot + 4));
  const pasted = pasteRooms(source, copyRooms(source, [...fixture.ids, fixture.room.id]), -8000, 0), result = applyCommand(source, pasted.command);
  const copied = result.roomSource.rooms.find(room => pasted.ids.includes(room.id) && room.switches?.length)!.switches![0]!;
  const door = buildPlayableLevel(result, -1).document.mechanisms.filter(item => item.kind === 'door').at(-1)!;
  expect(copied.slot).not.toBe(slot); expect(door.conditions).toBe(1 << (copied.slot + 4));
  expect(decodeProject(encodeProject(result)).roomSource).toEqual(result.roomSource);
});

it('retires independently mapped activation and graphics rooms without orphaning either attachment', () => {
  const original = openLevel('Separate switch attachments', originalFiles(1)), tables = importedAnimationTables(original).tables;
  const control = original.switches.find(item => item.zone >= 0)!;
  const activation = original.roomSource.rooms.find(room => room.originZone !== control.zone && original.points.some((point, index) => {
    const next = original.points[index + 1];
    return next && pointInRoomOutline(room.vertices, { x: (point.x + next.x) / 2, z: (point.z + next.z) / 2 });
  }))!;
  const point = original.points.findIndex((point, index) => {
    const next = original.points[index + 1];
    return next && pointInRoomOutline(activation.vertices, { x: (point.x + next.x) / 2, z: (point.z + next.z) / 2 });
  });
  // Native SwitchRoutine keeps the activation pair and panel pointer independent.
  // Exercise that supported input with real native points and a separate owner.
  const baseline = openLevel(original.name, { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined, { animations: {
    ...tables, doors: null, lifts: [], switches: tables.switches!.map((item, slot) => slot === control.id ? { ...item, zone: activation.recordZone, point } : { ...item, zone: -1 }),
  } }) }, 0, -1);
  const pasted = pasteRooms(baseline, copyRoomGroup(baseline, [activation.id]), 16000, 16000), document = applyCommand(baseline, pasted.command);
  const copy = document.roomSource.switchCopies![0]!;
  expect(copy.room).not.toBe(copy.graphicsRoom);
  const removed = applyCommand(document, editRoomSource(reassignCopiedSwitch(document, copy.slot, null))), built = buildPlayableLevel(removed, -1);
  for (const id of [copy.room, copy.graphicsRoom]) expect(removed.roomSource.rooms.find(room => room.id === id)!.retiredCopiedSwitches).toContain(control.id);
  expect(built.document.switches[copy.slot]!.zone).toBe(-1);
  expect(built.surfaceSources!.filter(item => item.sourceOffset === control.graphic - 10)).toHaveLength(1);
  expect(decodeProject(encodeProject(removed)).roomSource).toEqual(removed.roomSource);
});
