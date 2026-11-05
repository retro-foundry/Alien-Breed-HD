import { expect, it } from 'vitest';
import { switchCopyFixture } from './switchCopyFixture';
import { setRoomSwitchCopies } from '../src/model/switchCopies';
import { applyCommand, editRoomSource, numberPatch } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject, encodeGameLevel } from '../src/model/project';
import { removeSourceRooms, restoreRoomSource, roomSourceSnapshot, translateSourceRooms } from '../src/model/roomSource';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { openLevel } from '../src/model/level';
import { setRoomControllerCopies } from '../src/model/movingControllerCopies';

it('persists independent native switch settings and compiles source through normal preview/export', () => {
  const { document: original, switchCopy: copy, source } = switchCopyFixture();
  const change = editRoomSource(setRoomSwitchCopies(original, [copy])), document = applyCommand(original, change);
  expect(roomSourceSnapshot(document.roomSource).version).toBe(6);
  const saved = decodeProject(encodeProject(document));
  expect(saved.files).toEqual(original.files); expect(saved.roomSource.switchCopies).toEqual(document.roomSource.switchCopies);
  const type = source.type ? 0 : 1, countdown = source.countdown === 64 ? 63 : 64;
  const edited = applyCommand(saved, { label: 'Change original switch settings', patches: [numberPatch(saved, 'graphics', source.offset + 2, type, 1, false),
    numberPatch(saved, 'graphics', source.offset + 3, countdown, 1)] });
  const built = buildPlayableLevel(edited, -1);
  expect(built.document.switches[source.id]).toMatchObject({ type, countdown });
  expect(built.document.switches[copy.slot]).toMatchObject({ type: source.type, countdown: source.countdown });
  expect(encodeGameLevel(built.document).length).toBeGreaterThan(0);
  expect(applyCommand(document, change, true).roomSource).toBe(original.roomSource);
  expect(applyCommand(applyCommand(document, change, true), change).roomSource).toBe(document.roomSource);
  expect(setRoomSwitchCopies(document, [copy]).after).toBe(document.roomSource);
  expect(Object.isFrozen(saved.roomSource.switchCopies![0]!.bytes)).toBe(true);
});

it('keeps switch plans during group moves and removes their controlled rooms with one reversible edit', () => {
  const { document: original, switchCopy: copy } = switchCopyFixture();
  const document = applyCommand(original, editRoomSource(setRoomSwitchCopies(original, [copy])));
  const ids = document.roomSource.rooms.filter(room => room.originZone === null).map(room => room.id);
  const moved = applyCommand(document, editRoomSource(translateSourceRooms(document.roomSource, ids, 64, 64)));
  expect(moved.roomSource.switchCopies).toBe(document.roomSource.switchCopies);
  expect(buildPlayableLevel(moved, -1).document.switches[copy.slot]!.zone).toBeGreaterThanOrEqual(0);
  const deletion = editRoomSource(removeSourceRooms(moved.roomSource, ids)), deleted = applyCommand(moved, deletion);
  expect(deleted.roomSource.switchCopies).toBeUndefined();
  expect(buildPlayableLevel(deleted, -1).document.switches[copy.slot]!.zone).toBeLessThan(0);
  expect(applyCommand(deleted, deletion, true).roomSource).toBe(moved.roomSource);
});

it('rejects damaged switch source without discarding the editable project', () => {
  const { document, switchCopy: copy } = switchCopyFixture();
  const captured = setRoomSwitchCopies(document, [copy]).after, snapshot = JSON.parse(JSON.stringify(roomSourceSnapshot(captured)));
  const restore = (mutate: (input: typeof snapshot) => void) => {
    const input = structuredClone(snapshot); mutate(input); return () => restoreRoomSource(document, input);
  };
  expect(restore(input => { delete input.switchCopies; })).toThrow(/missing.*switch/);
  expect(restore(input => { delete input.controllerCopies; })).toThrow(/missing.*controller/);
  expect(restore(input => { input.switchCopies[0].bytes.pop(); })).toThrow(/complete 14-byte native record/);
  expect(restore(input => { input.switchCopies[0].bytes[0] = 256; })).toThrow(/settings byte/);
  expect(restore(input => { input.switchCopies.push(input.switchCopies[0]); })).toThrow(/repeats.*slot/);
  expect(restore(input => { input.switchCopies[0].unexpected = true; })).toThrow(/unsupported fields/);
  expect(restore(input => { input.version = 5; })).toThrow(/unsupported fields/);
  expect(document.roomSource).not.toHaveProperty('switchCopies');
});

it('retains both copied motion and switch plans through shared geometry edits and v6 source saves', () => {
  const moving = movingControllerCopyFixture('door'), switched = switchCopyFixture();
  const tables = importedAnimationTables(moving.document).tables;
  const native = openLevel('Combined plans', { ...moving.document.files, graphics: buildGraphicsStreams(moving.document, undefined, undefined,
    { animations: { ...tables, switches: importedAnimationTables(switched.document).tables.switches } }) }, 0, -1);
  const rooms = [...moving.document.roomSource.rooms], appearances = [...moving.document.roomAppearances], mapping = new Map<string, string>();
  let nextId = moving.document.roomSource.nextId;
  for (const room of switched.document.roomSource.rooms.filter(room => room.originZone === null)) {
    let target = rooms.find(item => item.originZone === null && item.recordZone === room.recordZone);
    if (!target) {
      target = { ...room, id: `room-${nextId++}` }; rooms.push(target);
      const appearance = switched.document.roomAppearances.find(item => item.room === room.id);
      if (appearance) appearances.push({ ...appearance, room: target.id });
    }
    mapping.set(room.id, target.id);
  }
  const combined = { ...native, roomSource: { rooms, nextId }, roomAppearances: appearances };
  const motion = applyCommand(combined, editRoomSource(setRoomControllerCopies(combined, [moving.copy])));
  const copy = { ...switched.switchCopy, room: mapping.get(switched.switchCopy.room)!, graphicsRoom: mapping.get(switched.switchCopy.graphicsRoom)! };
  const both = applyCommand(motion, editRoomSource(setRoomSwitchCopies(motion, [copy])));
  const moved = applyCommand(both, editRoomSource(translateSourceRooms(both.roomSource, rooms.filter(room => room.originZone === null).map(room => room.id), 64, 64)));
  const saved = decodeProject(encodeProject(moved)), built = buildPlayableLevel(saved, -1);
  expect(saved.roomSource.controllerCopies).toEqual(both.roomSource.controllerCopies);
  expect(saved.roomSource.switchCopies).toEqual(both.roomSource.switchCopies);
  expect(roomSourceSnapshot(saved.roomSource).version).toBe(6);
  expect(built.document.mechanisms).toHaveLength(2);
  expect(built.document.switches[copy.slot]!.zone).toBeGreaterThanOrEqual(0);
});
