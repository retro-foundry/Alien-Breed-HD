import { expect, it } from 'vitest';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { compileControllerCopies, setRoomControllerCopies } from '../src/model/movingControllerCopies';
import { applyCommand, editRoomSource, setMechanismRules } from '../src/model/commands';
import { setMechanismRange as changeRange } from '../src/model/mechanismRanges';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject, encodeGameLevel } from '../src/model/project';
import { roomSourceSnapshot, restoreRoomSource, translateSourceRooms, removeSourceRooms } from '../src/model/roomSource';

it.each(['door', 'lift'] as const)('saves independent %s settings and compiles them through the normal preview/export path', kind => {
  const { document: original, copy, mechanism } = movingControllerCopyFixture(kind);
  const command = editRoomSource(setRoomControllerCopies(original, [copy]));
  const captured = applyCommand(original, command), snapshot = captured.roomSource.controllerCopies![0]!;
  expect(roomSourceSnapshot(captured.roomSource).version).toBe(5);
  const saved = decodeProject(encodeProject(captured));
  expect(saved.roomSource.controllerCopies).toEqual(captured.roomSource.controllerCopies);
  expect(saved.files).toEqual(original.files);
  const edited = applyCommand(saved, setMechanismRules(saved, kind, mechanism.id, 1, kind === 'door' ? 5 : 3, kind === 'door' ? 1 : 3));
  const ranged = applyCommand(edited, changeRange(edited, kind, mechanism.id, mechanism.top + 0.25, mechanism.bottom));
  const built = buildPlayableLevel(ranged, -1), motions = built.document.mechanisms.filter(item => item.kind === kind);
  expect(motions).toHaveLength(2);
  expect(motions[0]).toMatchObject({ conditions: 1, top: mechanism.top + 0.25 });
  expect(motions[1]).toMatchObject({ conditions: 0, top: mechanism.top + 8.25 });
  expect(encodeGameLevel(built.document).length).toBeGreaterThan(0);
  expect(compileControllerCopies([snapshot])[0]!.header).toEqual(copy.header);
  expect(applyCommand(captured, command, true).roomSource).toBe(original.roomSource);
  expect(applyCommand(applyCommand(captured, command, true), command).roomSource).toBe(captured.roomSource);
  expect(setRoomControllerCopies(captured, [copy]).after).toBe(captured.roomSource);
  copy.header!.fill(0);
  expect(snapshot.header.some(byte => byte !== 0)).toBe(true);
  expect(Object.isFrozen(snapshot.header)).toBe(true);
  expect(Object.isFrozen(snapshot.graphicsRooms[0])).toBe(true);
});

it('retains controller plans when moving a copied group and restores deleted controllers with one Undo', () => {
  const { document: original, copy, added } = movingControllerCopyFixture('door');
  const captured = applyCommand(original, editRoomSource(setRoomControllerCopies(original, [copy])));
  const moved = applyCommand(captured, editRoomSource(translateSourceRooms(captured.roomSource, added.map(room => room.id), 128, 128)));
  expect(moved.roomSource.controllerCopies).toBe(captured.roomSource.controllerCopies);
  expect(buildPlayableLevel(moved, -1).document.mechanisms).toHaveLength(2);
  const deletion = editRoomSource(removeSourceRooms(moved.roomSource, added.map(room => room.id)));
  const deleted = applyCommand(moved, deletion);
  expect(deleted.roomSource.controllerCopies).toBeUndefined();
  expect(buildPlayableLevel(deleted, -1).document.mechanisms).toHaveLength(1);
  expect(applyCommand(deleted, deletion, true).roomSource).toBe(moved.roomSource);
});

it('rejects malformed controller source without replacing the editable document', () => {
  const { document, copy } = movingControllerCopyFixture('door');
  const source = setRoomControllerCopies(document, [copy]).after;
  const snapshot = JSON.parse(JSON.stringify(roomSourceSnapshot(source)));
  const restore = (mutate: (input: typeof snapshot) => void) => {
    const input = structuredClone(snapshot); mutate(input); return () => restoreRoomSource(document, input);
  };
  expect(restore(input => { delete input.controllerCopies; })).toThrow(/missing.*controller/);
  expect(restore(input => { input.controllerCopies[0].header.pop(); })).toThrow(/every native settings byte/);
  expect(restore(input => { input.controllerCopies[0].graphicsRooms.push(input.controllerCopies[0].graphicsRooms[0]); })).toThrow(/repeats.*identity/);
  expect(restore(input => { input.controllerCopies[0].header[0] = 256; })).toThrow(/settings byte/);
  expect(restore(input => { input.controllerCopies[0].unknown = true; })).toThrow(/unsupported fields/);
  expect(restore(input => { input.version = 4; })).toThrow(/unsupported fields/);
  expect(document.roomSource).not.toHaveProperty('controllerCopies');
});
