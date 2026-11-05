import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, undo, redo, numberPatch } from '../src/model/commands';
import { reassignNativeSwitch, moveSourceSwitch, setSourceSwitch, availableSwitchSlots, switchRequirements } from '../src/model/sourceSwitches';
import { activeNativeSwitches, switchSourceRoom } from '../src/model/nativeSwitches';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { restoreRoomSource, roomSourceSnapshot } from '../src/model/roomSource';
import { importedAnimationTables } from '../src/model/animationBuild';
import { switchRoomFixture } from './sourceSwitchFixture';
import { switchCopyFixture } from './switchCopyFixture';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { setRoomSwitchCopies } from '../src/model/switchCopies';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { movingRoomOwnership } from '../src/model/movingRoomEligibility';
import { assertStaticRoom } from '../src/model/roomStoreys';

it('reassigns an imported switch with the same condition number, removes its old panel and saves one undoable source edit', () => {
  const { document, room } = switchRoomFixture();
  const original = document.switches.find(item => item.zone >= 0)!;
  const history = execute({ document, past: [], future: [] }, editRoomSource(reassignNativeSwitch(document, original.id, { room: room.id, edge: 0 })));
  const source = history.document, built = buildPlayableLevel(source, -1), item = built.document.switches[original.id]!;
  expect(source.roomSource.rooms.at(-1)!.switches).toEqual([{ slot: original.id, boundary: 0, automatic: original.type !== 0, native: original.id }]);
  expect(built.topology!.cells.find(cell => cell.id === item.zone)!.sourceId).toBe(room.id);
  expect(built.surfaceSources!.some(record => record.sourceOffset === original.graphic - 10)).toBe(false);
  expect(built.document.mechanisms.map(item => item.conditions)).toEqual(document.mechanisms.map(item => item.conditions));
  expect(activeNativeSwitches(source).some(item => item.id === original.id)).toBe(false);
  expect(switchSourceRoom(source, original.id)).toBe(room.id);
  expect(availableSwitchSlots(source)).not.toContain(original.id);
  expect(decodeProject(encodeProject(source)).roomSource).toEqual(source.roomSource);
  expect(roomSourceSnapshot(source.roomSource).version).toBe(22);
  expect(undo(history).document.roomSource).toBe(document.roomSource);
  expect(redo(undo(history)).document.roomSource).toBe(source.roomSource);
  const edited = applyCommand(source, editRoomSource(setSourceSwitch(source, room.id, original.id, { edge: 1, automatic: !original.type })));
  expect(edited.roomSource.rooms.at(-1)!.switches![0]!.native).toBe(original.id);
  expect(() => buildPlayableLevel(edited, -1)).not.toThrow();
});

it('preserves opaque native timer/state/reset bytes and panel phase while moving an adopted switch again', () => {
  const fixture = switchRoomFixture(), original = fixture.document.switches.find(item => item.zone >= 0)!;
  const seeded = applyCommand(fixture.document, { label: 'Native state fixture', patches: [
    numberPatch(fixture.document, 'graphics', original.offset + 2, 7, 1, false),
    numberPatch(fixture.document, 'graphics', original.offset + 3, 91, 1, false),
    numberPatch(fixture.document, 'graphics', original.offset + 10, 0xff123456, 4, false),
  ] });
  const added = pasteRooms(seeded, copyRooms(seeded, [fixture.room.id]), 512, 0), document = applyCommand(seeded, added.command);
  let source = applyCommand(document, editRoomSource(reassignNativeSwitch(document, original.id, { room: fixture.room.id, edge: 0 })));
  const before = source.roomSource;
  source = applyCommand(source, editRoomSource(moveSourceSwitch(source, original.id, { room: added.ids[0]!, edge: 1 })));
  expect(before.rooms.find(room => room.id === fixture.room.id)!.switches).toHaveLength(1);
  expect(source.roomSource.rooms.find(room => room.id === fixture.room.id)!.switches).toBeUndefined();
  const built = buildPlayableLevel(source, -1).document, record = importedAnimationTables(built).tables.switches![original.id]!;
  expect([...record.bytes.slice(2, 4)]).toEqual([7, 91]); expect([...record.bytes.slice(10)]).toEqual([255, 18, 52, 86]);
  expect(built.zones[record.zone]!.lower.find(item => item.offset === record.graphic - 10)!.wall!.horizontalOffset & 2).toBe(2);
  expect(decodeProject(encodeProject(source)).roomSource).toEqual(source.roomSource);
});

it.each(Array.from({ length: 16 }, (_, slot) => slot + 1))('reassigns every active switch in original level %i without changing other controllers', level => {
  const { document, room } = switchRoomFixture(openLevel(`Original ${level}`, originalFiles(level)));
  expect(() => assertStaticRoom(document, room)).not.toThrow();
  for (const original of activeNativeSwitches(document)) {
    const source = applyCommand(document, editRoomSource(reassignNativeSwitch(document, original.id, { room: room.id, edge: 0 })));
    const built = buildPlayableLevel(source, -1), native = built.document.switches[original.id]!;
    expect(built.topology!.cells.find(cell => cell.id === native.zone)!.sourceId).toBe(room.id);
    expect(built.surfaceSources!.some(record => record.sourceOffset === original.graphic - 10)).toBe(false);
    for (const unchanged of document.switches.filter(item => item.zone >= 0 && item.id !== original.id)) {
      const record = built.document.switches[unchanged.id]!;
      expect(record).toMatchObject({ type: unchanged.type, countdown: unchanged.countdown });
      expect(record.state).toEqual(unchanged.state);
    }
  }
});

it('removes unrequired native controllers and panels, clears timer/state/pointers, and frees the slot for a clean new switch', () => {
  const fixture = switchCopyFixture(), original = fixture.source;
  const copied = applyCommand(fixture.document, editRoomSource(setRoomSwitchCopies(fixture.document, [fixture.switchCopy])));
  const document = applyCommand(copied, { label: 'Active native timer fixture', patches: [
    numberPatch(copied, 'graphics', original.offset + 2, 7, 1, false),
    numberPatch(copied, 'graphics', original.offset + 3, 91, 1, false),
    numberPatch(copied, 'graphics', original.offset + 10, 0xff123456, 4, false),
  ] });
  expect(switchRequirements(document, original.id)).toBe(false);
  const removed = applyCommand(document, editRoomSource(reassignNativeSwitch(document, original.id, null)));
  const prepared = removed;
  const built = buildPlayableLevel(prepared, -1), record = importedAnimationTables(built.document).tables.switches![original.id]!;
  expect(record.zone).toBe(-1); expect(record.point).toBe(0); expect(record.graphic).toBe(0);
  expect([...record.bytes.slice(2)]).toEqual(Array(12).fill(0));
  expect(availableSwitchSlots(prepared)).toContain(original.id);
  expect(built.document.switches[fixture.switchCopy.slot]!.zone).toBeGreaterThanOrEqual(0);
  const copiedRoom = prepared.roomSource.rooms.find(room => room.id === fixture.switchCopy.room)!;
  expect(movingRoomOwnership(prepared, copiedRoom)?.kind).toBe('switch');
  expect(() => assertStaticRoom(prepared, copiedRoom)).toThrow(/belongs to a door, lift or switch/);
  expect(decodeProject(encodeProject(prepared)).roomSource).toEqual(prepared.roomSource);
  const originalRoom = prepared.roomSource.rooms.find(room => room.originZone === original.zone)!;
  const pasted = pasteRooms(prepared, copyRooms(prepared, [originalRoom.id]), -14000, 0);
  const copiedAfterRemoval = buildPlayableLevel(applyCommand(prepared, pasted.command), -1);
  // Only the already independent copy retains this native panel template.
  expect(copiedAfterRemoval.surfaceSources!.filter(item => item.sourceOffset === original.graphic - 10)).toHaveLength(1);
  const fresh = switchRoomFixture(prepared);
  const replaced = applyCommand(fresh.document, editRoomSource(setSourceSwitch(fresh.document, fresh.room.id, original.id, { edge: 0, automatic: false })));
  expect(replaced.roomSource.rooms.at(-1)!.switches![0]!.native).toBeUndefined();
  const clean = buildPlayableLevel(replaced, -1).document.switches[original.id]!;
  expect(clean).toMatchObject({ type: 0, countdown: 0 }); expect([...clean.state]).toEqual([0, 0, 0, 0]);
});

it('copies an adopted switch independently and keeps captured native provenance after Undo', () => {
  const { document, room } = switchRoomFixture(), original = document.switches.find(item => item.zone >= 0)!;
  const source = applyCommand(document, editRoomSource(reassignNativeSwitch(document, original.id, { room: room.id, edge: 0 })));
  const clipboard = copyRooms(source, [room.id]);
  for (const destination of [source, document]) {
    const pasted = pasteRooms(destination, clipboard, 512, 0), result = applyCommand(destination, pasted.command);
    const copy = result.roomSource.rooms.at(-1)!.switches![0]!;
    expect(copy.native).toBe(original.id); expect(copy.slot).not.toBe(original.id);
    const built = buildPlayableLevel(result, -1).document;
    expect(built.switches[copy.slot]!.zone).toBeGreaterThanOrEqual(0);
    expect(built.switches[copy.slot]!.graphic).not.toBe(built.switches[original.id]!.graphic);
    expect(decodeProject(encodeProject(result)).roomSource).toEqual(result.roomSource);
  }
});

it('rejects required removal, unsupported source versions, duplicate removals and missing native templates', () => {
  const { document, room } = switchRoomFixture();
  const required = document.switches.find(item => item.zone >= 0 && switchRequirements(document, item.id));
  expect(required).toBeDefined();
  expect(() => reassignNativeSwitch(document, required!.id, null)).toThrow(/required by a door or lift/);
  const source = applyCommand(document, editRoomSource(reassignNativeSwitch(document, required!.id, { room: room.id, edge: 0 })));
  const snapshot = roomSourceSnapshot(source.roomSource);
  expect(() => restoreRoomSource(document, { ...snapshot, version: 21 })).toThrow(/unsupported fields/);
  expect(() => restoreRoomSource(document, { ...snapshot, removedNativeSwitches: [required!.id, required!.id] })).toThrow(/repeat/);
  const bad = { ...source, roomSource: { ...source.roomSource, removedNativeSwitches: [availableSwitchSlots(document)[0]!] } };
  expect(() => buildPlayableLevel(bad, -1)).toThrow(/no active native template/);
});
