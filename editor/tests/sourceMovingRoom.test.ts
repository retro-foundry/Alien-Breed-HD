import { expect, it } from 'vitest';
import { authoredMovingFixture } from './authoredMovingFixture';
import { setSourceMovingRoom, sourceMovingSettings } from '../src/model/sourceMovingRoom';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { roomSourceSnapshot, moveSourceCorner, restoreRoomSource } from '../src/model/roomSource';
import { encodeProject, decodeProject } from '../src/model/project';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { copyRoomGroup, pasteRooms } from '../src/model/roomClipboard';

it.each(['door', 'lift'] as const)('authors a persistent %s with native initial position, history, save, geometry and independent clipboard copies', kind => {
  const fixture = authoredMovingFixture(kind), document = fixture.document;
  const header = fixture.input.header.slice();
  if (kind === 'door') new DataView(header.buffer).setInt16(4, new DataView(header.buffer).getInt16(0));
  const value = { kind, header: [...header], wall: roomAppearanceChoices(document, fixture.room).lower[0]!.offset };
  const command = editRoomSource(setSourceMovingRoom(document, fixture.room.id, value));
  let history = execute({ document, past: [], future: [] }, command);
  expect(roomSourceSnapshot(history.document.roomSource).version).toBe(10);
  const saved = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(saved, -1);
  const mechanism = built.document.mechanisms.filter(item => item.kind === kind).at(-1)!;
  expect(mechanism.current).toBe(sourceMovingSettings(value).current);
  expect(built.document.zones[mechanism.zone]![kind === 'door' ? 'ceiling' : 'floor']).toBe(mechanism.current);
  expect(saved.files).toEqual(document.files);
  if (kind === 'door') {
    expect(built.document.zones[mechanism.zone]!.ceiling).toBe(built.document.zones[mechanism.zone]!.floor);
    expect(built.document.zones[mechanism.zone]!.lower.some(record => record.wall?.topFixed === mechanism.top * 256 && record.wall?.bottomFixed === mechanism.bottom * 256)).toBe(true);
  }
  history = undo(history); expect(history.document.roomSource).toBe(document.roomSource);
  history = redo(history); expect(history.document.roomSource.rooms.find(room => room.id === fixture.room.id)!.moving).toEqual(value);
  const corner = fixture.room.vertices[0]!;
  const moved = applyCommand(saved, editRoomSource(moveSourceCorner(saved.roomSource, fixture.room.id, 0, { x: corner.x - 16, z: corner.z })));
  expect(moved.roomSource.rooms.find(room => room.id === fixture.room.id)!.moving).toEqual(value);
  expect(() => buildPlayableLevel(moved, -1)).not.toThrow();
  const clipboard = copyRoomGroup(saved, [fixture.room.id]);
  expect(clipboard.ids).toContain(fixture.neighbour.id);
  const pasted = applyCommand(saved, pasteRooms(saved, clipboard, 3000, 1000).command);
  expect(pasted.roomSource.rooms.filter(room => room.moving)).toHaveLength(2);
  const copied = buildPlayableLevel(pasted, -1).document.mechanisms.filter(item => item.kind === kind).slice(-2);
  expect(copied[0]!.zone).not.toBe(copied[1]!.zone); expect(copied[0]!.walls[0]!.graphic).not.toBe(copied[1]!.walls[0]!.graphic);
  const removed = applyCommand(saved, editRoomSource(setSourceMovingRoom(saved, fixture.room.id, null)));
  expect(buildPlayableLevel(removed, -1).document.mechanisms.filter(item => item.kind === kind)).toHaveLength(document.mechanisms.filter(item => item.kind === kind).length);
});

it('rejects malformed source settings and unsupported old archive fields without discarding the project', () => {
  const fixture = authoredMovingFixture('door'), document = fixture.document, room = fixture.room;
  const valid = { kind: 'door' as const, header: [...fixture.input.header], wall: roomAppearanceChoices(document, room).lower[0]!.offset };
  for (const value of [{ ...valid, header: [] }, { ...valid, wall: -1 }, { ...valid, kind: 'water' }, { ...valid, extra: 1 }])
    expect(() => setSourceMovingRoom(document, room.id, value as typeof valid)).toThrow();
  const changed = applyCommand(document, editRoomSource(setSourceMovingRoom(document, room.id, valid)));
  const snapshot = roomSourceSnapshot(changed.roomSource);
  expect(() => restoreRoomSource(document, { ...snapshot, version: 9 })).toThrow(/unsupported source fields/);
  const owned = document.mechanisms.find(item => item.kind === 'door')!;
  expect(() => setSourceMovingRoom(document, `room-${owned.zone}`, valid)).toThrow(/existing door/);
});
