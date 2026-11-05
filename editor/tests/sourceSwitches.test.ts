import { importedAnimationTables } from '../src/model/animationBuild';
import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { roomSourceSnapshot, restoreRoomSource, insertSourceCorner, translateSourceRooms, removeSourceRooms } from '../src/model/roomSource';
import { setSourceSwitch } from '../src/model/sourceSwitches';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { encodeProject, decodeProject } from '../src/model/project';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';

import { switchRoomFixture } from './sourceSwitchFixture';
import { authoredMovingFixture } from './authoredMovingFixture';
import { setSourceMovingRoom, sourceMovingSettings } from '../src/model/sourceMovingRoom';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { availableSwitchSlots } from '../src/model/sourceSwitches';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';

it('creates a complete native panel and independent activation pair with source save, one Undo and Redo', () => {
  const { document, room, slot } = switchRoomFixture();
  const history = execute({ document, past: [], future: [] }, editRoomSource(setSourceSwitch(document, room.id, slot, { edge: 0, automatic: false })));
  const source = history.document, built = buildPlayableLevel(source, -1).document, item = built.switches[slot]!;
  const owner = built.zones[item.zone]!, panel = owner.lower.find(record => record.offset === item.graphic - 10)!;
  expect(panel.wall).toMatchObject({ texture: 11, stripStart: 0, stripEnd: 31, verticalMask: 31, verticalShift: 5, horizontalMask: 31, topFixed: (owner.floor - 64) * 256, bottomFixed: (owner.floor - 32) * 256 });
  expect(panel.points).toEqual([item.point, item.point + 1]); expect(item.type).toBe(0);
  const [a, b] = panel.points.map(id => built.points[id]!);
  expect(Math.hypot(b!.x - a!.x, b!.z - a!.z)).toBe(64);
  expect({ x: (a!.x + b!.x) / 2, z: (a!.z + b!.z) / 2 }).toEqual({ x: 20000, z: 20128 });
  expect(roomSourceSnapshot(source.roomSource).version).toBe(17);
  expect(decodeProject(encodeProject(source)).roomSource).toEqual(source.roomSource);
  expect(undo(history).document.roomSource).toBe(document.roomSource); expect(redo(undo(history)).document.roomSource).toBe(source.roomSource);
  expect(built.switches.filter(item => item.zone >= 0)).toHaveLength(document.switches.filter(item => item.zone >= 0).length + 1);
});

it('retains switch identity on inserted corners and room translations, and allows reassignment, reset editing and removal', () => {
  const { document, room, slot } = switchRoomFixture();
  let source = applyCommand(document, editRoomSource(setSourceSwitch(document, room.id, slot, { edge: 0, automatic: false })));
  source = applyCommand(source, editRoomSource(insertSourceCorner(source.roomSource, room.id, 0, { x: 20000, z: 20128 })));
  source = applyCommand(source, editRoomSource(translateSourceRooms(source.roomSource, [room.id], 100, 100)));
  let built = buildPlayableLevel(source, -1).document, item = built.switches[slot]!;
  expect((built.points[item.point]!.z + built.points[item.point + 1]!.z) / 2).toBe(20228);
  source = applyCommand(source, editRoomSource(setSourceSwitch(source, room.id, slot, { edge: 2, automatic: true })));
  built = buildPlayableLevel(source, -1).document; item = built.switches[slot]!; expect(item.type).toBe(1);
  expect(() => removeSourceRooms(source.roomSource, [room.id])).toThrow(/Remove authored switches/);
  source = applyCommand(source, editRoomSource(setSourceSwitch(source, room.id, slot, null)));
  expect(decodeProject(encodeProject(source)).roomSource.rooms.at(-1)!.switches).toBeUndefined();
  expect(buildPlayableLevel(source, -1).document.switches[slot]!.zone).toBeLessThan(0);
});

it('authors and copies switches when the imported map has no switch table at all', () => {
  const files = openLevel('Unpacked switch baseline', originalFiles(1)).files, graphics = files.graphics.slice(); new DataView(graphics.buffer).setInt32(8, 0);
  const original = openLevel('No inherited switches', { ...files, graphics });
  expect(original.switches).toHaveLength(0);
  const fixture = switchRoomFixture(original);
  const source = applyCommand(fixture.document, editRoomSource(setSourceSwitch(fixture.document, fixture.room.id, fixture.slot, { edge: 0, automatic: false })));
  expect(buildPlayableLevel(source, -1).document.switches.filter(item => item.zone >= 0)).toHaveLength(1);
  const pasted = pasteRooms(source, copyRooms(source, [fixture.room.id]), 512, 0), result = applyCommand(source, pasted.command);
  expect(buildPlayableLevel(result, -1).document.switches.filter(item => item.zone >= 0)).toHaveLength(2);
});

it('allocates independent slots and panels on room clipboard copies', () => {
  const { document, room, slot } = switchRoomFixture();
  const source = applyCommand(document, editRoomSource(setSourceSwitch(document, room.id, slot, { edge: 0, automatic: true })));
  const clipboard = copyRooms(source, [room.id]), pasted = pasteRooms(source, clipboard, 512, 0);
  const result = applyCommand(source, pasted.command), copied = result.roomSource.rooms.find(room => room.id === pasted.ids[0])!.switches![0]!;
  expect(copied.slot).not.toBe(slot); expect(copied.automatic).toBe(true);
  const built = buildPlayableLevel(result, -1).document;
  expect(built.switches[slot]!.point).not.toBe(built.switches[copied.slot]!.point);
  expect(built.switches[slot]!.graphic).not.toBe(built.switches[copied.slot]!.graphic);
  // The captured clipboard remains portable after the original edit is undone.
  const afterUndo = pasteRooms(document, clipboard, 512, 0), restored = applyCommand(document, afterUndo.command);
  expect(restored.roomSource.rooms.at(-1)!.switches).toHaveLength(1);
  expect(() => decodeProject(encodeProject(restored))).not.toThrow();
});

it('rejects used slots, short walls, missing identities and unsupported switch source fields', () => {
  const { document, room, slot } = switchRoomFixture();
  const active = document.switches.find(item => item.zone >= 0)!;
  expect(() => setSourceSwitch(document, room.id, active.id, { edge: 0, automatic: false })).toThrow(/already used/);
  const source = applyCommand(document, editRoomSource(setSourceSwitch(document, room.id, slot, { edge: 0, automatic: false })));
  const snapshot = roomSourceSnapshot(source.roomSource);
  expect(() => restoreRoomSource(document, { ...snapshot, rooms: snapshot.rooms.map(item => item.id === room.id ? { ...item, switches: [{ slot, boundary: 0, automatic: false, extra: 0 }] } : item) })).toThrow(/needs its slot/);
  const missing = { ...source, roomSource: { ...source.roomSource, rooms: source.roomSource.rooms.map(item => item.id === room.id ? { ...item, switches: [{ slot, boundary: 999, automatic: false }] } : item) } };
  expect(() => buildPlayableLevel(missing, -1)).toThrow(/lost its wall/);
  const short = { ...source, roomSource: { ...source.roomSource, rooms: source.roomSource.rooms.map(item => item.id === room.id ? { ...item, vertices: item.vertices.map(point => ({ ...point, z: 20000 + (point.z - 20000) / 8 })) } : item) } };
  expect(() => buildPlayableLevel(short, -1)).toThrow(/at least 64/);
});

it('keeps gated moving targets synchronized and remaps internal requirements when copying a switch and door together', () => {
  const fixture = authoredMovingFixture('door'), slot = availableSwitchSlots(fixture.document)[0]!;
  let source = applyCommand(fixture.document, editRoomSource(setSourceSwitch(fixture.document, fixture.neighbour.id, slot, { edge: 2, automatic: false })));
  const header = fixture.input.header.slice(), view = new DataView(header.buffer);
  view.setUint16(14, 1 << (slot + 4));
  source = applyCommand(source, editRoomSource(setSourceMovingRoom(source, fixture.room.id, { kind: 'door', header: [...header], wall: roomAppearanceChoices(source, fixture.room).lower[0]!.offset })));
  expect(() => setSourceSwitch(source, fixture.neighbour.id, slot, null)).toThrow(/required by a door/);
  const built = buildPlayableLevel(source, -1).document, mechanism = built.mechanisms.filter(item => item.kind === 'door').at(-1)!;
  expect(mechanism.conditions).toBe(1 << (slot + 4));
  expect(built.zones[mechanism.zone]!.lower.find(record => record.offset === importedAnimationTables(built).tables.doors!.at(-1)!.graphic)?.type).toBe(2);
  const pasted = pasteRooms(source, copyRooms(source, [fixture.room.id, fixture.neighbour.id]), 0, 1024), result = applyCommand(source, pasted.command);
  const copiedSwitch = result.roomSource.rooms.find(room => pasted.ids.includes(room.id) && room.switches?.length)!.switches![0]!;
  const copiedDoor = result.roomSource.rooms.find(room => pasted.ids.includes(room.id) && room.moving)!.moving!;
  expect(copiedSwitch.slot).not.toBe(slot);
  expect(sourceMovingSettings(copiedDoor).conditions).toBe(1 << (copiedSwitch.slot + 4));
  expect(() => buildPlayableLevel(result, -1)).not.toThrow();
});
