import { expect, it } from 'vitest';
import { copyRooms, copyRoomGroup, pasteRooms } from '../src/model/roomClipboard';
import { applyCommand, editRoomSource, setSourceTeleport, setZoneProperty } from '../src/model/commands';
import { roomHeightFixture } from './roomHeightFixture';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { switchCopyFixture } from './switchCopyFixture';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { editableTeleport } from '../src/model/sourceTeleports';
import { editableObjects } from '../src/model/sourceObjects';
import { setRoomControllerCopies } from '../src/model/movingControllerCopies';
import { setRoomSwitchCopies } from '../src/model/switchCopies';
import { createObject, objectTemplates } from '../src/model/objectAuthoring';
import { numberPatch, setMechanismRules } from '../src/model/commands';
import { linkedRoomCopyFixture } from './linkedRoomCopyFixture';
import { nativePartialRoomCopyFixture } from './nativeRoomCopyFixture';
import { BinaryReader } from '../src/model/binary';

it('copies authored rooms, settings and internal/external teleports as one reversible source edit', () => {
  const fixture = roomHeightFixture(), { from, to } = fixture;
  const centre = (room: typeof from) => ({ x: Math.round(room.vertices.reduce((n, p) => n + p.x, 0) / room.vertices.length),
    z: Math.round(room.vertices.reduce((n, p) => n + p.z, 0) / room.vertices.length) });
  const target = centre(to), external = fixture.document.roomSource.rooms.find(room => room.originZone === 0)!, outside = centre(external);
  let document = applyCommand(fixture.document, setSourceTeleport(fixture.document, from.id, to.id, target.x, target.z));
  document = applyCommand(document, setSourceTeleport(document, to.id, external.id, outside.x, outside.z));
  const clipboard = copyRooms(document, [from.id, to.id]);
  const changed = applyCommand(document, setZoneProperty(document, from.recordZone, 'brightness', 127));
  const result = pasteRooms(changed, clipboard, -10000, 0), pasted = applyCommand(changed, result.command);
  const first = pasted.roomSource.rooms.find(room => room.id === result.ids[0])!, second = pasted.roomSource.rooms.find(room => room.id === result.ids[1])!;
  expect(first.properties!.brightness).toBe(document.zones[from.recordZone]!.brightness);
  expect(editableTeleport(pasted, first)).toEqual({ room: first.id, destination: second.id, x: target.x - 10000, z: target.z });
  expect(editableTeleport(pasted, second)).toEqual({ room: second.id, destination: external.id, ...outside });
  expect(pasted.starts).toEqual(changed.starts); expect(pasted.sourceStarts).toEqual(changed.sourceStarts);
  expect(decodeProject(encodeProject(pasted)).roomSource).toEqual(pasted.roomSource);
  expect(buildPlayableLevel(pasted, -1).document.zones.length).toBeGreaterThan(changed.zones.length);
  expect(applyCommand(pasted, result.command, true).roomSource).toBe(changed.roomSource);
  expect(applyCommand(applyCommand(pasted, result.command, true), result.command).roomSource).toBe(pasted.roomSource);
});

it.each(['door', 'lift'] as const)('copies a complete existing %s group with independent controllers and objects', kind => {
  const fixture = movingControllerCopyFixture(kind);
  let document = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [fixture.copy])));
  const room = fixture.added[0]!, template = objectTemplates(document).find(item => !item.upper)!;
  const point = { x: room.vertices.reduce((n, p) => n + p.x / room.vertices.length, 0), z: room.vertices.reduce((n, p) => n + p.z / room.vertices.length, 0) };
  const actor = createObject(document, { template: template.id, room: room.id, ...point, y: template.y, upper: false });
  document = applyCommand(document, actor.command);
  const selected = fixture.added.map(room => room.id), clipboard = copyRooms(document, selected);
  const changed = applyCommand(document, { label: 'Change original actor health', patches: [numberPatch(document, 'data', document.objects[actor.id]!.offset + 2, 17, 2)] });
  const result = pasteRooms(changed, clipboard, -8000, 0), pasted = applyCommand(changed, result.command);
  const built = buildPlayableLevel(pasted, -1);
  expect(built.document.mechanisms.filter(item => item.kind === kind)).toHaveLength(3);
  expect(pasted.roomSource.controllerCopies).toHaveLength(2);
  expect(result.objects).toHaveLength(1);
  const copiedActor = editableObjects(pasted).find(item => item.id === result.objects[0])!;
  const capturedActor = editableObjects(document).find(item => item.id === actor.id)!;
  expect(copiedActor).toMatchObject({ x: capturedActor.x - 8000, z: capturedActor.z, y: capturedActor.y, health: capturedActor.health });
  expect(pasted.objects.length - document.objects.length).toBe(result.objects.length);
  for (const id of result.objects) expect(result.ids).toContain(editableObjects(pasted).find(item => item.id === id)!.room);
  expect(applyCommand(pasted, result.command, true).files).toEqual(changed.files);
});

it('copies a switch group into a fresh native slot and rejects incomplete or overlapping copies atomically', () => {
  const fixture = switchCopyFixture();
  const document = applyCommand(fixture.document, editRoomSource(setRoomSwitchCopies(fixture.document, [fixture.switchCopy])));
  const selected = document.roomSource.rooms.filter(room => room.originZone === null).map(room => room.id), clipboard = copyRooms(document, selected);
  const result = pasteRooms(document, clipboard, -8000, 0), pasted = applyCommand(document, result.command);
  expect(pasted.roomSource.switchCopies).toHaveLength(2);
  const [original, copied] = pasted.roomSource.switchCopies!;
  expect(copied!.slot).not.toBe(original!.slot); expect(result.ids).toContain(copied!.room);
  expect(buildPlayableLevel(pasted, -1).document.switches.filter(item => item.zone >= 0)).toHaveLength(3);
  expect(() => pasteRooms(document, clipboard, 0, 0)).toThrow();
  expect(() => pasteRooms(document, copyRooms(document, [fixture.switchCopy.room]), -8000, 0)).toThrow();
  expect(document.roomSource.switchCopies).toHaveLength(1);
});

it.each(['door', 'lift'] as const)('copies original native %s rooms and their actors through the native attachment branch', kind => {
  const fixture = movingControllerCopyFixture(kind);
  const document = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [fixture.copy])));
  const templates = new Set(fixture.added.map(room => room.recordZone));
  const ids = document.roomSource.rooms.filter(room => room.originZone !== null && templates.has(room.originZone)).map(room => room.id);
  const result = pasteRooms(document, copyRooms(document, ids), -8000, 0), pasted = applyCommand(document, result.command);
  expect(buildPlayableLevel(pasted, -1).document.mechanisms.filter(item => item.kind === kind)).toHaveLength(3);
  expect(result.objects.length).toBeGreaterThan(0);
  expect(pasted.roomSource.controllerCopies!.at(-1)!.room).not.toBe(fixture.copy.room);
});

it('remaps an internal copied switch gate and preserves partial native material footprints', () => {
  const fixture = linkedRoomCopyFixture(), result = pasteRooms(fixture.document, copyRooms(fixture.document, fixture.ids), -8000, 0);
  const pasted = applyCommand(fixture.document, result.command), copiedSwitch = pasted.roomSource.switchCopies!.at(-1)!;
  const motion = pasted.roomSource.controllerCopies!.at(-1)!;
  expect(new BinaryReader(new Uint8Array(motion.header), 'copied gate').u16(14)).toBe(1 << (copiedSwitch.slot + 4));
  expect(new BinaryReader(new Uint8Array(pasted.roomSource.controllerCopies![0]!.header), 'original gate').u16(14)).toBe(1 << (fixture.switchCopy.slot + 4));
  const partial = nativePartialRoomCopyFixture(), copy = pasteRooms(partial.document, copyRooms(partial.document, [partial.copy.id]), -8000, 0);
  const built = buildPlayableLevel(applyCommand(partial.document, copy.command), -1), cell = built.topology!.cells.find(cell => cell.sourceId === copy.ids[0])!;
  const polygon = built.document.zones[cell.id]!.lower.find(record => record.type === 1)!.polygon!;
  expect(polygon.points).toHaveLength(partial.polygon.points.length);
  expect(polygon.points.map(id => built.document.points[id])).toEqual(partial.polygon.points.map(id => expect.objectContaining({
    x: partial.source.points[id]!.x + partial.dx - 8000, z: partial.source.points[id]!.z + partial.dz,
  })));
});

it('rejects switch-capacity exhaustion with the entire destination unchanged', () => {
  const fixture = switchCopyFixture();
  let document = applyCommand(fixture.document, editRoomSource(setRoomSwitchCopies(fixture.document, [fixture.switchCopy])));
  const ids = document.roomSource.rooms.filter(room => room.originZone === null).map(room => room.id), clipboard = copyRooms(document, ids);
  for (const [x, z] of [[-8000, 0], [8000, 0], [0, -8000], [-8000, -8000], [8000, -8000], [0, 8000]])
    document = applyCommand(document, pasteRooms(document, clipboard, x!, z!).command);
  expect(buildPlayableLevel(document, -1).document.switches.filter(item => item.zone >= 0)).toHaveLength(8);
  const source = document.roomSource, files = document.files, links = document.sourceTeleports;
  expect(() => pasteRooms(document, clipboard, 8000, 8000)).toThrow(/native limit is eight/);
  expect(document.roomSource).toBe(source); expect(document.files).toBe(files); expect(document.sourceTeleports).toBe(links);
});

it('does not assign a copied switch to an unused condition bit reserved by an existing gate', () => {
  const fixture = linkedRoomCopyFixture(), firstFree = fixture.document.switches.find(item => item.zone < 0 && item.id !== fixture.switchCopy.slot)!.id;
  const original = fixture.document.mechanisms.find(item => item.kind === 'door')!;
  const document = applyCommand(fixture.document, setMechanismRules(fixture.document, 'door', original.id, 1 << (firstFree + 4), original.openRule, original.closeRule));
  const result = pasteRooms(document, copyRooms(fixture.document, fixture.ids), -8000, 0), pasted = applyCommand(document, result.command);
  const copy = pasted.roomSource.switchCopies!.at(-1)!;
  expect(copy.slot).not.toBe(firstFree);
  expect(pasted.mechanisms[0]!.conditions).toBe(1 << (firstFree + 4));
  expect(new BinaryReader(new Uint8Array(pasted.roomSource.controllerCopies!.at(-1)!.header), 'new gate').u16(14)).toBe(1 << (copy.slot + 4));
});

it('captures required motion attachments automatically without changing the requested anchor', () => {
  const fixture = movingControllerCopyFixture('door');
  const document = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [fixture.copy])));
  const captured = copyRoomGroup(document, [fixture.copy.room]);
  expect(captured.requestedIds).toEqual([fixture.copy.room]); expect(captured.ids[0]).toBe(fixture.copy.room);
  expect(new Set(captured.ids)).toEqual(new Set(fixture.added.map(room => room.id)));
  const result = pasteRooms(document, captured, -8000, 0);
  expect(buildPlayableLevel(applyCommand(document, result.command), -1).document.mechanisms).toHaveLength(3);
});
