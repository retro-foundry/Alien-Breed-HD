import { expect, it } from 'vitest';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { BinaryReader } from '../src/model/binary';
import { applyCommand, editRoomSource, execute, moveObject, redo, removeObject, setSourceObject, setSourceStart, setZoneProperty, undo } from '../src/model/commands';
import { openLevel } from '../src/model/level';
import { decodeProject, encodeGameLevel, encodeProject } from '../src/model/project';
import { buildRoomDataReferences } from '../src/model/roomDataReferences';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { editableObjects, objectIsActive, restoreSourceObjects, validateSourceObjects } from '../src/model/sourceObjects';
import { originalFiles } from './helpers';

function fixture() {
  const original = openLevel('Object authoring', originalFiles(1));
  const object = original.objects.find(object => object.type === 1 && object.zone >= 0 && !object.upper)!;
  expect(object).toBeDefined();
  const change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]));
  return { original, object, document: applyCommand(original, editRoomSource(change)), room: change.after.rooms.at(-1)! };
}

it('saves object placements independently of binary attributes, with atomic undo, redo and explicit deletion', () => {
  const { original, object, document, room } = fixture();
  const command = setSourceObject(document, object.id, room.id, 20032.5, 20064.25);
  let history = execute({ document, past: [], future: [] }, command);
  expect(history.document.files).toEqual(original.files);
  expect(editableObjects(history.document).find(item => item.id === object.id)).toMatchObject({ room: room.id, zone: -1, x: 20032.5, z: 20064.25 });
  expect(objectIsActive(editableObjects(history.document).find(item => item.id === object.id)!)).toBe(true);
  expect(execute(history, setSourceObject(history.document, object.id, room.id, 20032.5, 20064.25))).toBe(history);
  expect(() => applyCommand(history.document, command)).toThrow(/object placements changed/);
  const recovered = decodeProject(encodeProject(history.document));
  expect(recovered.sourceObjects).toEqual(history.document.sourceObjects);
  expect(recovered.sourceStarts).toBeUndefined();
  expect(recovered.files).toEqual(original.files);
  const unrelated = applyCommand(history.document, setZoneProperty(history.document, 0, 'brightness', 7));
  expect(unrelated.sourceObjects).toBe(history.document.sourceObjects);
  history = execute(history, removeObject(history.document, object.id));
  expect(objectIsActive(editableObjects(history.document).find(item => item.id === object.id)!)).toBe(false);
  const topology = compileRoomTopology(history.document, history.document.roomSource);
  const bytes = new BinaryReader(buildRoomDataReferences(history.document, history.document.roomSource, topology).data, 'removed');
  expect(bytes.i16(object.offset + 12)).toBe(-1); expect(bytes.i16(object.offset + 26)).toBe(-1);
  history = undo(history); expect(history.document.sourceObjects).toEqual(recovered.sourceObjects);
  history = undo(history); expect(history.document.sourceObjects).toBeUndefined();
  history = redo(history); expect(history.document.sourceObjects).toEqual(recovered.sourceObjects);
  expect(() => encodeGameLevel(history.document)).toThrow(/stale geometry or gameplay/);
});

it('compiles an authored object and start in a new room, preserving all other object bytes and point slots', () => {
  const { original, document, room, object } = fixture();
  let source = applyCommand(document, setSourceObject(document, object.id, room.id, 20032.5, 20064.25));
  source = applyCommand(source, setSourceStart(source, 1, room.id, 20064, 20032));
  source = decodeProject(encodeProject(source));
  const wall = original.zones[0]!.lower.find(record => record.wall)!.wall!;
  const built = buildSourceLevel(source, { resolvedExit: -1,
    newRoomGraphics: new Map([[room.id, { walls: { lower: [wall] }, wallInsertion: { lower: 0 } }]]),
    newRoomLighting: new Map([[room.id, { kind: 'copy', source: 0 }]]) });
  const compiled = built.document.objects.find(item => item.id === object.id)!;
  expect(compiled).toMatchObject({ x: 20032.5, z: 20064.25, type: object.type, y: object.y, health: object.health, cid: object.cid });
  expect(built.topology.cells[compiled.zone]!.sourceId).toBe(room.id);
  const bytes = new BinaryReader(built.document.files.data, 'compiled');
  expect(bytes.i16(object.offset + 26)).toBe(compiled.zone);
  const retained = original.files.data.slice(object.offset, object.offset + 64), view = new DataView(retained.buffer);
  view.setInt16(12, compiled.zone); view.setInt16(26, compiled.zone);
  expect(built.document.files.data.slice(object.offset, object.offset + 64)).toEqual(retained);
  for (const other of original.objects) if (other.id !== object.id) {
    const avatar = new BinaryReader(original.files.data, 'source').i32(46);
    if (other.offset === avatar) continue;
    expect(built.document.files.data.slice(other.pointOffset, other.pointOffset + 8))
      .toEqual(original.files.data.slice(other.pointOffset, other.pointOffset + 8));
  }
  expect(built.document.sourceObjects).toBeUndefined();
  expect(decodeProject(encodeProject(built.document)).objects).toEqual(built.document.objects);
  expect(encodeGameLevel(built.document).length).toBeGreaterThan(0);
});

it('keeps deleted-room dependencies recoverable and rejects placements outside reshaped rooms', () => {
  const { document, room, object } = fixture();
  const placed = applyCommand(document, setSourceObject(document, object.id, room.id, 20032, 20064));
  const deleted = applyCommand(placed, editRoomSource(removeSourceRooms(placed.roomSource, [room.id])));
  const recovered = decodeProject(encodeProject(deleted));
  expect(recovered.sourceObjects).toEqual(placed.sourceObjects);
  expect(() => buildRoomDataReferences(recovered, recovered.roomSource, compileRoomTopology(recovered, recovered.roomSource))).toThrow(new RegExp(`Object ${object.id}.*deleted`));
  const moved = { ...placed, roomSource: { ...placed.roomSource, rooms: placed.roomSource.rooms.map(item => item.id !== room.id ? item
    : { ...item, vertices: item.vertices.map(point => ({ x: point.x + 256, z: point.z })) }) } };
  expect(() => buildRoomDataReferences(moved, moved.roomSource, compileRoomTopology(moved, moved.roomSource))).toThrow(/outside editable room/);
  expect(() => setSourceObject(document, object.id, room.id, 0, 0)).toThrow(/inside/);
  const removed = applyCommand(recovered, removeObject(recovered, object.id));
  expect(() => buildRoomDataReferences(removed, removed.roomSource, compileRoomTopology(removed, removed.roomSource))).not.toThrow();
});

it('routes legacy object controls to the authored placement and protects avatars and shared point identities', () => {
  const { document, room, object } = fixture();
  const placed = applyCommand(document, setSourceObject(document, object.id, room.id, 20032, 20064));
  const target = document.roomSource.rooms.find(room => room.originZone === object.zone)!;
  const moved = applyCommand(placed, moveObject(placed, object.id, object.x, object.z, object.zone));
  expect(moved.sourceObjects).toEqual([{ id: object.id, room: target.id, x: object.x, z: object.z }]);
  expect(moved.files).toEqual(document.files);
  const avatarOffset = new BinaryReader(document.files.data, 'source').i32(46), avatar = document.objects.find(item => item.offset === avatarOffset)!;
  expect(() => setSourceObject(document, avatar.id, room.id, 20032, 20064)).toThrow(/Select the player in Starts/);
  const shared = { ...document, objects: document.objects.map(item => item.id === avatar.id ? { ...item, pointOffset: object.pointOffset } : item) };
  expect(() => setSourceObject(shared, object.id, room.id, 20032, 20064)).toThrow(/shares a position slot/);
  const upper = { ...document, objects: document.objects.map(item => item.id === object.id ? { ...item, upper: true } : item) };
  expect(() => setSourceObject(upper, object.id, room.id, 20032, 20064)).toThrow(/upper floor/);
});

it('accepts ordinary decimals, rejects invalid coordinates and checks project source schema', () => {
  const { document, room, object } = fixture(), item = { id: object.id, room: room.id, x: 20032, z: 20064 };
  expect(() => restoreSourceObjects([item, item])).toThrow(/Duplicate/);
  expect(() => restoreSourceObjects([{ ...item, unexpected: true }])).toThrow(/Invalid/);
  for (const x of [NaN, Infinity, 32768, -32768 - 1 / 65536]) expect(() => restoreSourceObjects([{ ...item, x }])).toThrow(/Object X/);
  expect(restoreSourceObjects([{ ...item, x: 32768 - 1 / 65536 }])[0]!.x).toBe(32768 - 1 / 65536);
  expect(() => validateSourceObjects(document, [{ ...item, id: 255 }])).toThrow(/no original record/);
  const placed = applyCommand(document, setSourceObject(document, object.id, room.id, 20032, 20064));
  const entries = unzipSync(encodeProject(placed));
  const manifest = JSON.parse(strFromU8(entries['alien-breed-project.json']!)); expect(manifest.version).toBe(6);
  delete entries['object-placements.json'];
  expect(() => decodeProject(zipSync(entries))).toThrow(/placements are missing/);
  entries['object-placements.json'] = strToU8('{'); expect(() => decodeProject(zipSync(entries))).toThrow(/not valid JSON/);
  entries['object-placements.json'] = strToU8(JSON.stringify([item]));
  entries['alien-breed-project.json'] = strToU8(JSON.stringify({ ...manifest, version: 4 }));
  expect(() => decodeProject(zipSync(entries))).toThrow(/require project format version 6/);
});

it('keeps decimal input readable in source and rounds only compiled coordinates to the native point-table precision', () => {
  const { document, room, object } = fixture();
  const placed = applyCommand(document, setSourceObject(document, object.id, room.id, 20032.1, 20064.3));
  const restored = decodeProject(encodeProject(placed));
  expect(restored.sourceObjects).toEqual([{ id: object.id, room: room.id, x: 20032.1, z: 20064.3 }]);
  const topology = compileRoomTopology(restored, restored.roomSource), built = buildRoomDataReferences(restored, restored.roomSource, topology);
  const bytes = new BinaryReader(built.data, 'compiled decimal position');
  expect(bytes.i32(object.pointOffset)).toBe(Math.round(20032.1 * 65536));
  expect(bytes.i32(object.pointOffset + 4)).toBe(Math.round(20064.3 * 65536));
  expect(restored.files).toEqual(document.files);
});
