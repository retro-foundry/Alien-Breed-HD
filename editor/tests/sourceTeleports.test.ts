import { expect, it } from 'vitest';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { BinaryReader } from '../src/model/binary';
import { applyCommand, editRoomSource, execute, redo, setSourceTeleport, setTeleport, setZoneProperty, undo } from '../src/model/commands';
import { openLevel } from '../src/model/level';
import { decodeProject, encodeGameLevel, encodeProject } from '../src/model/project';
import { buildRoomDataReferences } from '../src/model/roomDataReferences';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { editableTeleport, restoreSourceTeleports } from '../src/model/sourceTeleports';
import { originalFiles } from './helpers';

function fixture() {
  const original = openLevel('Teleporter authoring', originalFiles(1));
  let document = original;
  for (const x of [20000, 21000]) document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource,
    setDraftCorners(beginRoomDraft(document, document.roomSource, 0), [{ x, z: 20000 }, { x: x + 128, z: 20000 }, { x: x + 128, z: 20128 }, { x, z: 20128 }]))));
  return { original, document, from: document.roomSource.rooms.at(-2)!, to: document.roomSource.rooms.at(-1)! };
}

it('retains authored teleporters across undo, other edits and project recovery without changing import bytes', () => {
  const { original, document, from, to } = fixture();
  const command = setSourceTeleport(document, from.id, to.id, 21064, 20064);
  let history = execute({ document, past: [], future: [] }, command);
  const restored = decodeProject(encodeProject(history.document));
  expect(restored.sourceTeleports).toEqual([{ room: from.id, destination: to.id, x: 21064, z: 20064 }]);
  expect(restored.sourceStarts).toBeUndefined(); expect(restored.sourceObjects).toBeUndefined();
  expect(restored.files).toEqual(original.files);
  expect(editableTeleport(restored, restored.roomSource.rooms.find(room => room.id === from.id)!)).toMatchObject({ destination: to.id, x: 21064, z: 20064 });
  expect(applyCommand(history.document, setZoneProperty(history.document, 0, 'brightness', 2)).sourceTeleports).toBe(history.document.sourceTeleports);
  expect(execute(history, setSourceTeleport(history.document, from.id, to.id, 21064, 20064))).toBe(history);
  expect(() => applyCommand(history.document, command)).toThrow(/teleporter links changed/);
  history = undo(history); expect(history.document.sourceTeleports).toBeUndefined();
  history = redo(history); expect(history.document.sourceTeleports).toEqual(restored.sourceTeleports);
  expect(() => encodeGameLevel(history.document)).toThrow(/stale geometry or gameplay/);
});

it('writes the containing destination cell and landing coordinates into every cell of a split source room', () => {
  const { document, from, to } = fixture();
  const concave = [{ x: 20000, z: 20000 }, { x: 20000, z: 20128 }, { x: 20064, z: 20128 },
    { x: 20064, z: 20064 }, { x: 20128, z: 20064 }, { x: 20128, z: 20000 }];
  const source = { ...document, roomSource: { ...document.roomSource, rooms: document.roomSource.rooms.map(room => room.id === from.id || room.id === to.id
    ? { ...room, vertices: concave.map(point => ({ ...point, x: point.x + (room.id === to.id ? 1000 : 0) })), wallSources: concave.map(() => null) } : room).toReversed() } };
  const placed = applyCommand(source, setSourceTeleport(source, from.id, to.id, 21100, 20032));
  const topology = compileRoomTopology(placed, placed.roomSource), built = buildRoomDataReferences(placed, placed.roomSource, topology);
  expect(topology.sourceCells.get(from.id)!.length).toBeGreaterThan(1); expect(topology.sourceCells.get(to.id)!.length).toBeGreaterThan(1);
  for (const id of topology.sourceCells.get(from.id)!) {
    const header = new BinaryReader(built.headers.get(id)!, 'source teleport');
    expect(topology.cells[header.i16(38)]!.sourceId).toBe(to.id);
    expect(header.i16(40)).toBe(21100); expect(header.i16(42)).toBe(20032);
  }
  const disabled = applyCommand(placed, setSourceTeleport(placed, from.id, null, 21100, 20032));
  const rebuilt = buildRoomDataReferences(disabled, disabled.roomSource, topology);
  for (const id of topology.sourceCells.get(from.id)!) expect(new BinaryReader(rebuilt.headers.get(id)!, 'disabled').i16(38)).toBe(-1);
});

it('retains broken destination dependencies for repair, including inherited links, and ignores deleted emitters', () => {
  const { document, from, to } = fixture();
  const placed = applyCommand(document, setSourceTeleport(document, from.id, to.id, 21064, 20064));
  const deleted = applyCommand(placed, editRoomSource(removeSourceRooms(placed.roomSource, [to.id])));
  const restored = decodeProject(encodeProject(deleted));
  expect(restored.sourceTeleports).toEqual(placed.sourceTeleports);
  expect(() => buildRoomDataReferences(restored, restored.roomSource, compileRoomTopology(restored, restored.roomSource))).toThrow(/teleport.*deleted/);
  const disabled = applyCommand(restored, setSourceTeleport(restored, from.id, null, 21064, 20064));
  expect(() => buildRoomDataReferences(disabled, disabled.roomSource, compileRoomTopology(disabled, disabled.roomSource))).not.toThrow();
  const noEmitter = applyCommand(placed, editRoomSource(removeSourceRooms(placed.roomSource, [from.id])));
  expect(() => buildRoomDataReferences(noEmitter, noEmitter.roomSource, compileRoomTopology(noEmitter, noEmitter.roomSource))).not.toThrow();
  const legacy = openLevel('Level 9', originalFiles(9)), pad = legacy.zones.find(zone => zone.teleport.zone >= 0 && zone.id !== zone.teleport.zone)!;
  const target = legacy.roomSource.rooms.find(room => room.originZone === pad.teleport.zone)!;
  const legacyDeleted = { ...legacy, roomSource: removeSourceRooms(legacy.roomSource, [target.id]).after };
  const padRoom = legacyDeleted.roomSource.rooms.find(room => room.originZone === pad.id)!;
  expect(editableTeleport(legacyDeleted, padRoom).destination).toBeUndefined();
  const originalPad = legacy.roomSource.rooms.find(room => room.originZone === pad.id)!;
  const inheritedDisabled = applyCommand(legacy, setSourceTeleport(legacy, originalPad.id, null, pad.teleport.x, pad.teleport.z));
  const inheritedTopology = compileRoomTopology(inheritedDisabled, inheritedDisabled.roomSource);
  const inheritedBuild = buildRoomDataReferences(inheritedDisabled, inheritedDisabled.roomSource, inheritedTopology);
  for (const id of inheritedTopology.sourceCells.get(originalPad.id)!)
    expect(new BinaryReader(inheritedBuild.headers.get(id)!, 'disabled inherited link').i16(38)).toBe(-1);
  expect(legacy.zones.find(zone => zone.id === pad.id)!.teleport.zone).toBeGreaterThanOrEqual(0);
});

it('validates destination outlines and routes numeric legacy controls to their authored link', () => {
  const document = openLevel('original', originalFiles(1)), room = document.roomSource.rooms[0]!, start = document.starts[0]!;
  const target = document.roomSource.rooms.find(room => room.originZone === start.zone)!;
  const placed = applyCommand(document, setSourceTeleport(document, room.id, target.id, start.x, start.z));
  const disabled = applyCommand(placed, setTeleport(placed, room.originZone!, -1, start.x, start.z));
  expect(disabled.sourceTeleports).toEqual([{ room: room.id, destination: null, x: start.x, z: start.z }]);
  expect(disabled.files).toEqual(document.files);
  const { document: drawn, from, to } = fixture();
  expect(() => setSourceTeleport(drawn, from.id, to.id, 0, 0)).toThrow(/inside/);
  expect(() => setSourceTeleport(drawn, from.id, to.id, 21064.5, 20064)).toThrow(/Teleport destination X/);
  expect(() => setSourceTeleport(drawn, 'missing', null, 0, 0)).toThrow(/existing editable room/);
  const moved = { ...drawn, roomSource: { ...drawn.roomSource, rooms: drawn.roomSource.rooms.map(room => room.id === to.id
    ? { ...room, vertices: room.vertices.map(point => ({ ...point, x: point.x + 256 })) } : room) } };
  const stale = { ...moved, sourceTeleports: restoreSourceTeleports([{ room: from.id, destination: to.id, x: 21064, z: 20064 }]) };
  expect(() => buildRoomDataReferences(stale, stale.roomSource, compileRoomTopology(stale, stale.roomSource))).toThrow(/teleport.*outside editable room/);
});

it('rejects duplicate links, unsupported fields and missing or incompatible archive records', () => {
  const { document, from, to } = fixture(), link = { room: from.id, destination: to.id, x: 21064, z: 20064 };
  expect(() => restoreSourceTeleports([link, link])).toThrow(/Duplicate/);
  expect(() => restoreSourceTeleports([{ ...link, unknown: true }])).toThrow(/Invalid/);
  expect(() => restoreSourceTeleports([{ ...link, destination: '' }])).toThrow(/Invalid/);
  expect(() => restoreSourceTeleports([{ ...link, z: Infinity }])).toThrow(/Teleport destination Z/);
  const placed = applyCommand(document, setSourceTeleport(document, from.id, to.id, link.x, link.z));
  const entries = unzipSync(encodeProject(placed)), manifest = JSON.parse(strFromU8(entries['alien-breed-project.json']!));
  expect(manifest.version).toBe(7);
  delete entries['teleporter-links.json']; expect(() => decodeProject(zipSync(entries))).toThrow(/links are missing/);
  entries['teleporter-links.json'] = strToU8('{'); expect(() => decodeProject(zipSync(entries))).toThrow(/not valid JSON/);
  entries['teleporter-links.json'] = strToU8(JSON.stringify([link]));
  entries['alien-breed-project.json'] = strToU8(JSON.stringify({ ...manifest, version: 4 }));
  expect(() => decodeProject(zipSync(entries))).toThrow(/require project format version 7/);
});
