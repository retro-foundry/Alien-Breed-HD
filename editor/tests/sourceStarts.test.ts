import { expect, it } from 'vitest';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { BinaryReader } from '../src/model/binary';
import { applyCommand, editRoomSource, execute, moveObject, redo, removeObject, setSourceStart, setStart, setZoneProperty, undo } from '../src/model/commands';
import { openLevel } from '../src/model/level';
import { decodeProject, encodeGameLevel, encodeProject } from '../src/model/project';
import { buildRoomDataReferences } from '../src/model/roomDataReferences';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { restoreSourceStarts } from '../src/model/sourceStarts';
import { originalFiles } from './helpers';

function fixture() {
  const original = openLevel('Authored start', originalFiles(1));
  const change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]));
  const document = applyCommand(original, editRoomSource(change));
  return { original, document, room: change.after.rooms.at(-1)! };
}

it('saves source-only starts, preserves them through other edits, and undoes/redoes without changing import bytes', () => {
  const { original, document, room } = fixture();
  const command = setSourceStart(document, 1, room.id, 20032, 20064);
  let history = execute({ document, past: [], future: [] }, command);
  expect(history.document.files).toEqual(original.files);
  expect(history.document.starts).toEqual(original.starts);
  expect(Object.isFrozen(history.document.sourceStarts)).toBe(true);
  const restored = decodeProject(encodeProject(history.document));
  expect(restored.sourceStarts).toEqual(history.document.sourceStarts);
  expect(restored.roomSource).toEqual(document.roomSource);
  const entries = unzipSync(encodeProject(history.document));
  expect(JSON.parse(strFromU8(entries['alien-breed-project.json']!)).version).toBe(5);
  const other = applyCommand(history.document, setZoneProperty(history.document, 0, 'brightness', 3));
  expect(other.sourceStarts).toBe(history.document.sourceStarts);
  expect(execute(history, setSourceStart(history.document, 1, room.id, 20032, 20064))).toBe(history);
  expect(() => applyCommand(history.document, command)).toThrow(/player starts changed/);
  history = undo(history); expect(history.document.sourceStarts).toBeUndefined();
  history = redo(history); expect(history.document.sourceStarts).toEqual(restored.sourceStarts);
  expect(() => encodeGameLevel(history.document)).toThrow(/cannot yet compile/);
});

it('assigns both players to the containing convex cells and protects their authored avatar positions', () => {
  const { document, room } = fixture();
  const source = { ...document, roomSource: { ...document.roomSource, rooms: document.roomSource.rooms.map(item => item.id !== room.id ? item
    : { ...item, vertices: [{ x: 20000, z: 20000 }, { x: 20000, z: 20128 }, { x: 20064, z: 20128 },
      { x: 20064, z: 20064 }, { x: 20128, z: 20064 }, { x: 20128, z: 20000 }], wallSources: [null, null, null, null, null, null] }) } };
  let placed = applyCommand(source, setSourceStart(source, 1, room.id, 20032, 20100));
  placed = applyCommand(placed, setSourceStart(placed, 2, room.id, 20100, 20032));
  const topology = compileRoomTopology(placed, placed.roomSource);
  expect(topology.sourceCells.get(room.id)!.length).toBeGreaterThan(1);
  const built = buildRoomDataReferences(placed, placed.roomSource, topology), bytes = new BinaryReader(built.data, 'starts');
  for (const start of placed.sourceStarts!) {
    const offset = start.player === 1 ? 0 : 6, avatarOffset = bytes.i32(start.player === 1 ? 46 : 50);
    expect(bytes.i16(offset)).toBe(start.x); expect(bytes.i16(offset + 2)).toBe(start.z);
    expect(topology.cells[bytes.i16(offset + 4)]!.sourceId).toBe(room.id);
    const avatar = placed.objects.find(item => item.offset === avatarOffset)!;
    expect(bytes.i16(avatarOffset + 12)).toBe(bytes.i16(offset + 4));
    expect(bytes.i32(avatar.pointOffset) / 65536).toBe(start.x);
    expect(bytes.i32(avatar.pointOffset + 4) / 65536).toBe(start.z);
    expect(() => moveObject(placed, avatar.id, avatar.x, avatar.z, avatar.zone)).toThrow(/Select the player in Starts/);
    expect(() => removeObject(placed, avatar.id)).toThrow(/Select the player in Starts/);
  }
});

it('compiles a start and its multiplayer avatar into a newly drawn room after source reordering', () => {
  const { document, room } = fixture();
  const placed = applyCommand(document, setSourceStart(document, 1, room.id, 20032, 20064));
  const source = { ...placed, roomSource: { ...placed.roomSource, rooms: placed.roomSource.rooms.toReversed() } };
  const wall = document.zones[0]!.lower.find(record => record.wall)!.wall!;
  const built = buildSourceLevel(source, { resolvedExit: -1,
    newRoomGraphics: new Map([[room.id, { walls: { lower: [wall] }, wallInsertion: { lower: 0 } }]]),
    newRoomLighting: new Map([[room.id, { kind: 'copy', source: 0 }]]) });
  const start = built.document.starts[0]!;
  expect(start).toMatchObject({ player: 1, x: 20032, z: 20064 });
  expect(built.topology.cells[start.zone]!.sourceId).toBe(room.id);
  const bytes = new BinaryReader(built.document.files.data, 'compiled'), avatarOffset = bytes.i32(46);
  const avatar = built.document.objects.find(object => object.offset === avatarOffset)!;
  expect(avatar).toMatchObject({ zone: start.zone, x: start.x, z: start.z });
  expect(bytes.i16(avatarOffset + 26)).toBe(start.zone);
  expect(built.document.sourceStarts).toBeUndefined();
  expect(decodeProject(encodeProject(built.document)).starts).toEqual(built.document.starts);
  expect(encodeGameLevel(built.document).length).toBeGreaterThan(0);
});

it('retains dangling references for recovery but blocks deleted rooms and starts outside reshaped geometry', () => {
  const { document, room } = fixture();
  const placed = applyCommand(document, setSourceStart(document, 1, room.id, 20032, 20064));
  const deleted = applyCommand(placed, editRoomSource(removeSourceRooms(placed.roomSource, [room.id])));
  const restored = decodeProject(encodeProject(deleted));
  expect(restored.sourceStarts).toEqual(placed.sourceStarts);
  expect(() => buildRoomDataReferences(restored, restored.roomSource, compileRoomTopology(restored, restored.roomSource))).toThrow(/Player 1 start.*deleted/);
  const moved = { ...placed, roomSource: { ...placed.roomSource, rooms: placed.roomSource.rooms.map(item => item.id === room.id
    ? { ...item, vertices: item.vertices.map(point => ({ x: point.x + 256, z: point.z })) } : item) } };
  expect(() => buildRoomDataReferences(moved, moved.roomSource, compileRoomTopology(moved, moved.roomSource))).toThrow(/Player 1 start.*outside editable room/);
  expect(() => setSourceStart(document, 1, room.id, 0, 0)).toThrow(/inside/);
  expect(() => setSourceStart(document, 1, room.id, 20032.5, 20064)).toThrow(/Player start X/);
});

it('lets legacy start controls replace an authored start instead of editing an ignored baseline', () => {
  const document = openLevel('original', originalFiles(1)), start = document.starts[0]!;
  const room = document.roomSource.rooms.find(room => room.originZone === start.zone)!;
  const placed = applyCommand(document, setSourceStart(document, 1, room.id, start.x, start.z));
  const next = applyCommand(placed, setStart(placed, 1, start.x, start.z, start.zone));
  expect(next.sourceStarts).toEqual(placed.sourceStarts);
  expect(next.files).toEqual(document.files);
  expect(() => encodeGameLevel(placed)).toThrow(/cannot yet compile/);
});

it('rejects duplicate players, unsupported fields and damaged or incorrectly versioned project data', () => {
  const start = { player: 1, room: 'room-0', x: 0, z: 0 };
  expect(() => restoreSourceStarts([start, start])).toThrow(/duplicate/);
  expect(() => restoreSourceStarts([{ ...start, surprise: true }])).toThrow(/Invalid/);
  expect(() => restoreSourceStarts([{ ...start, x: 32768 }])).toThrow(/Player start X/);
  const { document, room } = fixture();
  const placed = applyCommand(document, setSourceStart(document, 1, room.id, 20032, 20064));
  const entries = unzipSync(encodeProject(placed));
  delete entries['player-starts.json'];
  expect(() => decodeProject(zipSync(entries))).toThrow(/starts are missing/);
  entries['player-starts.json'] = strToU8('{');
  expect(() => decodeProject(zipSync(entries))).toThrow(/not valid JSON/);
  entries['player-starts.json'] = strToU8(JSON.stringify([start]));
  const manifest = JSON.parse(strFromU8(entries['alien-breed-project.json']!));
  entries['alien-breed-project.json'] = strToU8(JSON.stringify({ ...manifest, version: 4 }));
  expect(() => decodeProject(zipSync(entries))).toThrow(/require project format version 5/);
  delete entries['alien-breed-project.json'];
  expect(() => decodeProject(zipSync(entries))).toThrow(/one saved project/);
});
