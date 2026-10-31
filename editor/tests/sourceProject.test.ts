import { describe, expect, it } from 'vitest';
import { zipSync, unzipSync, strToU8 } from 'fflate';
import { openLevel, exportLevel } from '../src/model/level';
import { encodeProject, decodeProject, encodeGameLevel } from '../src/model/project';
import { execute, undo, redo, editRoomSource, setZoneProperty, setPlaytestSlot, setExitZone } from '../src/model/commands';
import { beginRoomDraft, setDraftCorners, commitRoomDraft, roomSourceSnapshot, restoreRoomSource, roomSourceMatchesLevel, removeSourceRooms, moveSourceCorner } from '../src/model/roomSource';
import { originalFiles } from './helpers';

const outline = [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20064 }, { x: 20064, z: 20064 }, { x: 20064, z: 20256 }, { x: 20000, z: 20256 }];

describe('editable source project preservation', () => {
  it('migrates version 1 wall identities only where the old geometry provides evidence', () => {
    const document = openLevel('legacy source', originalFiles(1)), current = document.roomSource;
    const legacy = { ...roomSourceSnapshot(current), version: 1,
      rooms: current.rooms.map(({ wallSources: _, ...room }) => room) };
    expect(restoreRoomSource(document, legacy)).toEqual(current);
    const changed = { ...legacy, rooms: legacy.rooms.map((room, i) => i ? room : {
      ...room, vertices: room.vertices.map(point => ({ x: point.x + 20000, z: point.z + 20000 })) }) };
    expect(restoreRoomSource(document, changed).rooms[0]!.wallSources.every(line => line === null)).toBe(true);
  });
  it('saves moved wall identities and rejects incomplete or unrelated wall references', () => {
    const document = openLevel('wall identities', originalFiles(1)), room = document.roomSource.rooms.find(room => room.vertices.length === 4)!;
    const isolated = removeSourceRooms(document.roomSource, document.roomSource.rooms.filter(other => other.id !== room.id).map(other => other.id)).after;
    const point = room.vertices[0]!, change = moveSourceCorner(isolated, room.id, 0, { x: point.x + 1, z: point.z + 1 });
    const edited = { ...document, roomSource: change.after };
    expect(decodeProject(encodeProject(edited)).roomSource).toEqual(change.after);
    const snapshot = roomSourceSnapshot(change.after);
    expect(() => restoreRoomSource(document, { ...snapshot, rooms: snapshot.rooms.map(room => ({ ...room, wallSources: [] })) })).toThrow(/incomplete wall-source/);
    expect(() => restoreRoomSource(document, { ...snapshot, rooms: snapshot.rooms.map(room => ({ ...room, wallSources: room.wallSources.map(() => 32767) })) })).toThrow(/outside its settings template/);
  });
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('round trips level %i source independently of all three original game files', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level), level - 1), restored = decodeProject(encodeProject(document));
    expect(restored.roomSource).toEqual(document.roomSource);
    expect(roomSourceMatchesLevel(restored)).toBe(true);
    expect(exportLevel(restored)).toEqual(originalFiles(level));
    expect(Object.isFrozen(restored.roomSource.rooms[0]!.vertices)).toBe(true);
    const legacy = { ...roomSourceSnapshot(document.roomSource), version: 1,
      rooms: document.roomSource.rooms.map(({ wallSources: _, ...room }) => room) };
    expect(restoreRoomSource(document, legacy)).toEqual(document.roomSource);
  });
  it('saves a committed concave outline without replacing it with generated convex cells or a partial draft', () => {
    const document = openLevel('Authored geometry', originalFiles(1)), source = document.roomSource;
    const change = commitRoomDraft(source, setDraftCorners(beginRoomDraft(document, source, 0), outline));
    const edited = execute({ document, past: [], future: [] }, editRoomSource(change)).document;
    const draft = setDraftCorners(beginRoomDraft(edited, edited.roomSource, 0), outline.slice(0, 2));
    expect(draft.vertices).toHaveLength(2);
    const restored = decodeProject(encodeProject(edited));
    expect(restored.roomSource).toEqual(change.after);
    expect(restored.roomSource.rooms.at(-1)!.vertices).toHaveLength(6);
    expect(restored.roomSource.rooms).toHaveLength(source.rooms.length + 1);
    expect(exportLevel(restored)).toEqual(exportLevel(document));
    expect(() => encodeGameLevel(restored)).toThrow(/stale geometry/);
  });
  it('interleaves room edits, record edits, slot and exit setup in one unlimited history', () => {
    const document = openLevel('Source and records', originalFiles(1));
    const change = commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0), outline));
    let history = execute({ document, past: [], future: [] }, editRoomSource(change));
    history = execute(history, setZoneProperty(history.document, 3, 'brightness', 11));
    history = execute(history, setPlaytestSlot(history.document, 8));
    history = execute(history, setExitZone(history.document, 3));
    const restored = decodeProject(encodeProject(history.document));
    expect(restored).toMatchObject({ playtestSlot: 8, exitZone: 3 });
    expect(restored.roomSource).toEqual(change.after); expect(restored.zones[3]!.brightness).toBe(11);
    for (let i = 0; i < 4; i++) history = undo(history);
    expect(history.document.roomSource).toBe(document.roomSource);
    expect(exportLevel(history.document)).toEqual(originalFiles(1));
    expect(roomSourceMatchesLevel(history.document)).toBe(true);
    for (let i = 0; i < 4; i++) history = redo(history);
    expect(history.document.roomSource).toBe(change.after);
    expect(() => execute(history, editRoomSource(change))).toThrow(/room source changed/);
  });
  it('preserves an empty/deleted source so an unfinished project can still be recovered', () => {
    const document = openLevel('Empty source', originalFiles(1));
    const deleted = removeSourceRooms(document.roomSource, document.roomSource.rooms.map(room => room.id));
    const edited = execute({ document, past: [], future: [] }, editRoomSource(deleted)).document;
    expect(decodeProject(encodeProject(edited)).roomSource.rooms).toEqual([]);
  });
  it('does not create history or dirty source for an unchanged outline drawn in reverse', () => {
    const document = openLevel('Same outline', originalFiles(1)), room = document.roomSource.rooms[0]!;
    const change = commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, room.recordZone, room.id), room.vertices.toReversed()));
    const history = { document, past: [], future: [] };
    expect(execute(history, editRoomSource(change))).toBe(history);
  });
  it('rejects invalid source identities, references, outlines and missing source entries', () => {
    const document = openLevel('Checked source', originalFiles(1));
    const snapshot = () => structuredClone(roomSourceSnapshot(document.roomSource));
    expect(() => restoreRoomSource(document, { ...snapshot(), unknownAuthoringData: 'must not disappear' })).toThrow(/unsupported fields/);
    const duplicate = snapshot(); duplicate.rooms = [...duplicate.rooms, duplicate.rooms[0]!];
    expect(() => restoreRoomSource(document, duplicate)).toThrow(/duplicate room identity/);
    const reused = snapshot(); reused.nextId = 0;
    expect(() => restoreRoomSource(document, reused)).toThrow(/reuse an existing room/);
    const missing = snapshot(); missing.rooms = [{ ...missing.rooms[0]!, recordZone: -1 }];
    expect(() => restoreRoomSource(document, missing)).toThrow(/missing surface record/);
    const crossing = snapshot(); crossing.rooms = [{ ...crossing.rooms[0]!, vertices: [outline[0]!, outline[2]!, outline[1]!, outline[3]!], wallSources: [null, null, null, null] }];
    expect(() => restoreRoomSource(document, crossing)).toThrow(/crosses itself/);
    const entries = unzipSync(encodeProject(document)); delete entries['room-source.json'];
    expect(() => decodeProject(zipSync(entries))).toThrow(/room source is missing/);
    entries['room-source.json'] = strToU8('{');
    expect(() => decodeProject(zipSync(entries))).toThrow(/not valid JSON/);
    expect(document.roomSource.rooms).toHaveLength(138);
  });
});
