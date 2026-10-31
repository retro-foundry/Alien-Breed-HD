import { describe, expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { beginRoomDraft, commitRoomDraft, executeRoomChange, importRoomSource, redoRoomChange, removeSourceRooms, setDraftCorners, undoRoomChange, type RoomHistory } from '../src/model/roomSource';

const outline = [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 128 }, { x: 0, z: 128 }];
const document = openLevel('Level 1', originalFiles(1));
const initial = () => importRoomSource(document);

describe('editable room source and transactional gestures', () => {
  it('imports all original boundaries without changing game bytes or record relationships', () => {
    for (let level = 1; level <= 16; level++) {
      const document = openLevel(`Level ${level}`, originalFiles(level)), source = importRoomSource(document);
      expect(source.rooms).toHaveLength(document.zones.length);
      expect(new Set(source.rooms.map(room => room.id)).size).toBe(source.rooms.length);
      for (const room of source.rooms) {
        const zone = document.zones.find(zone => zone.id === room.recordZone)!;
        expect(room.vertices).toHaveLength(zone.boundaryIds.length);
        expect(Object.isFrozen(room.vertices)).toBe(true);
      }
    }
  });
  it('keeps partial and abandoned gestures out of source and history', () => {
    const source = initial(), history: RoomHistory = { source, past: [], future: [] };
    const draft = setDraftCorners(beginRoomDraft(document, source, 0), outline.slice(0, 2));
    expect(() => commitRoomDraft(source, draft)).toThrow(/three different corners/);
    expect(history.source).toBe(source); expect(history.past).toHaveLength(0);
    expect(source.rooms).toHaveLength(document.zones.length);
    expect(beginRoomDraft(document, source, 0).vertices).toEqual([]);
  });
  it('creates equivalent source for either drawing direction and starting corner', () => {
    const source = initial();
    const commit = (vertices: typeof outline) => commitRoomDraft(source, setDraftCorners(beginRoomDraft(document, source, 0), vertices));
    const expected = commit(outline).after;
    for (let i = 0; i < outline.length; i++) {
      const rotated = [...outline.slice(i), ...outline.slice(0, i)];
      expect(commit(rotated).after).toEqual(expected);
      expect(commit(rotated.toReversed()).after).toEqual(expected);
    }
    expect(expected.rooms.at(-1)?.recordZone).toBe(0);
  });
  it('commits create, outline replacement and multi-delete atomically with undo/redo', () => {
    const source = initial(); let history: RoomHistory = { source, past: [], future: [] };
    history = executeRoomChange(history, commitRoomDraft(source, setDraftCorners(beginRoomDraft(document, source, 0), outline)));
    const created = history.source, room = created.rooms.at(-1)!;
    history = executeRoomChange(history, commitRoomDraft(created, setDraftCorners(beginRoomDraft(document, created, 0, room.id), outline.map(p => ({ x: p.x + 16, z: p.z })))));
    const edited = history.source;
    expect(edited.rooms.at(-1)?.id).toBe(room.id);
    history = executeRoomChange(history, removeSourceRooms(edited, [room.id, source.rooms[0]!.id]));
    expect(history.source.rooms).toHaveLength(source.rooms.length - 1);
    history = undoRoomChange(history); expect(history.source).toBe(edited);
    history = undoRoomChange(history); expect(history.source).toBe(created);
    history = undoRoomChange(history); expect(history.source).toBe(source);
    history = redoRoomChange(redoRoomChange(redoRoomChange(history)));
    expect(history.source.rooms).toHaveLength(source.rooms.length - 1);
    expect(document.files).toEqual(openLevel('original', originalFiles(1)).files);
  });
  it('rejects stale drafts, stale commands, invalid identities and invalid outlines', () => {
    const source = initial(), draft = setDraftCorners(beginRoomDraft(document, source, 0), outline);
    const command = commitRoomDraft(source, draft);
    const history = executeRoomChange({ source, past: [], future: [] }, command);
    expect(() => commitRoomDraft(history.source, draft)).toThrow(/changed while/);
    expect(() => executeRoomChange(history, command)).toThrow(/changed before/);
    expect(() => beginRoomDraft(document, source, -1)).toThrow(/existing room/);
    expect(() => beginRoomDraft(document, source, 0, 'missing')).toThrow(/no longer exists/);
    expect(() => removeSourceRooms(source, ['missing'])).toThrow(/existing rooms/);
    expect(() => setDraftCorners(draft, [{ x: Infinity, z: 0 }])).toThrow(/whole coordinates/);
    expect(() => setDraftCorners(draft, Array.from({ length: 2049 }, () => outline[0]!))).toThrow(/too many corners/);
    expect(() => commitRoomDraft(source, setDraftCorners(draft, [outline[0]!, outline[2]!, outline[1]!, outline[3]!]))).toThrow(/crosses itself/);
  });
  it('drops redo only on a new committed edit and retains more than 100 edits', () => {
    const source = initial(); let history: RoomHistory = { source, past: [], future: [] };
    const id = source.rooms[0]!.id;
    for (let i = 0; i < 150; i++) history = executeRoomChange(history, commitRoomDraft(history.source,
      setDraftCorners(beginRoomDraft(document, history.source, 0, id), outline.map(p => ({ x: p.x + i, z: p.z })))));
    expect(history.past).toHaveLength(150);
    for (let i = 0; i < 150; i++) history = undoRoomChange(history);
    expect(history.source).toBe(source); expect(history.future).toHaveLength(150);
    beginRoomDraft(document, history.source, 0);
    expect(history.future).toHaveLength(150);
    history = executeRoomChange(history, commitRoomDraft(history.source, setDraftCorners(beginRoomDraft(document, history.source, 0), outline)));
    expect(history.future).toHaveLength(0);
  });
});
