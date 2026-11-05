import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, redo, setExitZone, setSourceExit, undo } from '../src/model/commands';
import { moveSourceCorner, removeSourceRooms, restoreRoomSource, roomSourceSnapshot, sourceMatchesLevel } from '../src/model/roomSource';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeGameLevel, encodeProject, openArchive } from '../src/model/project';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { roomHeightFixture } from './roomHeightFixture';
import { sourceExitFixture } from './sourceExitFixture';


it('saves a stable authored exit, compiles all concave cells as one exit room and exports the native setup', () => {
  const { document, from } = sourceExitFixture();
  const source = applyCommand(document, setSourceExit(document, from.id));
  const restored = decodeProject(encodeProject(source));
  expect(restored.roomSource.exitRoom).toBe(from.id);
  expect(roomSourceSnapshot(restored.roomSource).version).toBe(14);
  const built = buildPlayableLevel(restored, -1), members = built.topology!.cells.filter(cell => cell.sourceId === from.id);
  expect(members.length).toBeGreaterThan(1);
  for (const cell of members) expect(built.document.drawGroups![cell.id]).toBe(built.document.exitZone);
  const game = openArchive(encodeGameLevel(built.document), 'exit.zip')[0]!;
  expect(game.exitZone).toBe(built.document.exitZone);
  expect(game.drawGroups).toEqual(built.document.drawGroups);
});

it('keeps one exit through edits and copying, rejects deletion and validates damaged source', () => {
  const { document, from, to } = sourceExitFixture();
  let source = applyCommand(document, setSourceExit(document, from.id));
  source = applyCommand(source, editRoomSource(moveSourceCorner(source.roomSource, to.id, 0, { x: 20992, z: 20000 })));
  source = applyCommand(source, pasteRooms(source, copyRooms(source, [from.id]), -10000, 0).command);
  expect(source.roomSource.exitRoom).toBe(from.id);
  expect(() => removeSourceRooms(source.roomSource, [from.id])).toThrow(/another level exit/);
  expect(removeSourceRooms(source.roomSource, [to.id]).after.exitRoom).toBe(from.id);
  expect(() => setSourceExit(source, 'missing')).toThrow(/existing source room/);
  const snapshot = roomSourceSnapshot(source.roomSource);
  expect(() => restoreRoomSource(source, { ...snapshot, exitRoom: 'missing' })).toThrow(/existing source room/);
  expect(() => restoreRoomSource(source, { ...snapshot, exitRoom: 0 })).toThrow(/existing source room/);
  expect(() => restoreRoomSource(source, { ...snapshot, version: 13 })).toThrow(/unsupported fields/);
});

it('atomically changes between source, original, numeric and disabled exits with Undo and stale checks', () => {
  const { original } = roomHeightFixture();
  const initial = { document: original, past: [], future: [] };
  const command = setSourceExit(original, original.roomSource.rooms[0]!.id);
  let history = execute(initial, command);
  expect(sourceMatchesLevel(history.document)).toBe(false);
  expect(execute(history, setSourceExit(history.document, history.document.roomSource.exitRoom!))).toBe(history);
  expect(() => applyCommand(history.document, command)).toThrow(/room source changed/);
  history = undo(history); expect(history.document.roomSource.exitRoom).toBeUndefined();
  history = redo(history); expect(history.document.roomSource.exitRoom).toBe(command.roomSource!.after.exitRoom);
  for (const zone of [null, -1, 0]) {
    const changed = execute(history, setExitZone(history.document, zone));
    expect(changed.document.roomSource.exitRoom).toBeUndefined();
    expect(changed.document.exitZone).toBe(zone);
    expect(undo(changed).document.roomSource.exitRoom).toBe(history.document.roomSource.exitRoom);
  }
});
