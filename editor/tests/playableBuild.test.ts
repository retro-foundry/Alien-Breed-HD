import { expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { applyCommand, editRoomAppearance, editRoomSource, execute, redo, setExitZone, setSourceStart, undo } from '../src/model/commands';
import { openLevel } from '../src/model/level';
import { buildPlayableLevel, parseGameExitZones, preparePlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeGameLevel, encodeProject } from '../src/model/project';
import { restoreRoomAppearances, roomAppearanceBuildOptions, roomAppearanceChoices } from '../src/model/roomAppearance';
import { beginRoomDraft, commitRoomDraft, moveSourceCorner, removeSourceRooms, setDraftCorners } from '../src/model/roomSource';
import { originalFiles } from './helpers';

function fixture() {
  const original = openLevel('Playable source', originalFiles(1));
  const change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]));
  const document = applyCommand(original, editRoomSource(change)), room = change.after.rooms.at(-1)!;
  const choices = roomAppearanceChoices(document, room);
  return { original, document, room, appearance: { room: room.id, lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]! } };
}

it('keeps unchanged maps byte-identical and uses the packaged native exit configuration without a duplicate default table', () => {
  const original = openLevel('original', originalFiles(1));
  expect(buildPlayableLevel(original, -1).document).toBe(original);
  const gameData = readFileSync(new URL('../../src/game_data.c', import.meta.url), 'utf8');
  const values = /const\s+int16_t\s+end_zones\[16\]\s*=\s*\{([^}]+)\}/.exec(gameData)![1]!.trim().split(/\s*,\s*/).map(Number);
  expect(parseGameExitZones({ format: 'alien-breed-game-config', version: 1, exitZones: values })).toEqual(values);
  expect(() => parseGameExitZones({ format: 'alien-breed-game-config', version: 1, exitZones: [0] })).toThrow(/configuration is invalid/);
});

it('builds selected source appearance and starts for play/export without replacing editable source', () => {
  const { original, document, room, appearance } = fixture();
  let source = applyCommand(document, editRoomAppearance(document, appearance));
  source = applyCommand(source, setSourceStart(source, 1, room.id, 20064, 20064));
  source = decodeProject(encodeProject(source));
  const built = buildPlayableLevel(source, -1), start = built.document.starts[0]!;
  expect(built.topology!.cells[start.zone]!.sourceId).toBe(room.id);
  expect(start).toMatchObject({ x: 20064, z: 20064 });
  const template = original.zones[0]!, generated = built.document.zones.find(zone => zone.id === start.zone)!;
  for (const wall of generated.lower.filter(record => record.wall)) expect(wall).toMatchObject({ top: template.ceiling, bottom: template.floor });
  expect(built.document.roomAppearances).toBeUndefined();
  expect(encodeGameLevel(built.document).length).toBeGreaterThan(0);
  expect(source.files).toEqual(original.files); expect(source.roomAppearances).toEqual([appearance]);
  expect(built.source).toBe(source);
});

it('persists appearance independently, preserves it through outline edits and supports atomic stale-checked undo/redo', () => {
  const { document, room, appearance } = fixture(), command = editRoomAppearance(document, appearance);
  let history = execute({ document, past: [], future: [] }, command);
  expect(execute(history, editRoomAppearance(history.document, appearance))).toBe(history);
  expect(() => applyCommand(history.document, command)).toThrow(/appearance changed/);
  const changed = applyCommand(history.document, editRoomSource(moveSourceCorner(history.document.roomSource, room.id, 0, { x: 19992, z: 20000 })));
  expect(changed.roomAppearances).toBe(history.document.roomAppearances);
  const restored = decodeProject(encodeProject(changed)); expect(restored.roomAppearances).toEqual([appearance]);
  const removed = applyCommand(changed, editRoomSource(removeSourceRooms(changed.roomSource, [room.id])));
  expect(decodeProject(encodeProject(removed)).roomAppearances).toEqual([appearance]);
  expect(roomAppearanceBuildOptions(removed).newRoomGraphics.size).toBe(0);
  history = undo(history); expect(history.document.roomAppearances).toBeUndefined();
  history = redo(history); expect(history.document.roomAppearances).toEqual([appearance]);
});

it('rejects missing and stale material selections rather than inventing new wall or lighting data', () => {
  const { document, room, appearance } = fixture();
  expect(() => buildPlayableLevel(document, -1)).toThrow(/needs an appearance/);
  expect(() => editRoomAppearance(document, { ...appearance, lowerWall: 0 })).toThrow(/no longer matches/);
  expect(() => editRoomAppearance(document, { ...appearance, lightingPoint: 2047 })).toThrow(/no longer matches/);
  expect(() => editRoomAppearance(document, { ...appearance, room: document.roomSource.rooms[0]!.id })).not.toThrow();
  expect(() => editRoomAppearance(document, { ...appearance, room: 'missing' })).toThrow(/existing room/);
  expect(() => restoreRoomAppearances([appearance, appearance])).toThrow(/Duplicate/);
  expect(() => restoreRoomAppearances([{ ...appearance, unknown: true }])).toThrow(/Invalid/);
  const placed = applyCommand(document, editRoomAppearance(document, appearance)), entries = unzipSync(encodeProject(placed));
  const manifest = JSON.parse(strFromU8(entries['alien-breed-project.json']!)); expect(manifest.version).toBe(8);
  delete entries['room-appearances.json']; expect(() => decodeProject(zipSync(entries))).toThrow(/appearances are missing/);
  entries['room-appearances.json'] = strToU8('{'); expect(() => decodeProject(zipSync(entries))).toThrow(/not valid JSON/);
  entries['room-appearances.json'] = strToU8(JSON.stringify([appearance]));
  entries['alien-breed-project.json'] = strToU8(JSON.stringify({ ...manifest, version: 4 }));
  expect(() => decodeProject(zipSync(entries))).toThrow(/require project format version 8/);
  expect(room.id).toBe(appearance.room);
});

it('shares immutable build artifacts for the same complete source and rebuilds after an edit', async () => {
  const { document, appearance } = fixture();
  let source = applyCommand(document, editRoomAppearance(document, appearance));
  source = applyCommand(source, setExitZone(source, -1));
  const first = preparePlayableLevel(source), second = preparePlayableLevel(source);
  expect(first).toBe(second); const built = await first; expect(built.document.exitZone).toBe(-1);
  const next = applyCommand(source, setSourceStart(source, 1, appearance.room, 20064, 20064));
  expect(await preparePlayableLevel(next)).not.toBe(built);
  expect((await preparePlayableLevel(next)).document.starts[0]).toMatchObject({ x: 20064, z: 20064 });
});
