import { expect, it } from 'vitest';
import { applyCommand, editRoomAppearance, editRoomSource } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { beginRoomDraft, commitRoomDraft, moveSourceCorner, restoreRoomSource, roomSourceSnapshot, setDraftCorners, translateSourceRooms } from '../src/model/roomSource';
import { clearRoomWallMaterials, hasSurfaceMaterialOverride, setSurfaceMaterial, surfaceMaterialCurrent, surfaceMaterialTarget, wallMaterialChoices } from '../src/model/wallMaterials';
import { restoreBoundaryMaterials } from '../src/model/boundaryMaterials';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { sourceBoundaryIndex } from '../src/model/boundaryMaterials';

function targetFor(document: ReturnType<typeof roomHeightFixture>['document'], room: string, upper = false) {
  const built = buildPlayableLevel(document, -1);
  const identity = built.surfaceSources!.find(item => built.topology!.cells.find(cell => cell.id === item.room)?.sourceId === room
    && item.upper === upper && item.boundaryIndex !== undefined)!;
  const record = (upper ? built.document.zones[identity.room]!.upper : built.document.zones[identity.room]!.lower).find(item => item.offset === identity.offset)!;
  const target = surfaceMaterialTarget(built, identity.room, { offset: identity.offset, upper, type: record.type })!;
  if (!('index' in target)) throw new Error('Expected an authored boundary.');
  const style = wallMaterialChoices(document).find(item => item.texture !== record.texture)!;
  return { built, identity, record, target, style };
}

it.each([false, true])('paints only one authored boundary and keeps geometry/sampling phase (upper=%s)', upper => {
  const { document, from } = roomHeightFixture(upper ? twoStoreyHeightTemplate() : 0);
  const { built, identity, record, target, style } = targetFor(document, from.id, upper);
  const command = editRoomSource(setSurfaceMaterial(document, target, style.offset)), painted = applyCommand(document, command);
  const result = buildPlayableLevel(painted, -1);
  const before = upper ? built.document.zones[identity.room]!.upper : built.document.zones[identity.room]!.lower;
  const after = upper ? result.document.zones[identity.room]!.upper : result.document.zones[identity.room]!.lower;
  for (let i = 0; i < before.length; i++) {
    if (before[i]!.offset !== record.offset) expect(after[i]).toEqual(before[i]);
    else expect(after[i]!.wall).toEqual({ ...record.wall!, texture: style.wall!.texture, verticalMask: style.wall!.verticalMask,
      verticalShift: style.wall!.verticalShift, horizontalMask: style.wall!.horizontalMask });
  }
  expect(hasSurfaceMaterialOverride(painted, target)).toBe(true);
  expect(surfaceMaterialCurrent(painted, target)!.texture).toBe(style.texture);
  expect(applyCommand(painted, command, true)).toEqual(document);
  expect(setSurfaceMaterial(painted, target, style.offset).after).toBe(painted.roomSource);
  const another = applyCommand(painted, editRoomSource(setSurfaceMaterial(painted, { ...target, index: (target.index + 1) % from.vertices.length }, style.offset)));
  expect(setSurfaceMaterial(another, target, style.offset).after).toBe(another.roomSource);
  expect(painted.files).toEqual(document.files);
});

it('persists identities through source saves, reversed outlines, corner moves, translation and independent clipboard copies', () => {
  const { document, to: from } = roomHeightFixture();
  const { target, style } = targetFor(document, from.id);
  const painted = applyCommand(document, editRoomSource(setSurfaceMaterial(document, target, style.offset)));
  const saved = decodeProject(encodeProject(painted)), snapshot = roomSourceSnapshot(saved.roomSource);
  expect(snapshot.version).toBe(8); expect(saved.roomSource).toEqual(painted.roomSource);
  expect(() => restoreRoomSource(document, { ...snapshot, version: 7 })).toThrow(/unsupported source fields/);
  const room = saved.roomSource.rooms.find(item => item.id === from.id)!;
  const reversed = commitRoomDraft(saved.roomSource, setDraftCorners(beginRoomDraft(saved, saved.roomSource, room.recordZone, room.id), [...room.vertices].reverse()));
  expect(reversed.after).toBe(saved.roomSource);
  const moved = applyCommand(saved, editRoomSource(moveSourceCorner(saved.roomSource, room.id, 0, { x: room.vertices[0]!.x - 16, z: room.vertices[0]!.z })));
  expect(moved.roomSource.rooms.find(item => item.id === room.id)!.boundaryMaterials).toEqual(room.boundaryMaterials);
  expect(() => buildPlayableLevel(moved, -1)).not.toThrow();
  const translated = applyCommand(moved, editRoomSource(translateSourceRooms(moved.roomSource, [room.id], -512, 0)));
  expect(translated.roomSource.rooms.find(item => item.id === room.id)!.boundaryMaterials).toEqual(room.boundaryMaterials);
  expect(() => buildPlayableLevel(translated, -1)).not.toThrow();
  const paste = pasteRooms(document, copyRooms(painted, [from.id]), -8000, 0), copied = applyCommand(document, paste.command);
  expect(copied.roomSource.rooms.at(-1)!.boundaryMaterials).toEqual(room.boundaryMaterials);
  expect(roomSourceSnapshot(copied.roomSource).version).toBe(8);
  expect(() => buildPlayableLevel(copied, -1)).not.toThrow();
  const reset = applyCommand(painted, editRoomSource(setSurfaceMaterial(painted, target, null)));
  expect(hasSurfaceMaterialOverride(reset, target)).toBe(false);
  expect(buildPlayableLevel(reset, -1).document.files).toEqual(buildPlayableLevel(document, -1).document.files);
});

it('paints repaired doorway fills independently of retained native records', () => {
  const { document, copy } = nativeRoomCopyFixture();
  const { built, target, style } = targetFor(document, copy.id);
  const painted = applyCommand(document, editRoomSource(setSurfaceMaterial(document, target, style.offset)));
  const result = buildPlayableLevel(painted, -1);
  expect(result.document.files.graphics).not.toEqual(built.document.files.graphics);
  expect(painted.roomSource.rooms.find(item => item.id === copy.id)!.wallMaterials).toBeUndefined();
  expect(painted.roomSource.rooms.find(item => item.id === copy.id)!.boundaryMaterials!.overrides).toHaveLength(1);
});

it('retains a painted edge across a partial portal split and a corner move that rotates normalization', () => {
  const { document, from } = roomHeightFixture();
  const index = from.vertices.findIndex((a, i) => a.x === 20256 && from.vertices[(i + 1) % from.vertices.length]!.x === 20256);
  const style = wallMaterialChoices(document).find(item => item.texture !== surfaceMaterialCurrent(document, { room: from.id, index, upper: false })!.texture)!;
  const painted = applyCommand(document, editRoomSource(setSurfaceMaterial(document, { room: from.id, index, upper: false }, style.offset)));
  const appended = applyCommand(painted, editRoomSource(commitRoomDraft(painted.roomSource, setDraftCorners(beginRoomDraft(painted, painted.roomSource, 0),
    [{ x: 20256, z: 20064 }, { x: 20384, z: 20064 }, { x: 20384, z: 20192 }, { x: 20256, z: 20192 }]))));
  const adjacent = appended.roomSource.rooms.at(-1)!, choices = roomAppearanceChoices(appended, adjacent);
  const joined = applyCommand(appended, editRoomAppearance(appended, { room: adjacent.id, lowerWall: choices.lower[0]!.offset,
    upperWall: null, lightingPoint: choices.points[0]! }));
  const built = buildPlayableLevel(joined, -1);
  const pieces = built.surfaceSources!.filter(item => item.boundaryIndex === index && !item.upper && built.topology!.cells[item.room]!.sourceId === from.id);
  expect(pieces).toHaveLength(2);
  for (const piece of pieces) expect(built.document.zones[piece.room]!.lower.find(item => item.offset === piece.offset)!.texture).toBe(style.texture);
  const corner = from.vertices[0]!, point = { x: corner.x + 64, z: corner.z - 64 };
  const moved = applyCommand(joined, editRoomSource(moveSourceCorner(joined.roomSource, from.id, 0, point)));
  const room = moved.roomSource.rooms.find(item => item.id === from.id)!;
  expect(room.vertices[0]).not.toEqual(point);
  const before = painted.roomSource.rooms.find(item => item.id === from.id)!;
  const inherited = room.vertices.flatMap((start, i) => sourceBoundaryIndex(from, start, room.vertices[(i + 1) % room.vertices.length]!) === index ? [i] : []);
  expect(inherited).toHaveLength(3);
  for (const i of inherited) expect(room.boundaryMaterials!.edges[i]).toBe(before.boundaryMaterials!.edges[index]);
  expect(() => buildPlayableLevel(moved, -1)).not.toThrow();
});

it('keeps orphaned boundary paint saveable and repairable without silently attaching it to a new edge', () => {
  const { document, to: from } = roomHeightFixture();
  const { target, style } = targetFor(document, from.id);
  const painted = applyCommand(document, editRoomSource(setSurfaceMaterial(document, target, style.offset)));
  const room = painted.roomSource.rooms.find(item => item.id === from.id)!;
  const replaced = applyCommand(painted, editRoomSource(commitRoomDraft(painted.roomSource, setDraftCorners(
    beginRoomDraft(painted, painted.roomSource, room.recordZone, room.id), room.vertices.map(p => ({ x: p.x + 512, z: p.z }))))));
  const reopened = decodeProject(encodeProject(replaced));
  expect(() => buildPlayableLevel(reopened, -1)).toThrow(/no longer draws painted boundary/);
  const repaired = applyCommand(reopened, editRoomSource(clearRoomWallMaterials(reopened, room.id)));
  expect(() => buildPlayableLevel(repaired, -1)).not.toThrow();
  expect(() => restoreBoundaryMaterials({ nextId: 4, edges: [0, 1], overrides: [] }, 4)).toThrow();
  expect(() => restoreBoundaryMaterials({ nextId: 1, edges: [1], overrides: [] }, 1)).toThrow();
  const entry = room.boundaryMaterials!.overrides[0]!;
  expect(() => restoreBoundaryMaterials({ ...room.boundaryMaterials, overrides: [entry, entry] }, 4)).toThrow(/duplicate/);
  expect(() => restoreBoundaryMaterials({ ...room.boundaryMaterials, overrides: [{ ...entry, record: 1 }] }, 4)).toThrow(/native records/);
});
