import { expect, it } from 'vitest';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { clearRoomWallMaterials, restoreWallMaterials, setWallMaterial, validateWallMaterialSampling, wallMaterialChoices, wallMaterialTarget, type WallMaterialTarget } from '../src/model/wallMaterials';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from './roomHeightFixture';
import { decodeProject, encodeProject } from '../src/model/project';
import { roomSourceSnapshot, restoreRoomSource, moveSourceCorner, translateSourceRooms } from '../src/model/roomSource';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { setRoomControllerCopies } from '../src/model/movingControllerCopies';

function fixture(upper = false) {
  const { document, copy: copied } = nativeRoomCopyFixture(upper ? twoStoreyHeightTemplate() : 0);
  const room = document.roomSource.rooms.find(room => room.id === copied.id)!;
  const record = (upper ? document.zones[room.recordZone]!.upper : document.zones[room.recordZone]!.lower).find(record => record.wall && record.texture !== 11)!;
  const style = wallMaterialChoices(document).find(item => item.texture !== record.texture)!;
  const target: WallMaterialTarget = { room: room.id, upper, record: record.offset };
  return { document, room, record, style, target };
}

it('rejects out-of-sheet phases and unsafe shifts before compiling a painted preview', () => {
  const { document, room, record, style, target } = fixture();
  const material = { record: target.record, upper: false, texture: 0, verticalMask: 63, verticalShift: 6, horizontalMask: 255 };
  expect(() => validateWallMaterialSampling(record.wall!, material, 0)).not.toThrow();
  expect(() => validateWallMaterialSampling(record.wall!, material, 100)).toThrow(/retained texture alignment/);
  expect(() => validateWallMaterialSampling(record.wall!, material, -1)).toThrow(/retained texture alignment/);
  expect(() => validateWallMaterialSampling(record.wall!, { ...material, verticalShift: 255 }, 0)).toThrow();
  expect(() => validateWallMaterialSampling(record.wall!, { ...material, verticalShift: 0, verticalMask: 0 }, 0)).not.toThrow();
  const painted = applyCommand(document, editRoomSource(setWallMaterial(document, target, style.offset)));
  const corrupted = { ...painted, roomSource: { ...painted.roomSource, rooms: painted.roomSource.rooms.map(item => item.id === room.id
    ? { ...item, wallMaterials: item.wallMaterials!.map(item => ({ ...item, verticalShift: 255, verticalMask: 255 })) } : item) } };
  expect(() => buildPlayableLevel(corrupted, -1)).toThrow(/retained texture alignment/);
});

it.each([false, true])('resolves copied wall provenance and preserves non-material fields (upper=%s)', upper => {
  const { document, room, record, style, target } = fixture(upper), baseline = buildPlayableLevel(document, -1);
  const cell = baseline.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const identity = baseline.surfaceSources!.find(item => item.room === cell.id && item.upper === upper && item.sourceOffset === record.offset)!;
  expect(identity.offset).not.toBe(record.offset);
  expect(wallMaterialTarget(baseline, cell.id, { offset: identity.offset, upper, type: 0 })).toEqual(target);
  const painted = applyCommand(document, editRoomSource(setWallMaterial(document, target, style.offset))), built = buildPlayableLevel(painted, -1);
  const read = (artifact: typeof built, id: number) => (upper ? artifact.document.zones[cell.id]!.upper : artifact.document.zones[cell.id]!.lower).find(item => item.offset === id)!.wall!;
  const before = read(baseline, identity.offset), after = read(built, built.surfaceSources!.find(item => item.room === cell.id && item.upper === upper && item.sourceOffset === record.offset)!.offset);
  const { texture: _a, verticalMask: _b, verticalShift: _c, horizontalMask: _d, ...geometry } = before;
  const { texture: _e, verticalMask: _f, verticalShift: _g, horizontalMask: _h, ...retained } = after;
  expect(retained).toEqual(geometry); expect(after.texture).toBe(style.texture);
  expect(painted.files).toEqual(document.files);
  const original = built.topology!.cells.find(cell => cell.originZone === room.recordZone)!;
  const originalWall = built.surfaceSources!.find(item => item.room === original.id && item.upper === upper && item.sourceOffset === record.offset)!;
  expect((upper ? built.document.zones[original.id]!.upper : built.document.zones[original.id]!.lower).find(item => item.offset === originalWall.offset)!.texture).toBe(record.texture);
  expect(applyCommand(painted, editRoomSource(setWallMaterial(painted, target, style.offset))).roomSource).toBe(painted.roomSource);
});

it('saves v7 paint without flattening source, survives moves and clipboard snapshots, and resets with Undo', () => {
  const { document, room, style, target } = fixture();
  const command = editRoomSource(setWallMaterial(document, target, style.offset)), painted = applyCommand(document, command);
  const snapshot = roomSourceSnapshot(painted.roomSource); expect(snapshot.version).toBe(7);
  expect(snapshot.controllerCopies).toEqual([]); expect(snapshot.switchCopies).toEqual([]);
  const saved = decodeProject(encodeProject(painted)); expect(saved.roomSource).toEqual(painted.roomSource); expect(saved.files).toEqual(document.files);
  const moved = applyCommand(saved, editRoomSource(translateSourceRooms(saved.roomSource, [room.id], 400, 0)));
  expect(moved.roomSource.rooms.find(item => item.id === room.id)!.wallMaterials).toEqual(room.wallMaterials ?? painted.roomSource.rooms.find(item => item.id === room.id)!.wallMaterials);
  expect(buildPlayableLevel(moved, -1).surfaceSources).toBeDefined();
  const pasted = pasteRooms(document, copyRooms(painted, [room.id]), -8000, 0), copy = applyCommand(document, pasted.command);
  expect(copy.roomSource.rooms.find(item => item.id === pasted.ids[0])!.wallMaterials).toEqual(painted.roomSource.rooms.find(item => item.id === room.id)!.wallMaterials);
  const reset = editRoomSource(clearRoomWallMaterials(painted, room.id)), clean = applyCommand(painted, reset);
  expect(clean.roomSource.rooms.find(item => item.id === room.id)!.wallMaterials).toBeUndefined();
  expect(applyCommand(clean, reset, true).roomSource).toBe(painted.roomSource);
  expect(applyCommand(painted, command, true).roomSource).toBe(document.roomSource);
});

it.each(['door', 'lift'] as const)('retains independent %s settings and native graphics attachments through paint/save', kind => {
  const input = movingControllerCopyFixture(kind);
  const document = applyCommand(input.document, editRoomSource(setRoomControllerCopies(input.document, [input.copy])));
  const room = input.added.find(room => document.zones[room.recordZone]!.lower.some(record => record.wall && record.texture !== 11))!;
  const record = document.zones[room.recordZone]!.lower.find(record => record.wall && record.texture !== 11)!;
  const style = wallMaterialChoices(document).find(style => style.texture !== record.texture)!;
  const painted = applyCommand(document, editRoomSource(setWallMaterial(document, { room: room.id, upper: false, record: record.offset }, style.offset)));
  const reopened = decodeProject(encodeProject(painted)), built = buildPlayableLevel(reopened, -1);
  expect(reopened.roomSource.controllerCopies).toEqual(document.roomSource.controllerCopies);
  expect(built.document.mechanisms.filter(item => item.kind === kind)).toHaveLength(2);
  expect(built.document.mechanisms.filter(item => item.kind === kind).every(item => item.walls.length > 0)).toBe(true);
});

it('keeps paints on reshaped source and rejects malformed snapshots, missing faces and switch graphics', () => {
  const { document, room, style, target } = fixture();
  const painted = applyCommand(document, editRoomSource(setWallMaterial(document, target, style.offset)));
  const corner = room.vertices[0]!;
  const changed = moveSourceCorner(painted.roomSource, room.id, 0, { x: corner.x - 4, z: corner.z });
  expect(changed.after.rooms.find(item => item.id === room.id)!.wallMaterials).toEqual(painted.roomSource.rooms.find(item => item.id === room.id)!.wallMaterials);
  const snapshot = roomSourceSnapshot(painted.roomSource);
  expect(() => restoreRoomSource(document, { ...snapshot, version: 6 })).toThrow(/unsupported source fields/);
  const { switchCopies: _missing, ...incomplete } = snapshot;
  expect(() => restoreRoomSource(document, incomplete)).toThrow(/missing its copied switch/);
  expect(() => restoreWallMaterials([{ record: 1, upper: false, texture: 999, verticalMask: 1, verticalShift: 1, horizontalMask: 1 }])).toThrow();
  const material = painted.roomSource.rooms.find(item => item.id === room.id)!.wallMaterials![0]!;
  expect(() => restoreWallMaterials([material, material])).toThrow(/duplicate/);
  const control = document.zones.flatMap(zone => zone.lower.filter(record => record.texture === 11).map(record => ({ zone, record })))[0]!;
  expect(() => setWallMaterial(document, { room: document.roomSource.rooms.find(room => room.originZone === control.zone.id)!.id, upper: false, record: control.record.offset }, style.offset)).toThrow(/switch/);
  const orphaned = { ...painted, roomSource: { ...painted.roomSource, rooms: painted.roomSource.rooms.map(item => item.id === room.id
    ? { ...item, wallSources: item.vertices.map(() => null) } : item) } };
  expect(() => buildPlayableLevel(orphaned, -1)).toThrow(/no longer draws painted wall/);
  const repaired = applyCommand(orphaned, editRoomSource(clearRoomWallMaterials(orphaned, room.id)));
  expect(() => buildPlayableLevel(repaired, -1)).not.toThrow();
});
