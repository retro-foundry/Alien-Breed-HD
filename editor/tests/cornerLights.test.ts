import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { setCornerLight, clearCornerLights, restoreCornerLights, encodeCornerWord, decodeCornerWord } from '../src/model/cornerLights';
import { switchRoomFixture } from './sourceSwitchFixture';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { encodeProject, decodeProject } from '../src/model/project';
import { roomSourceSnapshot, restoreRoomSource, moveSourceCorner, translateSourceRooms, removeSourceCorner } from '../src/model/roomSource';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from './roomHeightFixture';
import { BinaryReader } from '../src/model/binary';
import { createRoomOpening } from '../src/model/roomOpening';
import { setUpperStorey, setSkyCeiling } from '../src/model/roomStoreys';
import { roomPlaneRecords } from '../src/model/roomPlanes';

it('ports native point animation words without blending flag bytes into brightness', () => {
  for (let animation = 0; animation <= 15; animation++) for (let strength = 1; strength <= 16; strength++) for (const brightness of [0, 23, 127]) {
    const word = encodeCornerWord(brightness, animation, strength), decoded = decodeCornerWord(word);
    expect(decoded.brightness).toBe(brightness); expect(decoded.animation).toBe(animation);
    if (animation) expect(decoded.strength).toBe(strength);
  }
  expect(encodeCornerWord(-20, 0, 1)).toBe(-20);
  expect(decodeCornerWord(0x7102)).toEqual({ brightness: 2, animation: 1, strength: 8 });
  expect(() => encodeCornerWord(-1, 1, 1)).toThrow(/nonnegative/);
  expect(() => encodeCornerWord(128, 0, 1)).toThrow(); expect(() => encodeCornerWord(0, 16, 1)).toThrow(); expect(() => encodeCornerWord(0, 1, 17)).toThrow();
});

it('saves explicit corner words in v20 and retains them through move, copy, deletion and Undo', () => {
  const { document, room } = switchRoomFixture(), point = room.vertices[0]!;
  const history = execute({ document, past: [], future: [] }, editRoomSource(setCornerLight(document, room.id, 0, null, { lowerBrightness: -12, upperBrightness: 0x7102 })));
  expect(roomSourceSnapshot(history.document.roomSource).version).toBe(20);
  expect(() => restoreRoomSource(document, { ...roomSourceSnapshot(history.document.roomSource), version: 19 })).toThrow(/unsupported source fields/);
  const saved = decodeProject(encodeProject(history.document)); expect(saved.roomSource).toEqual(history.document.roomSource); expect(saved.files).toEqual(document.files);
  expect(undo(history).document.roomSource).toBe(document.roomSource); expect(redo(undo(history)).document.roomSource).toBe(history.document.roomSource);
  const moved = applyCommand(history.document, editRoomSource(moveSourceCorner(history.document.roomSource, room.id, 0, { x: point.x + 16, z: point.z })));
  const shifted = applyCommand(moved, editRoomSource(translateSourceRooms(moved.roomSource, [room.id], 512, 0)));
  expect(shifted.roomSource.rooms.at(-1)!.cornerLights![0]).toMatchObject({ x: point.x + 528, z: point.z, lowerBrightness: -12, upperBrightness: 0x7102 });
  expect(() => buildPlayableLevel(shifted, -1)).not.toThrow();
  const pasted = pasteRooms(document, copyRooms(history.document, [room.id]), 1024, 0), copied = applyCommand(document, pasted.command);
  expect(copied.roomSource.rooms.at(-1)!.cornerLights![0]).toMatchObject({ x: point.x + 1024, z: point.z, lowerBrightness: -12, upperBrightness: 0x7102 });
  const removed = applyCommand(history.document, editRoomSource(removeSourceCorner(history.document.roomSource, room.id, 0)));
  expect(removed.roomSource.rooms.at(-1)!.cornerLights).toBeUndefined();
});

it('allocates independent native point identities for all four normal floor/ceiling channels at one corner', () => {
  const fixture = nativeRoomCopyFixture(twoStoreyHeightTemplate()), room = fixture.copy, point = room.vertices[0]!;
  let document = fixture.document;
  const zone = document.zones[room.recordZone]!, planes = [...zone.lower.filter(item => [1, 2].includes(item.type)), ...zone.upper.filter(item => [1, 2].includes(item.type))];
  expect(planes).toHaveLength(4);
  for (let index = 0; index < planes.length; index++) document = applyCommand(document, editRoomSource(setCornerLight(document, room.id, 0, planes[index]!.offset, { lowerBrightness: -10 + index * 4, upperBrightness: 3 + index * 3 })));
  const built = buildPlayableLevel(document, -1), reader = new BinaryReader(built.document.files.data, 'compiled corner lighting');
  const cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!, ids = new Set<number>();
  for (let index = 0; index < planes.length; index++) {
    const identity = built.surfaceSources!.find(item => item.room === cell.id && item.sourceOffset === planes[index]!.offset)!;
    const record = (identity.upper ? built.document.zones[cell.id]!.upper : built.document.zones[cell.id]!.lower).find(item => item.offset === identity.offset)!;
    const id = record.points.find(id => built.document.points[id]!.x === point.x && built.document.points[id]!.z === point.z)!;
    ids.add(id); expect(reader.i16(built.document.pointBrightnessOffset + id * 4)).toBe(-10 + index * 4); expect(reader.i16(built.document.pointBrightnessOffset + id * 4 + 2)).toBe(3 + index * 3);
  }
  expect(ids.size).toBe(4);
});

it('preserves corner assignments on surviving opening corners and exposes strict repairable diagnostics', () => {
  const { document, room } = switchRoomFixture();
  const source = applyCommand(document, editRoomSource(setCornerLight(document, room.id, 0, null, { lowerBrightness: 7, upperBrightness: 0 })));
  const proposal = createRoomOpening(source, room.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 }, source.roomAppearances![0]!, 'sky'), opened = applyCommand(source, proposal.command);
  expect(opened.roomSource.rooms.some(room => room.cornerLights?.length)).toBe(true);
  expect(() => buildPlayableLevel(decodeProject(encodeProject(opened)), -1)).not.toThrow();
  expect(() => restoreCornerLights([{ x: 0, z: 0, lowerBrightness: 1, upperBrightness: 2 }], room.vertices)).toThrow(/existing room corner/);
  expect(() => restoreCornerLights([{ ...room.vertices[0], lowerBrightness: 1, upperBrightness: 2, extra: 0 }])).toThrow();
  const reset = applyCommand(source, editRoomSource(clearCornerLights(source, room.id))); expect(reset.roomSource.rooms.at(-1)!.cornerLights).toBeUndefined();
});

it('removes upper-only face lighting with sky/storey removal and retains scoped lighting only on its actual opening parts', () => {
  const fixture = nativeRoomCopyFixture(twoStoreyHeightTemplate()), room = fixture.copy;
  let source = fixture.document;
  const template = source.zones[room.recordZone]!, floor = template.upper.find(item => item.type === 1)!, ceiling = template.upper.find(item => item.type === 2)!;
  source = applyCommand(source, editRoomSource(setCornerLight(source, room.id, 0, floor.offset, { lowerBrightness: 0, upperBrightness: 9 })));
  source = applyCommand(source, editRoomSource(setCornerLight(source, room.id, 0, ceiling.offset, { lowerBrightness: 0, upperBrightness: 17 })));
  source = applyCommand(source, editRoomSource(setSkyCeiling(source, room.id, true, true)));
  expect(source.roomSource.rooms.at(-1)!.cornerLights).toHaveLength(1);
  source = applyCommand(source, editRoomSource(setUpperStorey(source, room.id, null)));
  expect(source.roomSource.rooms.at(-1)!.cornerLights).toBeUndefined();
  expect(() => buildPlayableLevel(decodeProject(encodeProject(source)), -1)).not.toThrow();

  const fresh = switchRoomFixture(), plane = roomPlaneRecords(fresh.document, fresh.room, false).find(item => item.type === 1)!;
  const lit = applyCommand(fresh.document, editRoomSource(setCornerLight(fresh.document, fresh.room.id, 0, plane.offset, { lowerBrightness: 21, upperBrightness: 0 })));
  const proposal = createRoomOpening(lit, fresh.room.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 }, lit.roomAppearances![0]!, 'sky'), opened = applyCommand(lit, proposal.command);
  const parts = opened.roomSource.rooms.filter(room => proposal.rooms.includes(room.id));
  expect(parts.filter(room => room.cornerLights?.length)).toHaveLength(1);
  expect(() => buildPlayableLevel(decodeProject(encodeProject(opened)), -1)).not.toThrow();
});
