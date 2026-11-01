import { expect, it } from 'vitest';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { captureRoomProperties, restoreRoomProperties } from '../src/model/roomProperties';
import { setRoomProperties, roomSourceSnapshot, restoreRoomSource, translateSourceRooms, setRoomHeights } from '../src/model/roomSource';
import { applyCommand, editRoomSource, numberPatch, execute, undo, redo } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { roomSettings, roomHeightFields } from '../src/model/roomHeights';
import { decodeProject, encodeProject } from '../src/model/project';

it('captures native properties independently of later template edits through save/build and one undo', () => {
  const fixture = nativeRoomCopyFixture(); let document = fixture.document;
  const properties = { ...captureRoomProperties(roomSettings(document, fixture.copy)), brightness: 17, upperBrightness: -12,
    control: -3, upperControl: 7, floorSound: 2, upperFloorSound: 3, backdrop: true };
  document = applyCommand(document, editRoomSource(setRoomProperties(document.roomSource, fixture.copy.id, properties)));
  const original = document.zones[fixture.copy.recordZone]!;
  document = applyCommand(document, { label: 'Change template', patches: [numberPatch(document, 'data', original.offset + 22, 6, 2)] });
  const saved = decodeProject(encodeProject(document)), built = buildPlayableLevel(saved, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === fixture.copy.id)!;
  expect(built.document.zones[cell.id]).toMatchObject(properties);
  expect(built.document.zones[0]!.brightness).toBe(6);
  expect(roomSourceSnapshot(saved.roomSource).version).toBe(4);
  const inherit = editRoomSource(setRoomProperties(saved.roomSource, fixture.copy.id, null));
  let history = execute({ document: saved, past: [], future: [] }, inherit);
  expect(roomSettings(history.document, history.document.roomSource.rooms.find(room => room.id === fixture.copy.id)!).brightness).toBe(6);
  history = undo(history); expect(history.document.roomSource).toBe(saved.roomSource);
  history = redo(history); expect(history.document.roomSource.rooms.find(room => room.id === fixture.copy.id)!.properties).toBeUndefined();
});

it('retains captured properties across height edits, geometry movement and version-four source restoration', () => {
  const { document, copy } = nativeRoomCopyFixture();
  const properties = captureRoomProperties(roomSettings(document, copy));
  let source = setRoomProperties(document.roomSource, copy.id, properties).after;
  const settings = roomSettings(document, copy), heights = Object.fromEntries(roomHeightFields.map(field => [field, settings[field]]));
  source = setRoomHeights(document, source, copy.id, { ...heights, floor: settings.floor + 8 }).after;
  source = translateSourceRooms(source, [copy.id], 16, 0).after;
  const restored = restoreRoomSource(document, JSON.parse(JSON.stringify(roomSourceSnapshot(source))));
  expect(restored.rooms.find(room => room.id === copy.id)).toMatchObject({ properties, heights: { ...heights, floor: settings.floor + 8 } });
});

it('rejects incomplete or unrepresentable property snapshots before changing source', () => {
  const { document, copy } = nativeRoomCopyFixture(), properties = captureRoomProperties(roomSettings(document, copy));
  for (const invalid of [{ ...properties, brightness: 32768 }, { ...properties, control: 128 }, { ...properties, backdrop: 1 }, { ...properties, water: Infinity }])
    expect(() => setRoomProperties(document.roomSource, copy.id, invalid)).toThrow();
  expect(() => restoreRoomProperties({ brightness: 0 })).toThrow();
  expect(document.roomSource.rooms.find(room => room.id === copy.id)!.properties).toBeUndefined();
});
