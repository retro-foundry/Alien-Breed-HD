import { expect, it } from 'vitest';
import { applyRoomLightPreset } from '../src/model/roomLightPresets';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from './roomHeightFixture';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { BinaryReader } from '../src/model/binary';
import { decodeProject, encodeProject } from '../src/model/project';
import { setCornerLight } from '../src/model/cornerLights';
import { switchRoomFixture } from './sourceSwitchFixture';
import { finishRoomDraft } from '../src/model/roomCreation';
import { beginRoomDraft, setDraftCorners, translateSourceRooms } from '../src/model/roomSource';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { setUpperStorey, storeyMaterials } from '../src/model/roomStoreys';
import { roomSettings } from '../src/model/roomHeights';

// Hand-transcribed PRESETS / UPPERPRESETS words, not expectations derived
// from the implementation's preset table.
const words = [[255, 1], [1, 255], [1, 241], [241, 1], [241, 15], [15, 241]];
for (const upper of [false, true]) it.each(words.map((pair, preset) => ({ pair, preset })))('applies original preset $preset atomically on ' + (upper ? 'upper' : 'lower') + ' normal planes', ({ pair, preset }) => {
  const fixture = nativeRoomCopyFixture(twoStoreyHeightTemplate()), room = fixture.copy;
  const template = fixture.document.zones[room.recordZone]!, floor = (upper ? template.upper : template.lower).find(record => record.type === 1)!;
  const original = applyCommand(fixture.document, editRoomSource(setCornerLight(fixture.document, room.id, 0, floor.offset, { lowerBrightness: 0x7102, upperBrightness: 0x2306 })));
  const history = execute({ document: original, past: [], future: [] }, editRoomSource(applyRoomLightPreset(original, room.id, upper, preset)));
  const saved = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(saved, -1), reader = new BinaryReader(built.document.files.data, 'preset points');
  const cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!, zone = built.document.zones[cell.id]!;
  for (const record of (upper ? zone.upper : zone.lower).filter(record => [1, 2].includes(record.type))) {
    for (const id of record.points) expect(reader.i16(built.document.pointBrightnessOffset + id * 4 + (upper ? 2 : 0))).toBe(pair[record.type - 1]);
  }
  const retained = saved.roomSource.rooms.find(item => item.id === room.id)!.cornerLights!.find(light => light.record === floor.offset && light.x === room.vertices[0]!.x && light.z === room.vertices[0]!.z)!;
  expect(upper ? retained.lowerBrightness : retained.upperBrightness).toBe(upper ? 0x7102 : 0x2306);
  expect(undo(history).document.roomSource).toBe(original.roomSource);
  expect(redo(undo(history)).document.roomSource).toBe(history.document.roomSource);
  expect(saved.files).toEqual(original.files);
});

it('rejects missing storeys and invalid preset indices without changing source', () => {
  const fixture = nativeRoomCopyFixture(), room = fixture.copy;
  expect(() => applyRoomLightPreset(fixture.document, room.id, true, 0)).toThrow(/upper storey/);
  expect(() => applyRoomLightPreset(fixture.document, room.id, false, 10)).toThrow();
  expect(() => applyRoomLightPreset(fixture.document, 'missing', false, 0)).toThrow(/Select a room/);
});

for (const upper of [false, true]) it.each([6, 7, 8, 9])('authors original glowing preset %i on ' + (upper ? 'upper' : 'lower') + ' with retained point selectors', preset => {
  const fixture = nativeRoomCopyFixture(twoStoreyHeightTemplate()), room = fixture.copy;
  const before = buildPlayableLevel(fixture.document, -1), cell = before.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const geometricIds = room.vertices.map(point => cell.vertices.find(id => before.document.points[id]!.x === point.x && before.document.points[id]!.z === point.z)!);
  const change = applyRoomLightPreset(fixture.document, room.id, upper, preset), document = applyCommand(fixture.document, editRoomSource(change));
  const saved = decodeProject(encodeProject(document));
  const built = buildPlayableLevel(saved, -1), reader = new BinaryReader(built.document.files.data, 'glowing preset');
  for (const face of (upper ? built.document.zones[cell.id]!.upper : built.document.zones[cell.id]!.lower).filter(face => [1, 2].includes(face.type))) {
    for (const id of face.points) {
      const point = built.document.points[id]!, corner = room.vertices.findIndex(vertex => vertex.x === point.x && vertex.z === point.z);
      const c = geometricIds[corner]!;
      // leveld.asc PRESETS keys 7,8,9,0, including the floor complement.
      const glow = (0xf000 + ((c % 5) + 1) * 256) << 16 >> 16;
      const complement = (0xf000 + (((1000 - c) % 5) + 1) * 256) << 16 >> 16;
      const expected = preset === 6 ? (face.type === 1 ? 241 : glow) : preset === 7 ? (face.type === 1 ? glow : 241) : preset === 8 ? glow : face.type === 1 ? complement : glow;
      expect(reader.i16(built.document.pointBrightnessOffset + id * 4 + (upper ? 2 : 0))).toBe(expected);
    }
  }
  const assignments = saved.roomSource.rooms.find(item => item.id === room.id)!.cornerLights!;
  const moved = translateSourceRooms(saved.roomSource, [room.id], 2000, 0);
  expect(moved.after.rooms.find(item => item.id === room.id)!.cornerLights!.map(({ x: _x, z: _z, ...words }) => words)).toEqual(assignments.map(({ x: _x, z: _z, ...words }) => words));
  const clipboard = copyRooms(saved, [room.id]), pasted = applyCommand(saved, pasteRooms(saved, clipboard, 3000, 0).command);
  expect(pasted.roomSource.rooms.at(-1)!.cornerLights!.map(({ x: _x, z: _z, ...words }) => words)).toEqual(assignments.map(({ x: _x, z: _z, ...words }) => words));
  expect(() => buildPlayableLevel(pasted, -1)).not.toThrow();
});

it('keeps lower and upper presets independent when both storeys use the same source patterns', () => {
  const fixture = switchRoomFixture(), room = fixture.room, materials = storeyMaterials(fixture.document, room), heights = roomSettings(fixture.document, room);
  const surfaces = fixture.document.roomAppearances![0]!.surfaces!;
  let document = applyCommand(fixture.document, editRoomSource(setUpperStorey(fixture.document, room.id, {
    floor: heights.ceiling, ceiling: heights.ceiling - 128, wall: materials.walls[0]!.offset, lightingPoint: materials.walls[0]!.points[0]!, floorMaterial: surfaces.lowerFloor, ceilingMaterial: surfaces.lowerCeiling,
  })));
  document = applyCommand(document, editRoomSource(applyRoomLightPreset(document, room.id, false, 0)));
  document = applyCommand(document, editRoomSource(applyRoomLightPreset(document, room.id, true, 5)));
  const built = buildPlayableLevel(document, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!, zone = built.document.zones[cell.id]!;
  const reader = new BinaryReader(built.document.files.data, 'shared pattern preset');
  for (const upper of [false, true]) for (const face of (upper ? zone.upper : zone.lower).filter(record => [1, 2].includes(record.type))) {
    for (const point of face.points) expect(reader.i16(built.document.pointBrightnessOffset + point * 4 + (upper ? 2 : 0))).toBe((upper ? [15, 241] : [255, 1])[face.type - 1]);
  }
});

it('lights all generated cells of a concave newly drawn room', () => {
  const fixture = switchRoomFixture(), room = fixture.room;
  const vertices = [[20000, 20000], [20256, 20000], [20256, 20128], [20128, 20128], [20128, 20256], [20000, 20256]].map(([x, z]) => ({ x: x!, z: z! }));
  let document = applyCommand(fixture.document, finishRoomDraft(fixture.document, setDraftCorners(beginRoomDraft(fixture.document, fixture.document.roomSource, room.recordZone, room.id), vertices)));
  document = applyCommand(document, editRoomSource(applyRoomLightPreset(document, room.id, false, 4)));
  const built = buildPlayableLevel(document, -1), reader = new BinaryReader(built.document.files.data, 'concave preset');
  const cells = built.topology!.cells.filter(cell => cell.sourceId === room.id);
  expect(cells.length).toBeGreaterThan(1);
  for (const cell of cells) for (const face of built.document.zones[cell.id]!.lower.filter(record => [1, 2].includes(record.type))) {
    for (const point of face.points) expect(reader.i16(built.document.pointBrightnessOffset + point * 4)).toBe(face.type === 1 ? 241 : 15);
  }
});
