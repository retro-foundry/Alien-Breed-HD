import { expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { clearRoomPolygonMaterials, polygonMaterialTarget, restorePolygonMaterials, setPolygonMaterial } from '../src/model/polygonMaterials';
import { floorPatternOffsets, floorRepeatScales, floorTextureRgba, readFloorTextures } from '../src/model/floorTextures';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { decodeProject, encodeProject } from '../src/model/project';
import { moveSourceCorner, roomSourceSnapshot, restoreRoomSource, translateSourceRooms } from '../src/model/roomSource';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';

it.each([false, true].flatMap(upper => [1, 2].map(type => ({ upper, type }))))('paints one source polygon with native provenance ($upper, type $type)', ({ upper, type }) => {
  const { document, copy, original } = nativeRoomCopyFixture(upper ? twoStoreyHeightTemplate() : 0);
  const record = (upper ? document.zones[copy.recordZone]!.upper : document.zones[copy.recordZone]!.lower).find(item => item.type === type)!;
  const target = { room: copy.id, record: record.offset, upper }, baseline = buildPlayableLevel(document, -1);
  const cell = baseline.topology!.cells.find(cell => cell.sourceId === copy.id)!;
  const identity = baseline.surfaceSources!.find(item => item.room === cell.id && item.upper === upper && item.sourceOffset === record.offset)!;
  expect(polygonMaterialTarget(baseline, cell.id, { offset: identity.offset, upper, type })).toEqual(target);
  const material = { texture: floorPatternOffsets.find(value => value !== record.texture)!, scale: -5 };
  const command = editRoomSource(setPolygonMaterial(document, target, material)), painted = applyCommand(document, command), built = buildPlayableLevel(painted, -1);
  for (const item of built.surfaceSources!) {
    const zone = built.document.zones[item.room]!, stream = item.upper ? zone.upper : zone.lower;
    const generated = stream.find(record => record.offset === item.offset)!;
    const prior = baseline.surfaceSources!.find(prior => prior.room === item.room && prior.upper === item.upper && prior.index === item.index)!;
    const priorZone = baseline.document.zones[prior.room]!, before = (item.upper ? priorZone.upper : priorZone.lower).find(record => record.offset === prior.offset)!;
    expect(generated).toEqual(item.room === cell.id && item.upper === upper && item.sourceOffset === record.offset
      ? { ...before, texture: material.texture, polygon: { ...before.polygon!, ...material } } : before);
  }
  expect(painted.roomSource.rooms.find(room => room.id === original.id)!.polygonMaterials).toBeUndefined();
  expect(painted.files).toEqual(document.files);
  expect(applyCommand(painted, command, true)).toEqual(document);
  expect(setPolygonMaterial(painted, target, material).after).toBe(painted.roomSource);
});

it('saves independent authored surface materials through v9, corner/room moves and clipboard copies', () => {
  const { document, to: room } = roomHeightFixture(), record = document.zones[room.recordZone]!.lower.find(record => record.type === 1)!;
  const target = { room: room.id, record: record.offset, upper: false };
  const painted = applyCommand(document, editRoomSource(setPolygonMaterial(document, target, { texture: 771, scale: 2 })));
  const saved = decodeProject(encodeProject(painted)), snapshot = roomSourceSnapshot(saved.roomSource);
  expect(snapshot.version).toBe(9); expect(saved.roomSource).toEqual(painted.roomSource); expect(saved.files).toEqual(document.files);
  expect(() => restoreRoomSource(document, { ...snapshot, version: 8 })).toThrow(/unsupported source fields/);
  const corner = room.vertices[0]!;
  const moved = applyCommand(saved, editRoomSource(moveSourceCorner(saved.roomSource, room.id, 0, { x: corner.x - 8, z: corner.z })));
  const translated = applyCommand(moved, editRoomSource(translateSourceRooms(moved.roomSource, [room.id], -512, 0)));
  expect(translated.roomSource.rooms.find(item => item.id === room.id)!.polygonMaterials).toEqual(saved.roomSource.rooms.find(item => item.id === room.id)!.polygonMaterials);
  expect(() => buildPlayableLevel(translated, -1)).not.toThrow();
  const pasted = applyCommand(document, pasteRooms(document, copyRooms(painted, [room.id]), -8000, 0).command);
  expect(roomSourceSnapshot(pasted.roomSource).version).toBe(9);
  expect(pasted.roomSource.rooms.at(-1)!.polygonMaterials).toEqual(saved.roomSource.rooms.find(item => item.id === room.id)!.polygonMaterials);
  expect(() => buildPlayableLevel(pasted, -1)).not.toThrow();
  const reset = applyCommand(saved, editRoomSource(clearRoomPolygonMaterials(saved, room.id)));
  expect(buildPlayableLevel(reset, -1).document.files).toEqual(buildPlayableLevel(document, -1).document.files);
});

it('rejects duplicate, out-of-bank or malformed paints before native compilation and allows resetting orphaned source', () => {
  const { document, to: room } = roomHeightFixture();
  const record = document.zones[room.recordZone]!.lower.find(item => item.type === 1)!, target = { room: room.id, record: record.offset, upper: false };
  const material = { ...target, texture: 0, scale: 0 }, { room: _id, ...stored } = material;
  expect(() => restorePolygonMaterials([stored, stored])).toThrow(/duplicate/);
  for (const value of [-1, 772, 65535, NaN]) expect(() => restorePolygonMaterials([{ ...stored, texture: value }])).toThrow();
  expect(() => restorePolygonMaterials([{ ...stored, upper: 0 }])).toThrow();
  expect(() => restorePolygonMaterials([{ ...stored, extra: 1 }])).toThrow();
  const painted = applyCommand(document, editRoomSource(setPolygonMaterial(document, target, { texture: 1, scale: 0 })));
  const bad = { ...painted, roomSource: { ...painted.roomSource, rooms: painted.roomSource.rooms.map(item => item.id === room.id
    ? { ...item, polygonMaterials: [{ ...stored, texture: 65535 }] } : item) } };
  expect(() => buildPlayableLevel(bad, -1)).toThrow(/tile offset/);
  const orphaned = { ...painted, roomSource: { ...painted.roomSource, rooms: painted.roomSource.rooms.map(item => item.id === room.id
    ? { ...item, polygonMaterials: [{ ...stored, record: 123456 }] } : item) } };
  expect(() => buildPlayableLevel(orphaned, -1)).toThrow(/no longer draws painted surface/);
  expect(() => encodeProject(orphaned)).not.toThrow();
  const repaired = applyCommand(orphaned, editRoomSource(clearRoomPolygonMaterials(orphaned, room.id)));
  expect(() => buildPlayableLevel(repaired, -1)).not.toThrow();
});

it('decodes all 16 AMOS patterns and 15 native palette levels without inventing a colour ramp', () => {
  const pixels = new Uint8Array(readFileSync(new URL('../../amiga/data/includes/floortile', import.meta.url)));
  const source = readFileSync(new URL('../../amiga/data/pal/FloorPalScaled.s', import.meta.url), 'utf8');
  const palette = Uint8Array.from([...source.matchAll(/\$([0-9a-f]{2})/gi)].map(match => Number.parseInt(match[1]!, 16)));
  const assets = readFloorTextures(pixels, palette);
  expect(floorPatternOffsets).toEqual([0, 1, 2, 3, 256, 257, 258, 259, 512, 513, 514, 515, 768, 769, 770, 771]);
  expect(floorRepeatScales).toEqual([-5, -4, -3, -2, -1, 0, 1, 2]);
  for (const tile of floorPatternOffsets) for (const shade of [0, 7, 14]) {
    const image = floorTextureRgba(assets, tile, shade);
    for (const [x, y] of [[0, 0], [11, 23], [63, 63]] as const) {
      const texel = pixels[tile + ((x * 4) | (y * 1024))]!, at = shade * 512 + texel * 2;
      const rgb = palette[at]! * 256 + palette[at + 1]!;
      expect([...image.slice((y * 64 + x) * 4, (y * 64 + x) * 4 + 4)]).toEqual([((rgb >> 8) & 15) * 17, ((rgb >> 4) & 15) * 17, (rgb & 15) * 17, 255]);
    }
  }
  expect(() => floorTextureRgba(assets, 772, 0)).toThrow(); expect(() => floorTextureRgba(assets, 0, 15)).toThrow();
  expect(() => readFloorTextures(pixels.slice(1), palette)).toThrow(/wrong size/);
});
