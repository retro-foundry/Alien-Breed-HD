import { expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { openLevel } from '../src/model/level';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from './roomHeightFixture';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { setPolygonMaterial, polygonMaterialTarget, polygonMaterialMode, restorePolygonMaterials } from '../src/model/polygonMaterials';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { roomSourceSnapshot, restoreRoomSource } from '../src/model/roomSource';
import { polygonMaterialType, type PolygonMaterialMode } from '../src/model/surfaces';
import { readFloorTextures, floorTextureRgba, floorPatternOffsets } from '../src/model/floorTextures';

it.each([false, true].flatMap(upper => [1, 2].flatMap(type => (['normal', 'bump', 'smooth'] as const).map(mode => ({ upper, type, mode })))))('converts $upper type $type to $mode with exact source geometry and independent saved history', ({ upper, type, mode }) => {
  const { document, copy } = nativeRoomCopyFixture(upper ? twoStoreyHeightTemplate() : 0), record = (upper ? document.zones[copy.recordZone]!.upper : document.zones[copy.recordZone]!.lower).find(r => r.type === type)!;
  const target = { room: copy.id, record: record.offset, upper }, before = buildPlayableLevel(document, -1);
  const command = editRoomSource(setPolygonMaterial(document, target, { mode, texture: mode === 'smooth' ? 0 : 771, scale: 2 }));
  let history = execute({ document, past: [], future: [] }, command);
  const saved = decodeProject(encodeProject(history.document)), snapshot = roomSourceSnapshot(saved.roomSource);
  expect(snapshot.version).toBe(11); expect(polygonMaterialMode(saved, target)).toBe(mode);
  expect(() => restoreRoomSource(document, { ...snapshot, version: 10 })).toThrow('version 11');
  const built = buildPlayableLevel(saved, -1), id = built.topology!.cells.find(c => c.sourceId === copy.id)!.id;
  const identity = built.surfaceSources!.find(s => s.room === id && s.upper === upper && s.sourceOffset === record.offset)!;
  const result = (upper ? built.document.zones[id]!.upper : built.document.zones[id]!.lower).find(r => r.offset === identity.offset)!;
  const previous = before.surfaceSources!.find(s => s.room === id && s.upper === upper && s.sourceOffset === record.offset)!;
  const prior = (upper ? before.document.zones[id]!.upper : before.document.zones[id]!.lower).find(r => r.offset === previous.offset)!;
  expect(result.polygon).toEqual({ ...prior.polygon, type: polygonMaterialType(type, mode), texture: mode === 'smooth' ? 0 : 771, scale: 2 });
  expect(polygonMaterialTarget(built, id, { offset: result.offset, type: result.type, upper })).toEqual(target);
  history = undo(history); expect(history.document.roomSource).toBe(document.roomSource); history = redo(history); expect(history.past).toHaveLength(1);
  const reset = applyCommand(saved, editRoomSource(setPolygonMaterial(saved, target, null)));
  expect(buildPlayableLevel(reset, -1).document.files).toEqual(before.document.files);
});
it('imports and repaints exported native bank modes without an implicit type conversion', () => {
  const base = nativeRoomCopyFixture();
  for (const originalType of [1, 2]) for (const mode of ['normal', 'bump', 'smooth'] as const) {
    const record = base.document.zones[base.copy.recordZone]!.lower.find(r => r.type === originalType)!;
    const painted = applyCommand(base.document, editRoomSource(setPolygonMaterial(base.document, { room: base.copy.id, record: record.offset, upper: false }, { mode, texture: 0, scale: 0 })));
    const built = buildPlayableLevel(painted, -1), imported = openLevel('Exported banks', built.document.files);
    const id = built.topology!.cells.find(c => c.sourceId === base.copy.id)!.id;
    const surface = imported.zones[id]!.lower.find(r => r.type === polygonMaterialType(originalType, mode))!;
    const target = polygonMaterialTarget({ source: imported, document: imported }, id, { offset: surface.offset, type: surface.type, upper: false })!;
    expect(target).toBeDefined();
    const changed = applyCommand(imported, editRoomSource(setPolygonMaterial(imported, target, { texture: mode === 'smooth' ? 0 : 771, scale: 2 })));
    expect(() => buildPlayableLevel(changed, -1)).not.toThrow();
  }
});
it('decodes real bump and smooth banks with their own palettes using the native interleaving', () => {
  for (const [tile, palette] of [['BumpTile', 'BumpPalScaled'], ['SmoothBumpTile', 'SmoothBumpPalScaled']]) {
    const pixels = new Uint8Array(readFileSync(new URL(`../../amiga/data/gfx/${tile}`, import.meta.url))), colours = new Uint8Array(readFileSync(new URL(`../../amiga/data/pal/${palette}`, import.meta.url)));
    const assets = readFloorTextures(pixels, colours);
    for (const offset of tile === 'BumpTile' ? floorPatternOffsets : [0]) for (const shade of [0,7,14]) {
      const image = floorTextureRgba(assets, offset, shade);
      for (const [x,y] of [[0,0],[11,23],[63,63]]) {
        const texel = pixels[offset + y! * 1024 + x! * 4]!, at = shade * 512 + texel * 2, rgb = colours[at]! * 256 + colours[at+1]!;
        expect([...image.slice((y!*64+x!)*4,(y!*64+x!)*4+4)]).toEqual([((rgb>>8)&15)*17,((rgb>>4)&15)*17,(rgb&15)*17,255]);
      }
    }
  }
  expect(() => restorePolygonMaterials([{ record: 0, upper: false, texture: 0, scale: 0, mode: 'imaginary' as PolygonMaterialMode }])).toThrow('material mode');
});
