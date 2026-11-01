import { expect, it } from 'vitest';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import fixtures from './fixtures/png-images.json';
import { pngDimensions } from '../src/model/png';
import { convertWallRgba, encodeWallIndices } from '../src/model/wallImage';
import { wallTextureRgba } from '../src/model/wallTextures';
import { applyCommand, execute, redo, replaceWallAsset, undo } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';
import { wallOverrideFixture } from './wallOverrideFixture';

const png = new Uint8Array(Buffer.from(fixtures.rgba7x5, 'base64'));
function paletteTexture() {
  const bytes = new Uint8Array(2064);
  for (let shade = 0; shade < 32; shade++) for (const [index, color] of [[1, 0xf00], [2, 0x0f0], [3, 0x00f]]) {
    bytes[shade * 64 + index! * 2] = color! >>> 8; bytes[shade * 64 + index! * 2 + 1] = color! & 255;
  }
  return { width: 3, height: 8, bytes };
}
it('validates independent PNG signature, bounded dimensions, chunk checksums and endings before browser decoding', () => {
  expect(pngDimensions(png)).toEqual({ width: 7, height: 5 });
  expect(pngDimensions(new Uint8Array(Buffer.from(fixtures.rgba4k, 'base64')))).toEqual({ width: 3840, height: 2160 });
  const badCrc = png.slice(); badCrc[29]! ^= 1;
  for (const bytes of [badCrc, png.slice(1), png.slice(0, -1), new Uint8Array([...png, 0])]) expect(() => pngDimensions(bytes)).toThrow();
  expect(() => pngDimensions(new Uint8Array(Buffer.from(fixtures.tooWide, 'base64')))).toThrow(/4096/);
  expect(() => pngDimensions(new Uint8Array(Buffer.from(fixtures.animated, 'base64')))).toThrow(/still PNG/);
});
it('writes the exact native packed index word and preserves every brightness palette byte', () => {
  const texture = paletteTexture(), indices = new Uint8Array(24);
  indices.set([1, 2, 3]); const encoded = encodeWallIndices(texture, indices);
  expect([...encoded.slice(2048, 2052)]).toEqual([0x0c, 0x41, 0, 0]);
  expect(encoded.slice(0, 2048)).toEqual(texture.bytes.slice(0, 2048));
  expect([...wallTextureRgba({ ...texture, bytes: encoded }, 0).slice(0, 12)]).toEqual([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
  expect(texture.bytes.slice(2048).every(byte => byte === 0)).toBe(true);
  const wide = { ...texture, width: 6, bytes: new Uint8Array(2080) }, wideIndices = new Uint8Array(48);
  wide.bytes.set(texture.bytes.slice(0, 2048)); wideIndices.set([1, 2, 3, 4, 5, 6]);
  expect([...encodeWallIndices(wide, wideIndices).slice(2064, 2068)]).toEqual([0x18, 0xa4, 0, 0]);
  indices[0] = 32; expect(() => encodeWallIndices(texture, indices)).toThrow(/0–31/);
});
it('matches native colours with the evidenced weighted metric and composites alpha onto palette background', () => {
  const texture = paletteTexture(), rgba = new Uint8ClampedArray(24 * 4); rgba.fill(255);
  rgba.set([255, 0, 0, 0, 128, 128, 128, 255, 0, 0, 255, 255]);
  const converted = convertWallRgba(texture, rgba);
  expect([...converted.bytes.slice(2048, 2050)]).toEqual([0x0c, 0x40]); // black, green (breaks the unweighted RGB tie), blue
  expect(converted.changedPixels).toBeGreaterThan(0);
  expect(converted.bytes.slice(0, 2048)).toEqual(texture.bytes.slice(0, 2048));
  expect(() => convertWallRgba(texture, rgba.slice(4))).toThrow(/fitted/);
});
it('stores original PNG source and native output in one undoable v10 asset edit, preserving legacy archives and rejecting source loss', () => {
  const { document, sheet, replacement } = wallOverrideFixture();
  let history = execute({ document, past: [], future: [] }, replaceWallAsset(document, sheet.id, replacement, png));
  const bytes = encodeProject(history.document), entries = unzipSync(bytes), manifest = JSON.parse(strFromU8(entries['alien-breed-project.json']!));
  expect(manifest).toMatchObject({ version: 10, wallSourcePng: [sheet.id] });
  expect(decodeProject(bytes).wallOverrides![0]).toEqual({ texture: sheet.id, bytes: replacement, sourcePng: png });
  history = undo(history); expect(history.document.wallOverrides).toBeUndefined();
  history = redo(history); expect(history.document.wallOverrides![0]!.sourcePng).toEqual(png);
  expect(applyCommand(history.document, replaceWallAsset(history.document, sheet.id, replacement)).wallOverrides![0]!.sourcePng).toEqual(png);
  entries['alien-breed-project.json'] = strToU8(JSON.stringify({ ...manifest, version: 9 }));
  expect(() => decodeProject(zipSync(entries))).toThrow(/version 10/);
  entries['alien-breed-project.json'] = strToU8(JSON.stringify(manifest)); delete entries[`assets/wall-sources/${sheet.id}.png`];
  expect(() => decodeProject(zipSync(entries))).toThrow(/PNG source list/);
});
