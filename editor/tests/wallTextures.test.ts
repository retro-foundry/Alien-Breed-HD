import { expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import oracle from './fixtures/original-walls.json';
import { parseWallCatalog, readWallTexture, wallTextureRgba } from '../src/model/wallTextures';

const catalog = oracle.walls.map(wall => ({ id: wall.id, name: wall.name, file: `${wall.id}.wad`, width: wall.width,
  rows: wall.height, size: 2048 + wall.width / 3 * wall.height * 2 }));
const envelope = (walls: unknown) => ({ format: 'alien-breed-wall-catalog', version: 1, walls });
const hash = (bytes: Uint8Array | Uint8ClampedArray) => createHash('sha256').update(bytes).digest('hex');
it('records the current native dump implementation so renderer-side IO changes require reviewed oracle regeneration', () => {
  const source = readFileSync(new URL('../../src/io.c', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  expect(createHash('sha256').update(source).digest('hex')).toBe(oracle.nativeSourceSha256);
});
for (const wall of oracle.walls) it(`matches native C decompression and every dumped RGBA pixel for ${wall.name}`, () => {
  const sheet = parseWallCatalog(envelope(catalog))[wall.id]!;
  const texture = readWallTexture(new Uint8Array(readFileSync(new URL(`../../amiga/data/includes/walls/${wall.name}`, import.meta.url))), sheet);
  expect(hash(texture.bytes)).toBe(wall.unpackedSha256);
  expect(hash(wallTextureRgba(texture, 16))).toBe(wall.rgbaSha256);
});

it('uses the actual palette levels and all three packed five-bit channels', () => {
  const bytes = new Uint8Array(2048 + 16);
  bytes.set([0x00, 0xf0, 0x0f, 0x00, 0x00, 0x0f], 64 + 2);
  const word = 1 | (2 << 5) | (3 << 10); bytes[2048] = word >>> 8; bytes[2049] = word & 255;
  const texture = { width: 3, height: 8, bytes };
  expect([...wallTextureRgba(texture, 1).slice(0, 12)]).toEqual([0, 255, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255]);
  expect([...wallTextureRgba(texture, 31).slice(0, 4)]).toEqual([0, 0, 0, 255]);
  for (const level of [-1, 32, 1.5, NaN]) expect(() => wallTextureRgba(texture, level)).toThrow();
  expect(() => wallTextureRgba({ ...texture, bytes: bytes.slice(1) }, 16)).toThrow(/dimensions/);
});

it('rejects invalid catalogs, mismatched dimensions and damaged sheets before showing any made-up pixels', () => {
  for (const change of [{ id: 9 }, { name: '../escape.wad' }, { file: '../escape.wad' }, { rows: 63 }, { width: 4 }, { size: 2050 }])
    expect(() => parseWallCatalog(envelope([{ ...catalog[0], ...change }]))).toThrow();
  expect(() => parseWallCatalog(envelope([]))).toThrow();
  expect(() => readWallTexture(new Uint8Array(2048), catalog[0]!)).toThrow(/damaged/);
});
