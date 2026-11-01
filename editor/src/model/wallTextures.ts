import { checkedInteger } from './binary';
import { unpackSb } from './sb';

export interface WallSheet { id: number; name: string; file: string; size: number; rows: number; width: number }
export interface WallTexture { width: number; height: number; bytes: Uint8Array }
export function parseWallCatalog(value: unknown): readonly WallSheet[] {
  if (!value || typeof value !== 'object' || !('format' in value) || value.format !== 'alien-breed-wall-catalog'
    || !('version' in value) || value.version !== 1 || !('walls' in value) || !Array.isArray(value.walls) || !value.walls.length || value.walls.length > 256)
    throw new Error('The wall material catalog is invalid. Rebuild the editor assets.');
  return Object.freeze(value.walls.map((item: unknown, id: number) => {
    if (!item || typeof item !== 'object') throw new Error('A wall material catalog entry is invalid.');
    const wall = item as WallSheet;
    if (wall.id !== id || typeof wall.name !== 'string' || !/^[\w.-]+\.wad$/i.test(wall.name) || wall.file !== `${id}.wad`
      || ![8, 16, 32, 64, 128].includes(wall.rows)) throw new Error(`Wall material ${id} has invalid native metadata.`);
    checkedInteger(wall.size, 2050, 16 * 1024 * 1024, 'Wall sheet byte count');
    checkedInteger(wall.width, 3, 32768, 'Wall sheet width');
    if (wall.width % 3 || wall.size !== 2048 + wall.width / 3 * wall.rows * 2) throw new Error(`Wall material ${id} dimensions do not match its packed bytes.`);
    return Object.freeze({ id, name: wall.name, file: wall.file, size: wall.size, rows: wall.rows, width: wall.width });
  }));
}

export function readWallTexture(source: Uint8Array, sheet: WallSheet): WallTexture {
  const bytes = unpackSb(source, sheet.name);
  if (bytes.length !== sheet.size || bytes.length !== 2048 + sheet.width / 3 * sheet.rows * 2)
    throw new Error(`Wall material ${sheet.name} has damaged or mismatched pixel data.`);
  return { width: sheet.width, height: sheet.rows, bytes };
}

/** io.c io_dump_textures and WallChunk.s: three 5-bit texels packed in
 * each big-endian word, columns in vertical strips. The first 2048 bytes
 * are 32 authored RGB12 palette levels, not a generated RGB colour ramp. */
export function wallTextureRgba(texture: WallTexture, paletteLevel: number): Uint8ClampedArray {
  checkedInteger(paletteLevel, 0, 31, 'Wall palette level');
  const { width, height, bytes } = texture;
  checkedInteger(width, 3, 32768, 'Wall sheet width');
  if (width % 3 || ![8, 16, 32, 64, 128].includes(height) || bytes.length !== 2048 + width / 3 * height * 2)
    throw new Error('Wall texture dimensions do not match its packed bytes.');
  const rgba = new Uint8ClampedArray(width * height * 4), palette = paletteLevel * 64;
  for (let x = 0; x < width; x++) for (let y = 0; y < height; y++) {
    const address = 2048 + (Math.floor(x / 3) * height + y) * 2;
    const texel = ((bytes[address]! * 256 + bytes[address + 1]!) >>> ((x % 3) * 5)) & 31;
    const color = bytes[palette + texel * 2]! * 256 + bytes[palette + texel * 2 + 1]!;
    const at = (y * width + x) * 4;
    rgba[at] = ((color >>> 8) & 15) * 17; rgba[at + 1] = ((color >>> 4) & 15) * 17;
    rgba[at + 2] = (color & 15) * 17; rgba[at + 3] = 255;
  }
  return rgba;
}
