import { wallTextureRgba, type WallTexture } from './wallTextures';

/** Native io.c / WallChunk.s packing: 3 five-bit indices per BE word.
 * Retain all 2048 authored palette bytes, including non-linear lighting. */
export function encodeWallIndices(texture: WallTexture, indices: Uint8Array): Uint8Array {
  wallTextureRgba(texture, 0); // Validate the native dimensions and palette span.
  if (indices.length !== texture.width * texture.height || indices.some(index => index > 31)) throw new Error('Wall pixels need one native 0–31 index per pixel.');
  const bytes = texture.bytes.slice();
  for (let x = 0; x < texture.width; x += 3) for (let y = 0; y < texture.height; y++) {
    const at = y * texture.width + x;
    const word = indices[at]! | (indices[at + 1]! << 5) | (indices[at + 2]! << 10);
    const offset = 2048 + ((x / 3) * texture.height + y) * 2;
    bytes[offset] = word >>> 8; bytes[offset + 1] = word & 255;
  }
  return bytes;
}

/** Modern PNG matching into an existing native palette. The 3R/4G/2B
 * absolute-distance expression appears in the inactive tail of
 * 256wallchunk.amos. Its .256wad format is NOT the native RGB12 format here;
 * this is an explicit modern import policy, not a claim of executing that tool. */
export function convertWallRgba(texture: WallTexture, rgba: Uint8ClampedArray): { bytes: Uint8Array; changedPixels: number } {
  wallTextureRgba(texture, 0);
  if (rgba.length !== texture.width * texture.height * 4) throw new Error('The image must first be fitted to the native sheet dimensions.');
  const colors = Array.from({ length: 32 }, (_, index) => {
    const word = texture.bytes[index * 2]! * 256 + texture.bytes[index * 2 + 1]!;
    return [((word >>> 8) & 15) * 17, ((word >>> 4) & 15) * 17, (word & 15) * 17];
  });
  const indices = new Uint8Array(texture.width * texture.height), cache = new Map<number, number>();
  let changedPixels = 0;
  for (let pixel = 0; pixel < indices.length; pixel++) {
    const at = pixel * 4, alpha = rgba[at + 3]!;
    const rgb = [0, 1, 2].map(channel => Math.round((rgba[at + channel]! * alpha + colors[0]![channel]! * (255 - alpha)) / 255));
    const key = (rgb[0]! << 16) | (rgb[1]! << 8) | rgb[2]!;
    let index = cache.get(key);
    if (index === undefined) {
      let distance = Infinity; index = 0;
      for (let color = 0; color < colors.length; color++) {
        const c = colors[color]!, d = Math.abs(rgb[0]! - c[0]!) * 3 + Math.abs(rgb[1]! - c[1]!) * 4 + Math.abs(rgb[2]! - c[2]!) * 2;
        if (d < distance) { distance = d; index = color; }
      }
      cache.set(key, index);
    }
    indices[pixel] = index;
    if (alpha !== 255 || colors[index]!.some((channel, c) => channel !== rgba[at + c])) changedPixels++;
  }
  return { bytes: encodeWallIndices(texture, indices), changedPixels };
}
