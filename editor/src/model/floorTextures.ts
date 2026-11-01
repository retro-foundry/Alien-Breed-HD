import type { PolygonMaterialMode } from './surfaces';
import { checkedInteger } from './binary';
import catalog from './nativeFloor.json' with { type: 'json' };

export interface FloorTextures { readonly pixels: Uint8Array; readonly palette: Uint8Array }
export const floorPaletteLevels = catalog.paletteLevels;
// leveld.asc DEFFLOORGRAPH / DEFROOFGRAPH compile R as (R/4)*256+(R mod 4).
export const floorPatternOffsets = Object.freeze(Array.from({ length: 16 }, (_, index) => (index >> 2) * 256 + (index & 3)));
// FGPUT / RGPUT q/w limits, exposed as repeat factors instead of AMOS keys.
export const floorRepeatScales = Object.freeze(Array.from({ length: 8 }, (_, index) => index - 5));
export function readFloorTextures(pixels: Uint8Array, palette: Uint8Array): FloorTextures {
  if (pixels.length !== catalog.size || palette.length !== floorPaletteLevels * 512)
    throw new Error('Native floor material files have the wrong size. Reinstall the editor assets and retry.');
  return { pixels, palette };
}

/** renderer.c FloorLine: 64x64 patterns interleaved four ways in both axes.
 * FloorPalScaled supplies 256 big-endian RGB12 colors per native shade. */
export function floorTextureRgba(assets: FloorTextures, tile: number, shade: number): Uint8ClampedArray {
  checkedInteger(tile, 0, catalog.size - 1 - 0xFCFC, 'Floor tile offset');
  checkedInteger(shade, 0, floorPaletteLevels - 1, 'Floor preview shade');
  readFloorTextures(assets.pixels, assets.palette);
  const rgba = new Uint8ClampedArray(64 * 64 * 4);
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
    const texel = assets.pixels[tile + y * 1024 + x * 4]!;
    const at = shade * 512 + texel * 2, color = assets.palette[at]! * 256 + assets.palette[at + 1]!;
    const out = (y * 64 + x) * 4;
    rgba[out] = ((color >>> 8) & 15) * 17; rgba[out + 1] = ((color >>> 4) & 15) * 17;
    rgba[out + 2] = (color & 15) * 17; rgba[out + 3] = 255;
  }
  return rgba;
}

const pending = new Map<PolygonMaterialMode, Promise<FloorTextures>>();
export function loadFloorTextures(mode: PolygonMaterialMode = 'normal'): Promise<FloorTextures> {
  let request = pending.get(mode);
  if (!request) {
    request = Promise.all((mode === 'normal' ? ['floortile', 'palette.bin'] : [`${mode}-tile`, `${mode}-palette.bin`]).map(async file => {
      const response = await fetch(`${import.meta.env.BASE_URL}reference/floors/${file}`);
      if (!response.ok) throw new Error(`Floor material ${file} could not be loaded. Check the editor assets and retry.`);
      return new Uint8Array(await response.arrayBuffer());
    })).then(([pixels, palette]) => readFloorTextures(pixels!, palette!)).catch(error => { pending.delete(mode); throw error; });
    pending.set(mode, request);
  }
  return request;
}
