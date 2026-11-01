import { parseWallCatalog, readWallTexture, type WallSheet, type WallTexture } from './wallTextures';
import type { WallOverride } from './wallOverrides';

let catalog: Promise<readonly WallSheet[]> | undefined;
const textures = new Map<number, Promise<WallTexture>>();
export function loadWallCatalog(): Promise<readonly WallSheet[]> {
  catalog ??= fetch(`${import.meta.env.BASE_URL}reference/walls/catalog.json`).then(async response => {
    if (!response.ok) throw new Error('The wall material catalog could not be loaded. Check the editor asset installation and retry.');
    return parseWallCatalog(await response.json());
  }).catch(error => { catalog = undefined; throw error; });
  return catalog;
}
export function loadWallTexture(sheet: WallSheet, overrides?: readonly WallOverride[]): Promise<WallTexture> {
  const custom = overrides?.find(asset => asset.texture === sheet.id);
  if (custom) return Promise.resolve(readWallTexture(custom.bytes, sheet));
  let pending = textures.get(sheet.id);
  if (!pending) {
    pending = fetch(`${import.meta.env.BASE_URL}reference/walls/${sheet.file}`).then(async response => {
      if (!response.ok) throw new Error(`Wall material ${sheet.name} could not be loaded. Check the editor assets and retry.`);
      return readWallTexture(new Uint8Array(await response.arrayBuffer()), sheet);
    }).catch(error => { textures.delete(sheet.id); throw error; });
    textures.set(sheet.id, pending);
  }
  return pending;
}
