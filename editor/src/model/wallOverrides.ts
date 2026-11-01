import catalog from './nativeWalls.json' with { type: 'json' };
import { parseWallCatalog, readWallTexture } from './wallTextures';
import { isSb } from './sb';

// Generated from io.c by prepare-assets. Paths, slots and byte dimensions are
// native declarations; projects cannot redefine them or add arbitrary files.
export const nativeWallSheets = parseWallCatalog(catalog);
export interface WallOverride { readonly texture: number; readonly bytes: Uint8Array }
export function restoreWallOverrides(value: unknown): readonly WallOverride[] {
  if (!Array.isArray(value) || !value.length || value.length > nativeWallSheets.length) throw new Error('Invalid project wall asset list.');
  const seen = new Set<number>();
  return Object.freeze(value.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('texture' in item) || typeof item.texture !== 'number' || !Number.isInteger(item.texture)
      || !('bytes' in item) || !(item.bytes instanceof Uint8Array) || Object.keys(item).some(key => !['texture', 'bytes'].includes(key))) throw new Error('Invalid project wall asset.');
    const sheet = nativeWallSheets[item.texture];
    if (!sheet || seen.has(item.texture)) throw new Error('Project wall assets contain a missing or duplicate native slot.');
    seen.add(item.texture);
    if (item.bytes.length !== sheet.size) throw new Error(`Wall ${sheet.name} must retain its native sheet size (${sheet.width} × ${sheet.rows}).`);
    if (isSb(item.bytes)) throw new Error('Project wall assets must contain decoded native bytes. Import packed sheets through the material tools.');
    return Object.freeze({ texture: item.texture, bytes: readWallTexture(item.bytes, sheet).bytes });
  }));
}
export function wallOverrideFiles(overrides: readonly WallOverride[] | undefined): Record<string, Uint8Array> {
  return Object.fromEntries((overrides ? restoreWallOverrides(overrides) : []).map(asset => [`includes/walls/${nativeWallSheets[asset.texture]!.name}`, asset.bytes]));
}
