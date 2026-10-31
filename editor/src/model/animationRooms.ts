import { BinaryReader } from './binary';
import type { LevelDocument } from './level';

/** src/level.c lift parsing and normalize_amiga_water_list prefer a raw slot
 * inside the room table. Only out-of-range lift/water words use the room's
 * stored file ID. Doors and switches always require direct slot identities.
 * In-range missing slots must not be redirected to an unrelated file alias. */
export function resolveAnimationRoom(document: LevelDocument, id: number, aliases = false): number | undefined {
  if (!Number.isInteger(id) || id < 0) return undefined;
  const slots = (new BinaryReader(document.files.graphics, 'animation room table').i32(12) - 16) / 4;
  return id < slots ? document.zones.find(zone => zone.id === id)?.id
    : aliases ? document.zones.find(zone => zone.fileId === id)?.id : undefined;
}
