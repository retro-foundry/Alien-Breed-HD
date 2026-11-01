import { normaliseRoomOutline, type Position } from './geometry';
import type { LevelDocument } from './level';
import type { SourceRoom } from './roomSource';
import { hasNativeWallMaterials } from './roomMaterials';

/** Compare geometry without discarding the source's collinear wall identities.
 * Inserted portal/T-junction points do not change a whole-room translation. */
export function outlineTranslation(original: readonly Position[], target: readonly Position[]): Position | undefined {
  const corners = (points: readonly Position[]) => {
    const outline = normaliseRoomOutline(points);
    return normaliseRoomOutline(outline.filter((b, index) => {
      const a = outline[(index + outline.length - 1) % outline.length]!, c = outline[(index + 1) % outline.length]!;
      return (b.x - a.x) * (c.z - b.z) !== (b.z - a.z) * (c.x - b.x);
    }));
  };
  const from = corners(original), to = corners(target);
  if (from.length !== to.length) return undefined;
  const x = to[0]!.x - from[0]!.x, z = to[0]!.z - from[0]!.z;
  return to.every((point, index) => point.x - from[index]!.x === x && point.z - from[index]!.z === z) ? { x, z } : undefined;
}

export function nativeRoomTranslation(document: LevelDocument, room: SourceRoom): Position | undefined {
  if (!hasNativeWallMaterials(room.originZone, room.wallSources)) return undefined;
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  return outlineTranslation(template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })), room.vertices);
}
