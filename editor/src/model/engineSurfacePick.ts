import { polygonPlane } from './surfaces';
import type { LevelDocument } from './level';

export interface NativeSurfacePick { offset: number; upper: boolean; type: number }
export interface EngineSurfacePick extends NativeSurfacePick {
  zone: number; kind: string; texture?: number; height?: number;
}

/** Resolve against the exact compiled artifact that produced the captured frame.
 * Generated offsets are inspection identities, never editable-source addresses. */
export function resolveEngineSurface(document: LevelDocument, zoneId: number, pick: NativeSurfacePick | null | undefined): EngineSurfacePick | undefined {
  if (!pick || !Number.isInteger(pick.offset) || typeof pick.upper !== 'boolean') return;
  const zone = document.zones.find(zone => zone.id === zoneId);
  const record = (pick.upper ? zone?.upper : zone?.lower)?.find(record => record.offset === pick.offset && record.type === pick.type);
  if (!record) return;
  const kind = record.type === 0 ? 'Wall' : record.type === 13 ? 'See-through wall' : record.type === 5 ? 'Curved wall'
    : record.type === 7 ? 'Water' : polygonPlane(record.type) === 'ceiling' ? 'Ceiling' : polygonPlane(record.type) === 'floor' ? 'Floor' : undefined;
  if (!kind) return;
  const height = record.type === 1 ? (pick.upper ? zone!.upperFloor : zone!.floor)
    : record.type === 2 ? (pick.upper ? zone!.upperCeiling : zone!.ceiling) : record.height;
  return { ...pick, zone: zoneId, kind, ...(record.texture === undefined ? {} : { texture: record.texture }),
    ...(height === undefined ? {} : { height }) };
}
