import { checkedInteger } from './binary';
import type { Graphic, LevelDocument, Zone } from './level';
import { encodePolygonSurface, encodeWallSurface } from './surfaces';

export interface MechanismIdentity { readonly kind: 'door' | 'lift'; readonly id: number }
/** A retired controller's static geometry, in original record identities.
 * Phase is captured from its effective wall list, never generated pointers. */
export interface StaticMechanismPose extends MechanismIdentity {
  readonly height: number;
  readonly plane: boolean;
  readonly surface: number | null;
  readonly walls: readonly { readonly record: number; readonly horizontalOffset: number; readonly verticalOffset: number }[];
}
export function restoreMechanismIdentities(value: unknown): readonly MechanismIdentity[] {
  if (!Array.isArray(value) || value.length > 512) throw new Error('Removed controllers need at most 512 native identities.');
  const seen = new Set<string>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).length !== 2 || !['door', 'lift'].includes(item.kind)) throw new Error('A removed controller needs its native kind and ID.');
    const id = checkedInteger(item.id, 0, 255, 'Native controller ID'), key = `${item.kind}:${id}`;
    if (seen.has(key)) throw new Error('Removed controller identities must be unique.');
    seen.add(key); return Object.freeze({ kind: item.kind as 'door' | 'lift', id });
  }));
}
export function restoreStaticMechanismPoses(value: unknown, template: Zone): readonly StaticMechanismPose[] {
  if (!Array.isArray(value) || value.length > 512) throw new Error('Static controller poses need a bounded list.');
  const identities = restoreMechanismIdentities(value.map(item => ({ kind: item?.kind, id: item?.id })));
  const records = new Map([...template.lower, ...template.upper].map(record => [record.offset, record]));
  return Object.freeze(value.map((item, index) => {
    if (!item || Object.keys(item).length !== 6 || Object.keys(item).some(key => !['kind', 'id', 'height', 'plane', 'surface', 'walls'].includes(key)) || typeof item.plane !== 'boolean') throw new Error('A static controller pose has missing or unsupported fields.');
    if (typeof item.height !== 'number' || !Number.isFinite(item.height)) throw new Error('A static controller height must be numeric.');
    const height = checkedInteger(item.height * 4, -32768, 32767, 'Static controller height in quarters') / 4;
    if (item.surface !== null && !records.get(item.surface)?.polygon) throw new Error('A static controller surface must belong to its original room template.');
    if (!Array.isArray(item.walls) || item.walls.length > records.size) throw new Error('Static controller wall attachments are invalid.');
    const seen = new Set<number>();
    const walls = Object.freeze(item.walls.map((wall: Record<string, unknown>) => {
      if (!wall || Object.keys(wall).length !== 3 || Object.keys(wall).some(key => !['record', 'horizontalOffset', 'verticalOffset'].includes(key))) throw new Error('A static wall needs its original record and two texture offsets.');
      const record = checkedInteger(wall.record as number, 0, 16777215, 'Static wall record');
      if (!records.get(record)?.wall || seen.has(record)) throw new Error('Static controller walls must be unique records in their own room template.');
      seen.add(record);
      return Object.freeze({ record, horizontalOffset: checkedInteger(wall.horizontalOffset as number, -32768, 32767, 'Static wall horizontal offset'), verticalOffset: checkedInteger(wall.verticalOffset as number, -32768, 32767, 'Static wall vertical offset') });
    }));
    return Object.freeze({ ...identities[index]!, height, plane: item.plane, surface: item.surface as number | null, walls });
  }));
}
export interface StaticRoomTemplate { readonly recordZone: number; readonly staticMechanisms?: readonly StaticMechanismPose[] }
export function roomTemplate(document: Pick<LevelDocument, 'zones'>, room: StaticRoomTemplate): Zone {
  const original = document.zones.find(zone => zone.id === room.recordZone);
  if (!original) throw new Error('The room has no original settings template.');
  if (!room.staticMechanisms?.length) return original;
  let result = original;
  for (const pose of restoreStaticMechanismPoses(room.staticMechanisms, original)) {
    const field = pose.kind === 'door' ? 'ceiling' : 'floor';
    const layer = (records: readonly Graphic[], upper: boolean) => records.map(record => {
      const wall = pose.walls.find(wall => wall.record === record.offset);
      if (wall && record.wall) return { ...record, wall: { ...record.wall, horizontalOffset: wall.horizontalOffset, verticalOffset: wall.verticalOffset,
        [pose.kind === 'door' ? 'bottomFixed' : 'topFixed']: pose.height * 256 } };
      // renderer.c ties normal 1/2 polygons to live zone planes. Other draw
      // types keep their explicit heights unless this is the attached surface.
      if (record.polygon && (record.offset === pose.surface || !upper && pose.plane && record.type === (field === 'floor' ? 1 : 2)))
        return { ...record, polygon: { ...record.polygon, heightQuarter: pose.height * 4 } };
      return record;
    });
    result = { ...result, ...(pose.plane ? { [field]: pose.height } : {}), lower: layer(result.lower, false), upper: layer(result.upper, true) };
  }
  return result;
}
export function roomRecordBytes(document: Pick<LevelDocument, 'files' | 'zones'>, record: Graphic): Uint8Array {
  const original = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(item => item.offset === record.offset);
  if (record.wall === original?.wall && record.polygon === original?.polygon) return document.files.graphics.slice(record.offset, record.offset + record.size);
  return record.wall ? encodeWallSurface(record.wall) : record.polygon ? encodePolygonSurface(record.polygon) : document.files.graphics.slice(record.offset, record.offset + record.size);
}
