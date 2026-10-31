import { checkedInteger } from './binary';
import { pointInRoomOutline } from './geometry';
import type { LevelDocument } from './level';
import type { SourceRoom } from './roomSource';

/** ToTelZone/ToTelX/ToTelZ are authored per logical room. Null disables the
 * link, while a missing referenced room remains a dependency to repair. */
export interface SourceTeleport { readonly room: string; readonly destination: string | null; readonly x: number; readonly z: number }

export function restoreSourceTeleports(value: unknown): readonly SourceTeleport[] {
  if (!Array.isArray(value) || value.length > 256) throw new Error('The editable teleporter list exceeds the room limit.');
  const rooms = new Set<string>();
  return Object.freeze(value.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('room' in item) || typeof item.room !== 'string' || !item.room || item.room.length > 128
      || !('destination' in item) || (item.destination !== null && (typeof item.destination !== 'string' || !item.destination || item.destination.length > 128))
      || !('x' in item) || typeof item.x !== 'number' || !('z' in item) || typeof item.z !== 'number'
      || Object.keys(item).some(key => !['room', 'destination', 'x', 'z'].includes(key))) throw new Error('Invalid editable teleporter link.');
    if (rooms.has(item.room)) throw new Error(`Duplicate teleporter link for room ${item.room}.`);
    rooms.add(item.room);
    return Object.freeze({ room: item.room, destination: item.destination,
      x: checkedInteger(item.x, -32768, 32767, 'Teleport destination X'), z: checkedInteger(item.z, -32768, 32767, 'Teleport destination Z') });
  }));
}

/** Undefined denotes an enabled imported link whose destination was deleted;
 * it must not appear disabled in the inspector. */
export function editableTeleport(document: LevelDocument, room: SourceRoom): { destination: string | null | undefined; x: number; z: number } {
  const authored = document.sourceTeleports?.find(item => item.room === room.id);
  if (authored) return authored;
  const template = document.zones.find(zone => zone.id === room.recordZone);
  if (!template) throw new Error(`Room ${room.id} has no teleporter settings template.`);
  const { zone, x, z } = template.teleport;
  return { destination: zone < 0 ? null : document.roomSource.rooms.find(item => item.originZone === zone)?.id, x, z };
}

export function placeSourceTeleport(document: LevelDocument, room: string, destination: string | null, x: number, z: number): readonly SourceTeleport[] {
  const links = restoreSourceTeleports([...(document.sourceTeleports ?? []).filter(item => item.room !== room), { room, destination, x, z }]);
  if (!document.roomSource.rooms.some(item => item.id === room)) throw new Error('Choose an existing editable room for this teleporter.');
  if (destination !== null) {
    const target = document.roomSource.rooms.find(item => item.id === destination);
    if (!target || !pointInRoomOutline(target.vertices, { x, z })) throw new Error('Place the teleport destination inside its selected editable room.');
  }
  if (document.sourceTeleports?.some(item => item.room === room && item.destination === destination && item.x === x && item.z === z)) return document.sourceTeleports;
  return links;
}
