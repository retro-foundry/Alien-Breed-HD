import { checkedInteger } from './binary';
import type { Zone } from './level';

export const roomPropertyFields = ['water', 'brightness', 'upperBrightness', 'control', 'upperControl', 'backdrop', 'floorSound', 'upperFloorSound'] as const;
export type RoomProperties = Readonly<Pick<Zone, typeof roomPropertyFields[number]>>;

export function restoreRoomProperties(value: unknown): RoomProperties {
  if (!value || typeof value !== 'object' || Object.keys(value).length !== roomPropertyFields.length
    || Object.keys(value).some(key => !(roomPropertyFields as readonly string[]).includes(key))) throw new Error('Room properties need every native lighting, water, sky and sound field.');
  const input = value as Record<string, unknown>, result: Record<string, number | boolean> = {};
  for (const field of roomPropertyFields) {
    const item = input[field];
    if (field === 'backdrop') {
      if (typeof item !== 'boolean') throw new Error('Room sky setting must be true or false.');
      result[field] = item; continue;
    }
    if (typeof item !== 'number') throw new Error(`Room ${field} is invalid.`);
    if (field === 'water') checkedInteger(item * 256, -2147483648, 2147483647, 'Native water height');
    else checkedInteger(item, field === 'control' || field === 'upperControl' ? -128 : -32768, field === 'control' || field === 'upperControl' ? 127 : 32767, `Room ${field}`);
    result[field] = item;
  }
  return Object.freeze(result) as RoomProperties;
}

export function captureRoomProperties(zone: Zone): RoomProperties {
  return restoreRoomProperties(Object.fromEntries(roomPropertyFields.map(field => [field, zone[field]])));
}

/** Defs.i/level.c's 48-byte zone header; relationship/list offsets are rebuilt
 * separately. These fields contain authored values, not generated pointers. */
export function writeRoomProperties(header: Uint8Array, value: RoomProperties): void {
  const properties = restoreRoomProperties(value);
  if (header.length !== 48) throw new Error('Room properties require a complete native zone header.');
  const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  view.setInt32(18, properties.water * 256);
  view.setInt16(22, properties.brightness); view.setInt16(24, properties.upperBrightness);
  view.setInt8(26, properties.control); view.setInt8(27, properties.upperControl);
  view.setInt16(36, properties.backdrop ? 1 : 0);
  view.setInt16(44, properties.floorSound); view.setInt16(46, properties.upperFloorSound);
}
