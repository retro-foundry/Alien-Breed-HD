import { checkedInteger } from './binary';
import type { Position } from './geometry';
import type { LevelDocument } from './level';
import type { RoomChange } from './roomSource';

/** Native pointBrights lower/upper words, optionally scoped to a graphics
 * face so floor, ceiling and wall endpoints can retain distinct identities. */
export interface CornerLight extends Position { readonly record?: number; readonly lowerBrightness: number; readonly upperBrightness: number }
/** level.c level_get_point_brightness / AB3DI.s currentPointBrights:
 * signed base byte, low-nibble effect, high-nibble blend minus one. */
export function decodeCornerWord(word: number) {
  checkedInteger(word, -32768, 32767, 'Corner light word');
  const brightness = word << 24 >> 24, selector = word >>> 8 & 15;
  return { brightness, animation: brightness >= 0 && selector >= 1 ? selector : 0, strength: (word >>> 12 & 15) + 1 };
}
export function encodeCornerWord(brightness: number, animation: number, strength: number): number {
  checkedInteger(brightness, -128, 127, 'Corner brightness'); checkedInteger(animation, 0, 15, 'Corner light animation'); checkedInteger(strength, 1, 16, 'Corner animation strength');
  if (!animation) return brightness;
  if (brightness < 0) throw new Error('Animated corner lighting needs a nonnegative base brightness. Choose Static or raise the brightness to zero or above.');
  return (((strength - 1) * 16 + animation) * 256 + brightness) << 16 >> 16;
}
export function restoreCornerLights(value: unknown, vertices?: readonly Position[]): readonly CornerLight[] {
  if (!Array.isArray(value) || value.length > 8192) throw new Error('Invalid room corner lighting list.');
  const seen = new Set<string>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item) || Object.keys(item).some(key => !['x', 'z', 'record', 'lowerBrightness', 'upperBrightness'].includes(key))) throw new Error('Corner lighting needs coordinates and lower/upper brightness words.');
    const x = checkedInteger(item.x, -32768, 32767, 'Lighting corner X'), z = checkedInteger(item.z, -32768, 32767, 'Lighting corner Z');
    if (vertices && !vertices.some(point => point.x === x && point.z === z)) throw new Error('Corner lighting must belong to an existing room corner.');
    const record = 'record' in item ? checkedInteger(item.record, 0, 16777215, 'Corner lighting face') : undefined;
    const key = `${x}:${z}:${record ?? 'all'}`; if (seen.has(key)) throw new Error('A corner has duplicate lighting for the same surface.'); seen.add(key);
    return Object.freeze({ x, z, ...(record === undefined ? {} : { record }), lowerBrightness: checkedInteger(item.lowerBrightness, -32768, 32767, 'Lower corner brightness'), upperBrightness: checkedInteger(item.upperBrightness, -32768, 32767, 'Upper corner brightness') });
  }));
}

export function setCornerLight(document: LevelDocument, id: string, corner: number, record: number | null, value: { lowerBrightness: number; upperBrightness: number } | null): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select a room before editing corner lighting.');
  checkedInteger(corner, 0, room.vertices.length - 1, 'Lighting corner');
  if (record !== null && !document.zones.some(zone => [...zone.lower, ...zone.upper].some(item => item.offset === record && item.points.length))) throw new Error('Choose an existing graphics face for this corner light.');
  const point = room.vertices[corner]!;
  const others = (room.cornerLights ?? []).filter(item => item.x !== point.x || item.z !== point.z || (item.record ?? null) !== record);
  const cornerLights = restoreCornerLights(value ? [...others, { ...point, ...(record === null ? {} : { record }), ...value }] : others, room.vertices);
  if (JSON.stringify(room.cornerLights ?? []) === JSON.stringify(cornerLights)) return { label: 'Edit corner lighting', before, after: before };
  const { cornerLights: _old, ...retained } = room;
  return { label: value ? 'Edit corner lighting' : 'Reset corner lighting', before, after: Object.freeze({ ...before,
    rooms: Object.freeze(before.rooms.map(item => item.id === id ? Object.freeze({ ...retained, ...(cornerLights.length ? { cornerLights } : {}) }) : item)) }) };
}

export function clearCornerLights(document: LevelDocument, id: string): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select an existing room to reset corner lighting.');
  const { cornerLights: _old, ...retained } = room;
  return { label: 'Reset room corner lighting', before, after: room.cornerLights?.length ? Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === id ? Object.freeze(retained) : item)) }) : before };
}
