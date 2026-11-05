import { checkedInteger } from './binary';
import type { Graphic, LevelDocument } from './level';
import type { RoomChange, SourceRoom } from './roomSource';
import { roomSettings } from './roomHeights';
import { polygonPlane } from './surfaces';
import { assertStaticRoom } from './roomStoreys';

export interface RoomPlaneSource { readonly upper: boolean; readonly record: number }

/** One original material reference per pattern; retain the explicitly selected
 * record when several surfaces use the same pattern. */
export function normalPlaneChoices(document: Pick<LevelDocument, 'zones'>, type: 1 | 2, selected?: number): Graphic[] {
  const choices = new Map<number, Graphic>();
  for (const zone of document.zones) for (const record of [...zone.lower, ...zone.upper])
    if (record.type === type && (!choices.has(record.texture!) || record.offset === selected)) choices.set(record.texture!, record);
  return [...choices.values()];
}
export function restorePlaneSources(value: unknown): readonly RoomPlaneSource[] {
  if (!Array.isArray(value) || value.length > 4) throw new Error('A room can have at most four added floor/ceiling surfaces.');
  const keys = new Set<string>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).length !== 2 || typeof item.upper !== 'boolean' || typeof item.record !== 'number') throw new Error('An added surface needs its storey and original material record.');
    const record = checkedInteger(item.record, 0, 16777215, 'Surface material record'), key = `${item.upper}:${record}`;
    if (keys.has(key)) throw new Error('Duplicate added room surface.'); keys.add(key);
    return Object.freeze({ upper: item.upper, record });
  }));
}

export function roomPlaneRecords(document: Pick<LevelDocument, 'zones' | 'roomAppearances'>, room: SourceRoom, upper: boolean): Graphic[] {
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const selection = upper ? room.upperStorey : undefined;
  const defaults = document.roomAppearances?.find(item => item.room === room.id)?.surfaces;
  const references = defaults && (upper ? [defaults.upperFloor, defaults.upperCeiling] : [defaults.lowerFloor, defaults.lowerCeiling]).filter((offset): offset is number => offset !== null);
  const inherited = selection ? [selection.floorMaterial, selection.ceilingMaterial].map(offset => {
    const record = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === offset);
    if (!record || ![1, 2].includes(record.type)) throw new Error('An upper storey has lost its normal floor or ceiling material reference.');
    return record;
  }) : references ? references.map(offset => {
    const record = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === offset);
    if (!record?.polygon || ![1, 2].includes(record.type)) throw new Error('A default room surface has lost its normal pattern reference. Choose the room appearance again.');
    return record;
  }) : (upper ? template.upper : template.lower).filter(record => polygonPlane(record.type));
  const added = (room.planeSources ?? []).filter(item => item.upper === upper).map(item => {
    const record = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === item.record);
    if (!record || ![1, 2].includes(record.type)) throw new Error('An added room surface has lost its normal pattern reference.');
    return record;
  });
  if (added.some((record, index) => inherited.some(item => polygonPlane(item.type) === polygonPlane(record.type)) || added.slice(0, index).some(item => item.type === record.type))) throw new Error('An added surface overlaps an existing floor or ceiling.');
  return [...inherited, ...added];
}

export function addRoomPlane(document: LevelDocument, id: string, upper: boolean, record: number): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select a room before adding a floor or ceiling.');
  assertStaticRoom(document, room);
  if (upper && !roomSettings(document, room).upper.length) throw new Error('Add an upper storey first.');
  const planeSources = restorePlaneSources([...(room.planeSources ?? []), { upper, record }]);
  const sky = [...(room.sky ?? [false, false])];
  const updated = { ...room, planeSources };
  const added = roomPlaneRecords(document, updated, upper).find(item => item.offset === record)!;
  if (added.type === 2) sky[upper ? 1 : 0] = false;
  const result = Object.freeze({ ...updated, sky: Object.freeze(sky) });
  return { label: added.type === 2 ? 'Create ceiling' : 'Create floor', before, after: Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === id ? result : item)) }) };
}
