import type { SourceWaterSurface } from './sourceWater';
import type { UpperStorey } from './roomStoreys';
import { checkedInteger } from './binary';
import type { LevelDocument, Zone } from './level';
import type { RoomProperties } from './roomProperties';

/** AMOS DEFHEIGHT/DEFROOFHEIGHT/DEFUPPER* edit independent air boundaries.
 * Current zone headers store 24.8; floor/roof draw records store WORD quarters.
 * Water is intentionally separate (DEFWATERHEIGHT), not moved with a floor. */
export interface RoomHeights { readonly floor: number; readonly ceiling: number; readonly upperFloor: number; readonly upperCeiling: number }
export const roomHeightFields = ['floor', 'ceiling', 'upperFloor', 'upperCeiling'] as const;
export function restoreRoomHeights(value: unknown, template: Zone, authored?: UpperStorey | false): Readonly<RoomHeights> {
  if (!value || typeof value !== 'object' || Object.keys(value).length !== 4
    || Object.keys(value).some(key => !(roomHeightFields as readonly string[]).includes(key))) throw new Error('Room heights need all four floor/ceiling values.');
  const result = {} as Record<typeof roomHeightFields[number], number>;
  const upper = authored !== false && (!!authored || template.upper.length > 0 && template.upperFloor > template.upperCeiling);
  for (const field of roomHeightFields) {
    const height = (value as Record<string, unknown>)[field];
    if (typeof height !== 'number' || !Number.isFinite(height)) throw new Error(`Room ${field} height is invalid.`);
    result[field] = height;
  }
  // Restoring the exact authored template must remain possible for closed
  // moving doors and native sentinel values. New planes need valid air bands.
  if (authored === undefined && roomHeightFields.every(field => result[field] === template[field])) return Object.freeze(result);
  for (const field of roomHeightFields) if (upper || !field.startsWith('upper'))
    checkedInteger(result[field] * 4, -32768, 32767, `${field} height in quarter units`);
  if (result.floor <= result.ceiling) throw new Error('The lower ceiling must be above the lower floor (a smaller height value).');
  if (upper) {
    if (result.upperFloor <= result.upperCeiling || result.upperFloor > result.ceiling)
      throw new Error('The upper ceiling must be above its floor, and the upper floor must be at or above the lower ceiling.');
  } else if (authored === false ? result.upperFloor !== result.upperCeiling : result.upperFloor !== template.upperFloor || result.upperCeiling !== template.upperCeiling)
    throw new Error('This template has no active upper storey. Choose a two-storey template before authoring upper heights.');
  return Object.freeze(result);
}

export function roomSettings(document: Pick<LevelDocument, 'zones'>, room: { recordZone: number; heights?: RoomHeights; properties?: RoomProperties; upperStorey?: UpperStorey | false; waterSurface?: SourceWaterSurface | false }): Zone {
  const template = document.zones.find(zone => zone.id === room.recordZone);
  if (!template) throw new Error('The room height settings have no original template.');
  const settings = room.heights || room.properties || room.upperStorey !== undefined ? { ...template, ...room.properties, ...room.heights, ...(room.upperStorey !== undefined ? { upper: room.upperStorey === false ? [] : template.lower } : {}) } : template;
  return room.waterSurface !== undefined ? { ...settings, water: room.waterSurface === false ? settings.floor + 4 : room.waterSurface.height } : settings;
}
export function airHeightsChanged(document: Pick<LevelDocument, 'zones'>, room: { recordZone: number; heights?: RoomHeights }): boolean {
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  return !!room.heights && roomHeightFields.some(field => room.heights![field] !== template[field]);
}
