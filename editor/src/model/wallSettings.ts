import { activeNativeSwitches } from './nativeSwitches';
import { checkedInteger } from './binary';
import { restoreBoundaryMaterials } from './boundaryMaterials';
import type { LevelDocument } from './level';
import type { RoomChange } from './roomSource';
import { surfaceMaterialCurrent, type SurfaceMaterialTarget } from './wallMaterials';
import { restoreWallRepeat, type WallRepeat } from './wallRepeat';

export interface WallSettingsValues {
  readonly brightness?: number;
  readonly horizontalOffset?: number;
  readonly verticalOffset?: number;
  readonly repeat?: WallRepeat;
}
export type WallSettings = WallSettingsValues & { readonly upper: boolean } & ({ readonly record: number } | { readonly boundary: number });

export function restoreWallSettings(value: unknown): readonly WallSettings[] {
  if (!Array.isArray(value) || value.length > 8192) throw new Error('Invalid wall settings list.');
  const seen = new Set<string>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item) || typeof item.upper !== 'boolean'
      || ('record' in item) === ('boundary' in item) || Object.keys(item).some(key => !['record', 'boundary', 'upper', 'brightness', 'horizontalOffset', 'verticalOffset', 'repeat'].includes(key))) throw new Error('Wall settings need one face or boundary identity and a storey.');
    const identity = 'record' in item ? { record: checkedInteger(item.record, 0, 16777215, 'Wall settings record') } : { boundary: checkedInteger(item.boundary, 0, 16777215, 'Wall settings boundary') };
    const fields = Object.fromEntries(['brightness', 'horizontalOffset', 'verticalOffset'].filter(key => key in item).map(key => [key, checkedInteger(item[key], -32768, 32767, `Wall ${key}`)]));
    const repeat = 'repeat' in item ? restoreWallRepeat(item.repeat) : undefined;
    if (!Object.keys(fields).length && !repeat) throw new Error('Wall settings need an alignment, repeat or brightness override.');
    const key = `${item.upper}:${'record' in identity ? 'record:' + identity.record : 'boundary:' + identity.boundary}`;
    if (seen.has(key)) throw new Error('A wall has duplicate settings overrides.');
    seen.add(key); return Object.freeze({ ...identity, upper: item.upper, ...fields, ...(repeat ? { repeat } : {}) });
  }));
}

export function wallSettingsCurrent(document: LevelDocument, target: SurfaceMaterialTarget): WallSettings | undefined {
  const room = document.roomSource.rooms.find(room => room.id === target.room);
  return room?.wallSettings?.find(item => item.upper === target.upper && ('record' in target
    ? 'record' in item && item.record === target.record : 'boundary' in item && item.boundary === room.boundaryMaterials?.edges[target.index]));
}

/** DEFLOWERWALLBRIGHT / DEFUPPERWALLBRIGHT and native WallRoutine's
 * FROMTILE/totalyoff. Keep these independent of the selected material sheet. */
export function setWallSettings(document: LevelDocument, target: SurfaceMaterialTarget, value: WallSettingsValues | null): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === target.room);
  if (!room) throw new Error('The wall settings room no longer exists.');
  if ('index' in target) checkedInteger(target.index, 0, room.vertices.length - 1, 'Wall settings edge');
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const wall = 'record' in target ? (target.upper ? template.upper : template.lower).find(record => record.offset === target.record)?.wall : surfaceMaterialCurrent(document, target);
  if (!wall) throw new Error('The wall no longer has a straight source face. Restore it or choose another wall.');
  if (wall.texture === 11 || 'record' in target && activeNativeSwitches(document).some(item => item.zone >= 0 && item.graphic - 10 === target.record)) throw new Error('Switch panels keep their native control graphics. Select the wall behind the panel instead.');
  const previous = wallSettingsCurrent(document, target);
  let boundaryMaterials = room.boundaryMaterials;
  if (value && 'index' in target) boundaryMaterials ??= restoreBoundaryMaterials({ nextId: room.vertices.length, edges: room.vertices.map((_, index) => index), overrides: [] }, room.vertices.length);
  const identity = 'record' in target ? { record: target.record } : { boundary: boundaryMaterials?.edges[target.index] };
  const remaining = (room.wallSettings ?? []).filter(item => item !== previous);
  const settings = restoreWallSettings(value ? [...remaining, { ...identity, upper: target.upper, ...value }] : remaining);
  if (JSON.stringify(room.wallSettings ?? []) === JSON.stringify(settings)) return { label: 'Change wall settings', before, after: before };
  const { wallSettings: _previous, ...retained } = room;
  const updated = Object.freeze({ ...retained, ...(boundaryMaterials ? { boundaryMaterials } : {}), ...(settings.length ? { wallSettings: settings } : {}) });
  return { label: value ? 'Change wall settings' : 'Reset wall settings', before,
    after: Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === room.id ? updated : item)) }) };
}

export function clearRoomWallSettings(document: LevelDocument, id: string): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select an existing room to reset room wall settings.');
  if (!room.wallSettings?.length) return { label: 'Reset room wall settings', before, after: before };
  const { wallSettings: _previous, ...retained } = room;
  return { label: 'Reset room wall settings', before, after: Object.freeze({ ...before,
    rooms: Object.freeze(before.rooms.map(item => item.id === id ? Object.freeze(retained) : item)) }) };
}
