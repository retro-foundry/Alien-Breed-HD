/* AMOS DEFWATERANIM/LEVELSAVE and Anims.s DoWaterAnims: 21 shared
 * animation slots, 24.8 heights, an initial signed velocity, then fixed
 * +/-128 velocity at each reversal. Lower-room water is independent of floors. */
import { BinaryReader, checkedInteger } from './binary';
import { importedAnimationTables } from './animationBuild';
import { resolveAnimationRoom } from './animationRooms';
import type { LevelDocument, Graphic } from './level';
import type { SourceRoom } from './roomSource';
import type { EditCommand } from './commands';
import { roomSettings } from './roomHeights';
import { hasNativeWallMaterials } from './roomMaterials';

export interface SourceWaterGroup { readonly id: number; readonly top: number; readonly bottom: number; readonly current: number; readonly velocity: number; readonly enabled: boolean }
export interface SourceWaterSurface { readonly material: number | null; readonly height: number; readonly group: number | null }
const height = (value: unknown, label: string) => {
  if (typeof value !== 'number') throw new Error(`${label} must be a number.`);
  return checkedInteger(value * 4, -32768, 32767, label) / 4;
};
const animatedHeight = (value: unknown, label: string) => {
  if (typeof value !== 'number') throw new Error(`${label} must be a number.`);
  return checkedInteger(value * 256, -2097152, 2097151, label) / 256;
};

export function restoreWaterSurface(value: unknown): SourceWaterSurface | false {
  if (value === false) return false;
  if (!value || typeof value !== 'object' || Object.keys(value).length !== 3 || !('material' in value) || !('height' in value) || !('group' in value)) throw new Error('Water needs its surface material, height and animation group.');
  return Object.freeze({ material: value.material === null ? null : checkedInteger(value.material as number, 0, 16777215, 'Water material reference'), height: height(value.height, 'Water height in quarter units'), group: value.group === null ? null : checkedInteger(value.group as number, 0, 20, 'Water animation group') });
}
export function restoreWaterGroups(value: unknown): readonly SourceWaterGroup[] {
  if (!Array.isArray(value) || value.length > 21) throw new Error('The native engine has 21 water animation groups.');
  const ids = new Set<number>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).length !== 6 || !['id', 'top', 'bottom', 'current', 'velocity', 'enabled'].every(key => key in item) || typeof item.enabled !== 'boolean') throw new Error('A water animation group needs its range, initial height/direction and enabled setting.');
    const id = checkedInteger(item.id, 0, 20, 'Water animation group'); if (ids.has(id)) throw new Error('Duplicate water animation group.'); ids.add(id);
    const top = animatedHeight(item.top, 'Top water height'), bottom = animatedHeight(item.bottom, 'Bottom water height'), current = animatedHeight(item.current, 'Initial water height');
    if (top > bottom || current < top || current > bottom) throw new Error('The top water height must be no greater than the bottom; the initial height must be inside that range. Smaller heights are higher.');
    return Object.freeze({ id, top, bottom, current, velocity: checkedInteger(item.velocity, -32768, 32767, 'Initial water velocity'), enabled: item.enabled });
  }));
}

export function waterGroupSettings(document: LevelDocument, id: number): SourceWaterGroup {
  checkedInteger(id, 0, 20, 'Water animation group');
  const authored = document.roomSource.waterGroups?.find(group => group.id === id);
  if (authored) return authored;
  const group = importedAnimationTables(document).tables.water[id];
  // AMOS initializes unused slot bounds to zero and saves initial velocity
  // +128. These values describe an unused slot; assignment requires explicit
  // configuration or an existing native group with live entries below.
  if (!group) return { id, top: 0, bottom: 0, current: 0, velocity: 128, enabled: true };
  const reader = new BinaryReader(group.header, 'water animation header');
  return { id, top: reader.i32(0) / 256, bottom: reader.i32(4) / 256, current: reader.i32(8) / 256, velocity: reader.i16(12), enabled: true };
}

export function inheritedWaterGroups(document: LevelDocument, room: SourceRoom): number[] {
  if (!hasNativeWallMaterials(room.originZone, room.wallSources)) return [];
  const original = room.originZone ?? room.recordZone;
  return importedAnimationTables(document).tables.water.flatMap((group, id) => group.entries.some(entry => resolveAnimationRoom(document, entry.zone, true) === original) ? [id] : []);
}
export function roomWaterGroup(document: LevelDocument, room: SourceRoom): number | null {
  if (room.waterSurface !== undefined) return room.waterSurface === false ? null : room.waterSurface.group;
  const groups = inheritedWaterGroups(document, room);
  if (groups.length > 1) throw new Error('This imported room belongs to several water animations. Choose one explicit water group to replace the conflicting assignments.');
  return groups[0] ?? null;
}
export function waterMaterialChoices(document: LevelDocument, room: SourceRoom): Graphic[] {
  const own = document.zones.find(zone => zone.id === room.recordZone)!;
  const choices = new Map<string, Graphic>();
  for (const record of [...own.lower, ...document.zones.flatMap(zone => [...zone.lower, ...zone.upper])])
    if ([7, 1, 2].includes(record.type) && record.polygon && record.points.length >= 3) {
      const key = `${record.type}:${record.texture}`; if (!choices.has(key)) choices.set(key, record);
    }
  return [...choices.values()];
}

export function assertWaterSurface(document: LevelDocument, room: SourceRoom, surface: SourceWaterSurface): void {
  const material = surface.material === null ? undefined : document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === surface.material);
  if (surface.material === null ? !document.zones.find(zone => zone.id === room.recordZone)!.lower.some(record => record.type === 7) : !material?.polygon || ![1, 2, 7].includes(material.type) || material.points.length < 3) throw new Error(`Water in ${room.id} needs an existing water shape or a valid water, floor or ceiling pattern for whole-room water.`);
  if (surface.group !== null && !document.roomSource.waterGroups?.some(group => group.id === surface.group) && !importedAnimationTables(document).tables.water[surface.group]?.entries.length) throw new Error(`Create or configure water group ${surface.group + 1} before assigning or building water in ${room.id}.`);
}

export function setRoomWater(document: LevelDocument, id: string, value: SourceWaterSurface | false | null): EditCommand {
  const room = document.roomSource.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select an existing room before editing water.');
  const { waterSurface: _previous, ...unchanged } = room;
  const waterSurface = value === null ? undefined : restoreWaterSurface(value);
  if (waterSurface) assertWaterSurface(document, room, waterSurface);
  const updated = Object.freeze({ ...unchanged, ...(waterSurface !== undefined ? { waterSurface } : {}) });
  if (JSON.stringify(updated) === JSON.stringify(room)) return { label: 'Change room water', patches: [] };
  return { label: value === false ? 'Remove room water' : value === null ? 'Inherit template water' : 'Change room water', patches: [], roomSource: { before: document.roomSource,
    after: Object.freeze({ ...document.roomSource, rooms: Object.freeze(document.roomSource.rooms.map(item => item.id === id ? updated : item)) }) } };
}
export function setWaterGroup(document: LevelDocument, group: SourceWaterGroup): EditCommand {
  const waterGroups = restoreWaterGroups([...(document.roomSource.waterGroups ?? []).filter(item => item.id !== group.id), group]);
  if (JSON.stringify(waterGroups) === JSON.stringify(document.roomSource.waterGroups)) return { label: 'Change water animation group', patches: [] };
  return { label: `Change water animation group ${group.id + 1}`, patches: [], roomSource: { before: document.roomSource, after: Object.freeze({ ...document.roomSource, waterGroups }) } };
}
export function waterStartHeight(document: LevelDocument, room: SourceRoom): number {
  if (room.waterSurface === false) return roomSettings(document, room).floor + 4;
  const group = roomWaterGroup(document, room);
  return group === null ? room.waterSurface ? room.waterSurface.height : roomSettings(document, room).water : waterGroupSettings(document, group).current;
}
