import { checkedInteger } from './binary';
import type { NewRoomGraphics, ClosingRoomGraphics } from './cellGraphics';
import type { LevelDocument } from './level';
import type { PointLightingBinding } from './pointLighting';
import type { SourceRoom } from './roomSource';
import { roomSettings } from './roomHeights';
import { hasNativeWallMaterials } from './roomMaterials';

export interface RoomAppearance { readonly room: string; readonly lowerWall: number; readonly upperWall: number | null; readonly lightingPoint: number; readonly surfaces?: RoomSurfaceDefaults }
export interface RoomSurfaceDefaults { readonly lowerFloor: number; readonly lowerCeiling: number; readonly upperFloor: number | null; readonly upperCeiling: number | null }

export function roomAppearanceChoices(document: LevelDocument, room: Pick<SourceRoom, 'recordZone' | 'heights' | 'upperStorey'>) {
  const settings = roomSettings(document, room);
  const template = room.upperStorey ? { ...settings, upper: [] } : settings;
  const walls = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).filter(record => record.wall);
  const lower = template.lower.filter(record => record.wall), upper = template.upper.filter(record => record.wall);
  const points = [...new Set([...template.lower, ...template.upper].flatMap(record => record.points))];
  return { template, lower: lower.length ? lower : walls, upper: template.upper.length ? upper.length ? upper : walls : [],
    points: points.length ? points : document.points.map(point => point.id) };
}

function restoreSurfaceDefaults(value: unknown): RoomSurfaceDefaults {
  if (!value || typeof value !== 'object' || Object.keys(value).length !== 4 || !['lowerFloor', 'lowerCeiling', 'upperFloor', 'upperCeiling'].every(key => key in value)) throw new Error('Default room surfaces need lower and upper floor/ceiling references.');
  const item = value as Record<string, unknown>;
  if ((item.upperFloor === null) !== (item.upperCeiling === null)) throw new Error('Default upper surfaces need both a floor and ceiling.');
  return Object.freeze(Object.fromEntries(Object.entries(item).map(([key, reference]) => [key, key.startsWith('upper') && reference === null ? null : checkedInteger(reference as number, 0, 16777215, 'Default surface reference')])) as unknown as RoomSurfaceDefaults);
}

/** A new room receives real, saved materials at creation. Prefer its selected
 * template, with real level materials supplying missing wall/plane styles.
 * Imported decorations and special graphics do not become its new outline. */
export function defaultRoomAppearance(document: LevelDocument, room: SourceRoom): RoomAppearance {
  const choices = roomAppearanceChoices(document, room), all = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]);
  const normal = (records: typeof all, type: 1 | 2) => records.find(record => record.type === type && record.points.length >= 3)
    ?? all.find(record => record.type === type && record.points.length >= 3);
  const lowerWall = choices.lower[0], upperWall = choices.template.upper.length ? choices.upper[0] : undefined;
  const lowerFloor = normal(choices.template.lower, 1), lowerCeiling = normal(choices.template.lower, 2);
  const upperFloor = choices.template.upper.length ? normal(choices.template.upper, 1) : undefined;
  const upperCeiling = choices.template.upper.length ? normal(choices.template.upper, 2) : undefined;
  if (!lowerWall || choices.template.upper.length && !upperWall || !lowerFloor || !lowerCeiling || choices.template.upper.length && (!upperFloor || !upperCeiling) || choices.points[0] === undefined) throw new Error('This level has no complete set of wall, floor, ceiling and lighting materials for a new room. Import the missing materials before drawing.');
  return Object.freeze({ room: room.id, lowerWall: lowerWall.offset, upperWall: upperWall?.offset ?? null, lightingPoint: choices.points[0], surfaces: Object.freeze({ lowerFloor: lowerFloor.offset, lowerCeiling: lowerCeiling.offset, upperFloor: upperFloor?.offset ?? null, upperCeiling: upperCeiling?.offset ?? null }) });
}

export function restoreRoomAppearances(value: unknown): readonly RoomAppearance[] {
  if (!Array.isArray(value) || value.length > 256) throw new Error('The room appearance list exceeds the room limit.');
  const rooms = new Set<string>();
  return Object.freeze(value.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('room' in item) || typeof item.room !== 'string' || !item.room || item.room.length > 128
      || !('lowerWall' in item) || typeof item.lowerWall !== 'number' || !('upperWall' in item) || (item.upperWall !== null && typeof item.upperWall !== 'number')
      || !('lightingPoint' in item) || typeof item.lightingPoint !== 'number'
      || Object.keys(item).some(key => !['room', 'lowerWall', 'upperWall', 'lightingPoint', 'surfaces'].includes(key))) throw new Error('Invalid room appearance settings.');
    if (rooms.has(item.room)) throw new Error(`Duplicate appearance settings for room ${item.room}.`);
    rooms.add(item.room);
    return Object.freeze({ room: item.room, lowerWall: checkedInteger(item.lowerWall, 0, 16777215, 'Lower wall style'),
      upperWall: item.upperWall === null ? null : checkedInteger(item.upperWall, 0, 16777215, 'Upper wall style'),
      lightingPoint: checkedInteger(item.lightingPoint, 0, 2047, 'Template lighting point'), ...('surfaces' in item ? { surfaces: restoreSurfaceDefaults(item.surfaces) } : {}) });
  }));
}

export function validateRoomAppearances(document: LevelDocument, value: unknown): readonly RoomAppearance[] {
  const appearances = restoreRoomAppearances(value);
  for (const appearance of appearances) {
    const room = document.roomSource.rooms.find(room => room.id === appearance.room);
    if (!room) continue; // Deleted source volumes retain their settings for undo.
    const choices = roomAppearanceChoices(document, room);
    if (appearance.surfaces) for (const [field, offset] of Object.entries(appearance.surfaces)) {
      if (offset === null) continue;
      const type = field.endsWith('Floor') ? 1 : 2;
      if (!document.zones.some(zone => [...zone.lower, ...zone.upper].some(record => record.offset === offset && record.type === type && record.points.length >= 3))) throw new Error(`Room ${room.id} has a missing default ${field} material. Choose its room appearance again.`);
    }
    if (!choices.lower.some(record => record.offset === appearance.lowerWall)
      || (appearance.upperWall !== null && choices.template.upper.length > 0 && !choices.upper.some(record => record.offset === appearance.upperWall))
      || (choices.template.upper.length > 0 && appearance.upperWall === null)
      || !choices.points.includes(appearance.lightingPoint)) throw new Error(`Room ${room.id} appearance no longer matches its template. Choose its wall style and lighting again.`);
  }
  return appearances;
}

export function setRoomAppearance(document: LevelDocument, appearance: RoomAppearance): readonly RoomAppearance[] {
  if (!document.roomSource.rooms.some(room => room.id === appearance.room)) throw new Error('Choose an existing room before changing its appearance.');
  const previous = document.roomAppearances?.find(item => item.room === appearance.room);
  if (!appearance.surfaces && previous?.surfaces) appearance = { ...appearance, surfaces: previous.surfaces };
  const result = validateRoomAppearances(document, [...(document.roomAppearances ?? []).filter(item => item.room !== appearance.room), appearance]);
  return previous && JSON.stringify(previous) === JSON.stringify(appearance) ? document.roomAppearances! : result;
}

/** Wall style is saved authoring state, initialized automatically for new rooms.
 * Its vertical span comes from the
 * selected room's actual floor/ceiling; phase, texture and sampling remain
 * those of the chosen source wall. Preserve its position in the draw stream. */
export function roomAppearanceBuildOptions(document: LevelDocument) {
  const appearances = validateRoomAppearances(document, document.roomAppearances ?? []);
  const newRoomGraphics = new Map<string, NewRoomGraphics>(), newRoomLighting = new Map<string, PointLightingBinding>();
  const closingRoomGraphics = new Map<string, ClosingRoomGraphics>();
  for (const room of document.roomSource.rooms) {
    const selected = appearances.find(item => item.room === room.id);
    if (!selected && hasNativeWallMaterials(room.originZone, room.wallSources)) continue;
    if (!selected) throw new Error(`Room ${room.id.replace(/^room-/, '')} needs an appearance. Select it and apply its wall style and lighting before building.`);
    const { template, lower, upper } = roomAppearanceChoices(document, room);
    const lowerRecord = lower.find(record => record.offset === selected.lowerWall)!;
    const upperRecord = upper.find(record => record.offset === selected.upperWall);
    const graphics: NewRoomGraphics = { walls: {
      // A style supplies sampling and texture, not the connection's mode.
      // Authored barriers explicitly emit 13 after normal perimeter assembly.
      lower: [{ ...lowerRecord.wall!, type: 0, topFixed: template.ceiling * 256, bottomFixed: template.floor * 256 }],
      upper: upperRecord ? [{ ...upperRecord.wall!, type: 0, topFixed: template.upperCeiling * 256, bottomFixed: template.upperFloor * 256 }] : [],
    }, wallInsertion: { lower: selected.surfaces ? 0 : Math.max(0, template.lower.indexOf(lowerRecord)), ...(upperRecord ? { upper: selected.surfaces ? 0 : Math.max(0, template.upper.indexOf(upperRecord)) } : {}) } };
    newRoomLighting.set(room.id, { kind: 'copy', source: selected.lightingPoint });
    if (hasNativeWallMaterials(room.originZone, room.wallSources)) closingRoomGraphics.set(room.id, { ...graphics, lightingPoint: selected.lightingPoint });
    else newRoomGraphics.set(room.id, graphics);
  }
  return { newRoomGraphics, newRoomLighting, closingRoomGraphics };
}
