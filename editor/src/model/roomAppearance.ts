import { checkedInteger } from './binary';
import type { NewRoomGraphics, ClosingRoomGraphics } from './cellGraphics';
import type { LevelDocument } from './level';
import type { PointLightingBinding } from './pointLighting';
import type { SourceRoom } from './roomSource';
import { roomSettings } from './roomHeights';
import { hasNativeWallMaterials } from './roomMaterials';

export interface RoomAppearance { readonly room: string; readonly lowerWall: number; readonly upperWall: number | null; readonly lightingPoint: number }

export function roomAppearanceChoices(document: LevelDocument, room: Pick<SourceRoom, 'recordZone' | 'heights'>) {
  const template = roomSettings(document, room);
  return { template, lower: template.lower.filter(record => record.wall), upper: template.upper.filter(record => record.wall),
    points: [...new Set([...template.lower, ...template.upper].flatMap(record => record.points))] };
}

export function restoreRoomAppearances(value: unknown): readonly RoomAppearance[] {
  if (!Array.isArray(value) || value.length > 256) throw new Error('The room appearance list exceeds the room limit.');
  const rooms = new Set<string>();
  return Object.freeze(value.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('room' in item) || typeof item.room !== 'string' || !item.room || item.room.length > 128
      || !('lowerWall' in item) || typeof item.lowerWall !== 'number' || !('upperWall' in item) || (item.upperWall !== null && typeof item.upperWall !== 'number')
      || !('lightingPoint' in item) || typeof item.lightingPoint !== 'number'
      || Object.keys(item).some(key => !['room', 'lowerWall', 'upperWall', 'lightingPoint'].includes(key))) throw new Error('Invalid room appearance settings.');
    if (rooms.has(item.room)) throw new Error(`Duplicate appearance settings for room ${item.room}.`);
    rooms.add(item.room);
    return Object.freeze({ room: item.room, lowerWall: checkedInteger(item.lowerWall, 0, 16777215, 'Lower wall style'),
      upperWall: item.upperWall === null ? null : checkedInteger(item.upperWall, 0, 16777215, 'Upper wall style'),
      lightingPoint: checkedInteger(item.lightingPoint, 0, 2047, 'Template lighting point') });
  }));
}

export function validateRoomAppearances(document: LevelDocument, value: unknown): readonly RoomAppearance[] {
  const appearances = restoreRoomAppearances(value);
  for (const appearance of appearances) {
    const room = document.roomSource.rooms.find(room => room.id === appearance.room);
    if (!room) continue; // Deleted source volumes retain their settings for undo.
    const choices = roomAppearanceChoices(document, room);
    if (!choices.lower.some(record => record.offset === appearance.lowerWall)
      || (appearance.upperWall !== null && !choices.upper.some(record => record.offset === appearance.upperWall))
      || (choices.template.upper.length > 0 && appearance.upperWall === null)
      || !choices.points.includes(appearance.lightingPoint)) throw new Error(`Room ${room.id} appearance no longer matches its template. Choose its wall style and lighting again.`);
  }
  return appearances;
}

export function setRoomAppearance(document: LevelDocument, appearance: RoomAppearance): readonly RoomAppearance[] {
  if (!document.roomSource.rooms.some(room => room.id === appearance.room)) throw new Error('Choose an existing room before changing its appearance.');
  const result = validateRoomAppearances(document, [...(document.roomAppearances ?? []).filter(item => item.room !== appearance.room), appearance]);
  const previous = document.roomAppearances?.find(item => item.room === appearance.room);
  return previous && JSON.stringify(previous) === JSON.stringify(appearance) ? document.roomAppearances! : result;
}

/** Wall style is an explicit author choice. Its vertical span comes from the
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
      lower: [{ ...lowerRecord.wall!, topFixed: template.ceiling * 256, bottomFixed: template.floor * 256 }],
      upper: upperRecord ? [{ ...upperRecord.wall!, topFixed: template.upperCeiling * 256, bottomFixed: template.upperFloor * 256 }] : [],
    }, wallInsertion: { lower: template.lower.indexOf(lowerRecord), ...(upperRecord ? { upper: template.upper.indexOf(upperRecord) } : {}) } };
    newRoomLighting.set(room.id, { kind: 'copy', source: selected.lightingPoint });
    if (hasNativeWallMaterials(room.originZone, room.wallSources)) closingRoomGraphics.set(room.id, { ...graphics, lightingPoint: selected.lightingPoint });
    else newRoomGraphics.set(room.id, graphics);
  }
  return { newRoomGraphics, newRoomLighting, closingRoomGraphics };
}
