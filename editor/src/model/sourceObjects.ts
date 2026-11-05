import { roomSettings } from './roomHeights';
import { BinaryReader, checkedInteger } from './binary';
import { pointInRoomOutline } from './geometry';
import type { LevelDocument, LevelObject } from './level';

/** Existing object identities retain their original record/point slots. A null
 * room explicitly removes an object; a missing room ID remains a repairable
 * dependency instead of silently deleting the enemy or pickup. */
export interface SourceObject { readonly id: number; readonly room: string | null; readonly x: number; readonly z: number }
export type EditableObject = LevelObject & { room?: string | null };

/** Accept normal decimal input and round only for the engine's 16.16 point
 * table, matching commands.moveObject. Authors do not enter fixed-point units. */
export function runtimeObjectPosition(position: { x: number; z: number }): { x: number; z: number } {
  const result = { x: 0, z: 0 };
  for (const axis of ['x', 'z'] as const) {
    if (!Number.isFinite(position[axis])) throw new Error(`Object ${axis.toUpperCase()} must be a finite coordinate.`);
    result[axis] = checkedInteger(Math.round(position[axis] * 65536), -2147483648, 2147483647, `Object ${axis.toUpperCase()} coordinate`) / 65536;
  }
  return result;
}

export function objectPlayer(document: LevelDocument, object: Pick<LevelObject, 'offset'>): 1 | 2 | undefined {
  const bytes = new BinaryReader(document.files.data, 'player avatars');
  return object.offset === bytes.i32(46) ? 1 : object.offset === bytes.i32(50) ? 2 : undefined;
}

export function editableObjects(document: LevelDocument): EditableObject[] {
  return document.objects.map(object => {
    const player = objectPlayer(document, object);
    const authored = document.sourceObjects?.find(item => item.id === object.id)
      ?? (player === undefined ? undefined : document.sourceStarts?.find(start => start.player === player));
    return authored ? { ...object, x: authored.x, z: authored.z, room: authored.room,
      zone: document.roomSource.rooms.find(room => room.id === authored.room)?.originZone ?? -1 } : object;
  });
}

export function objectIsActive(object: EditableObject): boolean {
  return object.room === undefined ? object.zone >= 0 : object.room !== null;
}

/** Shared boundary points can belong to several outlines. Retain the current
 * room when possible, then consider only rooms supporting this object's storey.
 * Explicit room selections still pass through placeSourceObject unchanged. */
export function objectDestinationRoom(document: LevelDocument, object: EditableObject, position: { x: number; z: number }) {
  const current = object.room ?? document.roomSource.rooms.find(room => room.originZone === object.zone)?.id;
  const contains = (room: LevelDocument['roomSource']['rooms'][number]) => pointInRoomOutline(room.vertices, position)
    && (!object.upper || !!roomSettings(document, room).upper.length);
  return document.roomSource.rooms.find(room => room.id === current && contains(room)) ?? document.roomSource.rooms.find(contains);
}

export function restoreSourceObjects(value: unknown): readonly SourceObject[] {
  if (!Array.isArray(value) || value.length > 256) throw new Error('The editable object list exceeds the object-table limit.');
  const ids = new Set<number>();
  return Object.freeze(value.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('id' in item) || typeof item.id !== 'number'
      || !('room' in item) || (item.room !== null && (typeof item.room !== 'string' || !item.room || item.room.length > 128))
      || !('x' in item) || typeof item.x !== 'number' || !('z' in item) || typeof item.z !== 'number'
      || Object.keys(item).some(key => !['id', 'room', 'x', 'z'].includes(key))) throw new Error('Invalid editable object placement.');
    const id = checkedInteger(item.id, 0, 255, 'Editable object identity');
    if (ids.has(id)) throw new Error(`Duplicate editable object ${id}.`);
    ids.add(id);
    runtimeObjectPosition({ x: item.x, z: item.z });
    return Object.freeze({ id, room: item.room, x: item.x, z: item.z });
  }));
}

export function validateSourceObjects(document: LevelDocument, value: unknown): readonly SourceObject[] {
  const objects = restoreSourceObjects(value);
  for (const item of objects) {
    const original = document.objects.find(object => object.id === item.id);
    if (!original) throw new Error(`Editable object ${item.id} has no original record.`);
    if (objectPlayer(document, original) !== undefined) throw new Error('Player avatars follow their player starts. Select the player in Starts to move it.');
    const position = runtimeObjectPosition(item);
    if (item.room !== null && (position.x !== original.x || position.z !== original.z)
      && document.objects.some(other => other.id !== item.id && other.pointOffset === original.pointOffset))
      throw new Error(`Object ${item.id} shares a position slot with another object. Separate point identities are required before moving it.`);
  }
  return objects;
}

export function placeSourceObject(document: LevelDocument, id: number, room: string | null, x: number, z: number): readonly SourceObject[] {
  const objects = validateSourceObjects(document, [...(document.sourceObjects ?? []).filter(item => item.id !== id), { id, room, x, z }]);
  if (room !== null) {
    const target = document.roomSource.rooms.find(item => item.id === room);
    if (!target || !pointInRoomOutline(target.vertices, runtimeObjectPosition({ x, z }))) throw new Error('Place the object inside its selected editable room.');
    const object = document.objects.find(item => item.id === id)!;
    if (object.upper && !roomSettings(document, target).upper.length)
      throw new Error('This object is on the upper floor. Choose a room with an upper floor before moving it.');
  }
  if (document.sourceObjects?.some(item => item.id === id && item.room === room && item.x === x && item.z === z)) return document.sourceObjects;
  return objects;
}
