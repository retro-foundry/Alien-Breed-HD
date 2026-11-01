import type { LevelDocument } from './level';
import type { EditCommand } from './commands';
import { editableObjects, objectIsActive, objectPlayer, placeSourceObject, runtimeObjectPosition } from './sourceObjects';
import { pointInRoomOutline } from './geometry';

export function selectedObjects(document: LevelDocument, ids: readonly number[]) {
  if (ids.length < 2 || new Set(ids).size !== ids.length) throw new Error('Select at least two different objects.');
  const objects = editableObjects(document);
  return ids.map(id => {
    const object = objects.find(object => object.id === id);
    if (!object || !objectIsActive(object) || objectPlayer(document, object) !== undefined)
      throw new Error(`Object ${id} cannot be grouped. Select active objects; player avatars are controlled through Starts.`);
    return object;
  });
}

export function editObjectGroup(document: LevelDocument, ids: readonly number[], operation:
  { kind: 'move'; x: number; z: number } | { kind: 'align'; axis: 'x' | 'z' } | { kind: 'delete' }): EditCommand {
  const objects = selectedObjects(document, ids);
  let working = document;
  for (const object of objects) {
    const position = runtimeObjectPosition(operation.kind === 'move' ? { x: object.x + operation.x, z: object.z + operation.z }
      : operation.kind === 'align' ? { x: operation.axis === 'x' ? objects[0]!.x : object.x, z: operation.axis === 'z' ? objects[0]!.z : object.z }
      : object);
    const currentRoom = object.room ?? document.roomSource.rooms.find(room => room.originZone === object.zone)?.id;
    const room = operation.kind === 'delete' ? null : document.roomSource.rooms.find(room => room.id === currentRoom && pointInRoomOutline(room.vertices, position))
      ?? document.roomSource.rooms.find(room => pointInRoomOutline(room.vertices, position));
    if (room === undefined) throw new Error(`Object ${object.id} would be outside the editable rooms. No group changes were applied.`);
    working = { ...working, sourceObjects: placeSourceObject(working, object.id, room === null ? null : room.id, position.x, position.z) };
  }
  return { label: `${operation.kind === 'move' ? 'Move' : operation.kind === 'align' ? `Align ${operation.axis.toUpperCase()}` : 'Delete'} ${objects.length} objects`, patches: [],
    sourceObjects: { before: document.sourceObjects, after: working.sourceObjects } };
}
