import { objectName, type LevelDocument } from './level';
import { numberPatch, type EditCommand } from './commands';
import { editableObjects, objectIsActive, objectPlayer, objectDestinationRoom, placeSourceObject, runtimeObjectPosition } from './sourceObjects';

export function objectSelectionIds(document: LevelDocument, search = '', upper?: boolean): number[] {
  const query = search.toLowerCase();
  return editableObjects(document).filter(object => objectIsActive(object) && objectPlayer(document, object) === undefined
    && (upper === undefined || object.upper === upper) && `${object.id} ${objectName(object.type)}`.toLowerCase().includes(query)).map(object => object.id);
}

export function objectsInRectangle(document: LevelDocument, start: { x: number; z: number }, end: { x: number; z: number }, upper: boolean): number[] {
  if (![start.x, start.z, end.x, end.z].every(Number.isFinite)) throw new Error('Selection rectangle coordinates must be finite.');
  const eligible = new Set(objectSelectionIds(document, '', upper));
  return editableObjects(document).filter(object => eligible.has(object.id)
    && object.x >= Math.min(start.x, end.x) && object.x <= Math.max(start.x, end.x)
    && object.z >= Math.min(start.z, end.z) && object.z <= Math.max(start.z, end.z)).map(object => object.id);
}

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
    const room = operation.kind === 'delete' ? null : objectDestinationRoom(document, object, position);
    if (room === undefined) throw new Error(`Object ${object.id} would be outside the editable rooms supporting its floor. No group changes were applied.`);
    working = { ...working, sourceObjects: placeSourceObject(working, object.id, room === null ? null : room.id, position.x, position.z) };
  }
  return { label: `${operation.kind === 'move' ? 'Move' : operation.kind === 'align' ? `Align ${operation.axis.toUpperCase()}` : 'Delete'} ${objects.length} objects`, patches: [],
    sourceObjects: { before: document.sourceObjects, after: working.sourceObjects } };
}

/** Renderer object Y words are half-world units. Validate every proposed word
 * before exposing one command, retaining each object's X/Z and storey binding. */
export function editObjectGroupHeight(document: LevelDocument, ids: readonly number[], offset: number | 'align'): EditCommand {
  const objects = selectedObjects(document, ids);
  if (offset !== 'align' && (!Number.isFinite(offset) || !Number.isInteger(offset * 2))) throw new Error('Object height offsets must use finite half-unit steps.');
  return { label: offset === 'align' ? `Align ${objects.length} object heights` : `Offset ${objects.length} object heights`, patches: objects.map(object => {
    const height = offset === 'align' ? objects[0]!.y : object.y + offset;
    if (height < -16384 || height > 16383.5) throw new Error(`Object ${object.id} would exceed the height range -16384 to 16383.5. No group heights were changed.`);
    return numberPatch(document, 'data', object.offset + 4, height * 2, 2);
  }) };
}
