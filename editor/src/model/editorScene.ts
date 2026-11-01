import { convexParts, pointInRoomOutline } from './geometry';
import { objectName, type LevelDocument } from './level';
import { roomSettings } from './roomHeights';
import { objectTemplates } from './objectAuthoring';
import { editableObjects, objectIsActive, objectDestinationRoom } from './sourceObjects';
import { editableStarts } from './sourceStarts';
import { editRoomSource, numberPatch, setSourceObject, setSourceStart, type EditCommand } from './commands';
import { moveSourceCorner, setRoomHeights } from './roomSource';
import { add, mul, rayTriangle, type Ray, type Vector3 } from './editorCamera';

export interface SceneFace { room: string; kind: 'floor' | 'ceiling' | 'wall'; points: readonly Vector3[] }
export interface SceneMarker { kind: 'object' | 'start'; id: number; room: string; position: Vector3; name: string }
export interface EditorScene { faces: readonly SceneFace[]; markers: readonly SceneMarker[] }
/** Editable source volumes, not a reconstruction of the game's decorative
 * graphics. Native roomSettings and actor Y remain the coordinate authority. */
export function editorScene(document: LevelDocument, upper: boolean): EditorScene {
  const faces: SceneFace[] = [], markers: SceneMarker[] = [];
  const rooms = document.roomSource.rooms.filter(room => !upper || roomSettings(document, room).upper.length > 0 && roomSettings(document, room).upperFloor > roomSettings(document, room).upperCeiling);
  for (const room of rooms) {
    const settings = roomSettings(document, room), floor = settings[upper ? 'upperFloor' : 'floor'], ceiling = settings[upper ? 'upperCeiling' : 'ceiling'];
    for (const part of convexParts(room.vertices)) for (let i = 1; i < part.length - 1; i++) {
      const triangle = [part[0]!, part[i]!, part[i + 1]!];
      faces.push({ room: room.id, kind: 'floor', points: triangle.map(p => ({ ...p, y: floor })) },
        { room: room.id, kind: 'ceiling', points: triangle.map(p => ({ ...p, y: ceiling })) });
    }
    room.vertices.forEach((point, i) => {
      const next = room.vertices[(i + 1) % room.vertices.length]!;
      faces.push({ room: room.id, kind: 'wall', points: [{ ...point, y: floor }, { ...next, y: floor }, { ...next, y: ceiling }, { ...point, y: ceiling }] });
    });
  }
  const eligible = new Set(objectTemplates(document).map(o => o.id));
  for (const actor of editableObjects(document)) {
    if (!eligible.has(actor.id) || !objectIsActive(actor) || actor.upper !== upper) continue;
    const room = rooms.find(room => actor.room ? room.id === actor.room : room.originZone === actor.zone);
    if (room) markers.push({ kind: 'object', id: actor.id, room: room.id, position: { x: actor.x, y: actor.y, z: actor.z }, name: `${objectName(actor.type)} ${actor.id}` });
  }
  if (!upper) for (const start of editableStarts(document)) {
    const authored = document.sourceStarts?.find(item => item.player === start.player);
    const room = rooms.find(room => authored ? room.id === authored.room : room.originZone === start.zone);
    if (room) markers.push({ kind: 'start', id: start.player, room: room.id,
      position: { x: authored?.x ?? start.x, y: roomSettings(document, room).floor, z: authored?.z ?? start.z }, name: `Player ${start.player} start` });
  }
  return { faces, markers };
}
export function pickSceneFace(faces: readonly SceneFace[], ray: Ray) {
  let hit: { face: SceneFace; position: Vector3; distance: number } | null = null;
  for (const face of faces) for (let i = 1; i < face.points.length - 1; i++) {
    const distance = rayTriangle(ray, face.points[0]!, face.points[i]!, face.points[i + 1]!);
    if (distance !== null && (!hit || distance < hit.distance)) hit = { face, position: add(ray.origin, mul(ray.direction, distance)), distance };
  }
  return hit;
}
export type SpatialEdit = { kind: 'marker'; marker: SceneMarker; position: Vector3 }
  | { kind: 'corner'; room: string; index: number; position: Vector3 }
  | { kind: 'height'; room: string; field: 'floor' | 'ceiling' | 'upperFloor' | 'upperCeiling'; value: number };
/** One gesture produces one ordinary editor command. Validation occurs before
 * history changes, so a rejected drag cannot leave position and height split. */
export function spatialEdit(document: LevelDocument, change: SpatialEdit): EditCommand {
  if (change.kind === 'height') {
    const room = document.roomSource.rooms.find(r => r.id === change.room);
    if (!room) throw new Error('The selected room no longer exists. Select it again.');
    const { floor, ceiling, upperFloor, upperCeiling } = roomSettings(document, room);
    return editRoomSource(setRoomHeights(document, document.roomSource, room.id, { floor, ceiling, upperFloor, upperCeiling, [change.field]: change.value }));
  }
  if (change.kind === 'corner') return editRoomSource(moveSourceCorner(document.roomSource, change.room, change.index, { x: change.position.x, z: change.position.z }));
  const { marker, position } = change;
  if (marker.kind === 'start') {
    const room = document.roomSource.rooms.find(r => r.id === marker.room && pointInRoomOutline(r.vertices, position))
      ?? document.roomSource.rooms.find(r => pointInRoomOutline(r.vertices, position));
    if (!room) throw new Error('Place the player start inside a room.');
    return setSourceStart(document, marker.id as 1 | 2, room.id, position.x, position.z);
  }
  const object = editableObjects(document).find(o => o.id === marker.id);
  if (!object || !objectIsActive(object) || !objectTemplates(document).some(o => o.id === marker.id)) throw new Error('Select an editable object before moving it.');
  const command: EditCommand = { label: `Move ${marker.name}`, patches: [] };
  if (position.x !== object.x || position.z !== object.z) {
    const room = objectDestinationRoom(document, object, position);
    if (!room) throw new Error('Place the object inside a room on its floor.');
    Object.assign(command, setSourceObject(document, object.id, room.id, position.x, position.z), { label: `Move ${marker.name}` });
  }
  if (position.y !== object.y) command.patches.push(numberPatch(document, 'data', object.offset + 4, position.y * 2, 2));
  return command;
}
