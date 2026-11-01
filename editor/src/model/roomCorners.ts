import { insertSourceCorner, removeSourceCorner } from './roomSource';
import { editRoomSource } from './commands';
import { pointInRoomOutline, type Position } from './geometry';
import type { LevelDocument } from './level';
import { editableObjects, objectIsActive, objectPlayer } from './sourceObjects';
import { editableStarts } from './sourceStarts';
import { editableTeleport } from './sourceTeleports';

/** Whole-coordinate points on a slanted edge form an integer lattice. Snap
 * along that lattice, rather than off the wall onto the square display grid. */
export function edgeCornerPoint(a: Position, b: Position, near: Position = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 }): Position {
  const dx = b.x - a.x, dz = b.z - a.z;
  let divisor = Math.abs(dx), remainder = Math.abs(dz);
  while (remainder) { const next = divisor % remainder; divisor = remainder; remainder = next; }
  if (divisor < 2) throw new Error('This edge has no whole-coordinate position between its ends. Lengthen the edge before adding a corner.');
  const t = Math.max(1, Math.min(divisor - 1, Math.round(((near.x - a.x) * dx + (near.z - a.z) * dz) / (dx * dx + dz * dz) * divisor)));
  return { x: a.x + dx / divisor * t, z: a.z + dz / divisor * t };
}
export function addRoomCorner(document: LevelDocument, room: string, edge: number, near?: Position) {
  const target = document.roomSource.rooms.find(item => item.id === room);
  if (!target || !target.vertices[edge]) throw new Error('Select an existing room edge to split.');
  const a = target.vertices[edge]!, b = target.vertices[(edge + 1) % target.vertices.length]!;
  return editRoomSource(insertSourceCorner(document.roomSource, room, edge, edgeCornerPoint(a, b, near)));
}
export function deleteRoomCorner(document: LevelDocument, room: string, index: number) {
  const change = removeSourceCorner(document.roomSource, room, index);
  const affected = new Map(change.after.rooms.filter(r => r !== document.roomSource.rooms.find(old => old.id === r.id)).map(r => [r.id, r]));
  const check = (id: string | undefined | null, point: Position, subject: string) => {
    const target = id && affected.get(id);
    if (target && !pointInRoomOutline(target.vertices, point)) throw new Error(`${subject} would be outside room ${target.id.replace(/^room-/, '')}. Move it before removing this corner. No corners were removed.`);
  };
  const roomFor = (zone: number) => document.roomSource.rooms.find(r => r.originZone === zone)?.id;
  for (const start of editableStarts(document)) check(start.room ?? roomFor(start.zone), start, `Player ${start.player} start`);
  for (const object of editableObjects(document)) if (objectIsActive(object) && objectPlayer(document, object) === undefined) check(object.room ?? roomFor(object.zone), object, `Object ${object.id}`);
  for (const origin of document.roomSource.rooms) { const link = editableTeleport(document, origin); check(link.destination, link, 'Teleport destination'); }
  return editRoomSource(change);
}
