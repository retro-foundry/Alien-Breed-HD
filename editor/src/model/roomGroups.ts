import { translateSourceRooms } from './roomSource';
import { editableStarts, placeSourceStart } from './sourceStarts';
import { editableObjects, objectIsActive, objectPlayer, placeSourceObject } from './sourceObjects';
import { editableTeleport, placeSourceTeleport } from './sourceTeleports';
import type { LevelDocument } from './level';
import type { EditCommand } from './commands';

/** Geometry and every moved room's positional dependencies form one command.
 * Work only on immutable provisional documents: any failure rejects it all. */
export function moveRoomGroup(document: LevelDocument, ids: readonly string[], x: number, z: number): EditCommand {
  const change = translateSourceRooms(document.roomSource, ids, x, z);
  if (change.after === change.before) return { label: change.label, patches: [] };
  const selected = new Set(ids);
  const roomForOrigin = (zone: number) => document.roomSource.rooms.find(room => room.originZone === zone)?.id;
  let working = { ...document, roomSource: change.after };
  for (const start of editableStarts(document)) {
    const room = start.room ?? roomForOrigin(start.zone);
    if (room && selected.has(room)) working = { ...working, sourceStarts: placeSourceStart(working, start.player, room, start.x + x, start.z + z) };
  }
  for (const object of editableObjects(document)) {
    const room = object.room ?? roomForOrigin(object.zone);
    if (room && selected.has(room) && objectIsActive(object) && objectPlayer(document, object) === undefined)
      working = { ...working, sourceObjects: placeSourceObject(working, object.id, room, object.x + x, object.z + z) };
  }
  for (const room of document.roomSource.rooms) {
    const link = editableTeleport(document, room);
    if (link.destination && selected.has(link.destination))
      working = { ...working, sourceTeleports: placeSourceTeleport(working, room.id, link.destination, link.x + x, link.z + z) };
  }
  return { label: change.label, patches: [], roomSource: { before: document.roomSource, after: working.roomSource },
    ...(working.sourceStarts !== document.sourceStarts ? { sourceStarts: { before: document.sourceStarts, after: working.sourceStarts } } : {}),
    ...(working.sourceObjects !== document.sourceObjects ? { sourceObjects: { before: document.sourceObjects, after: working.sourceObjects } } : {}),
    ...(working.sourceTeleports !== document.sourceTeleports ? { sourceTeleports: { before: document.sourceTeleports, after: working.sourceTeleports } } : {}) };
}
