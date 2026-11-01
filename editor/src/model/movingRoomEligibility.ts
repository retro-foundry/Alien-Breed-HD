import type { LevelDocument } from './level';
import type { SourceRoom } from './roomSource';
import { mechanismDependencies } from './mechanismDependencies';

/** Keep existing native animation ownership intact when adding a controller.
 * Original room IDs are retained, so starts, actors and other room links do
 * not need a destructive imported-to-fresh identity conversion. */
export interface MovingRoomOwnership { reason: string; kind?: 'door' | 'lift' | 'switch'; room?: string }
export function movingRoomOwnership(document: LevelDocument, room: SourceRoom): MovingRoomOwnership | undefined {
  if (document.roomSource.controllerCopies?.some(copy => copy.room === room.id)) return { reason: 'This room already has a copied controller. Edit its existing settings below.' };
  if (room.originZone === null) return;
  for (const mechanism of document.mechanisms) {
    const dependency = mechanismDependencies(document, mechanism.kind, mechanism.id);
    const adjacent = document.zones.find(zone => zone.id === room.originZone)!.boundaryIds.some(id => document.lines[id]!.connect === dependency.room);
    if (adjacent && dependency.room !== room.originZone) return { reason: `This room adjoins an existing ${mechanism.kind}. Adjacent moving rooms need synchronized controller grouping.`, kind: mechanism.kind, room: document.roomSource.rooms.find(item => item.originZone === dependency.room)?.id };
    if (dependency.graphicsRooms.includes(room.originZone) || dependency.walls.some(wall => wall.collisionRooms.includes(room.originZone!)))
      return { reason: `An existing ${mechanism.kind} owns movement or wall attachments in this room. Edit that mechanism instead of adding another controller.`, kind: mechanism.kind, room: document.roomSource.rooms.find(item => item.originZone === dependency.room)?.id };
  }
  const zone = document.zones.find(zone => zone.id === room.originZone)!;
  const graphics = new Set([...zone.lower, ...zone.upper].map(record => record.offset));
  const linkedSwitch = document.switches.find(item => item.zone >= 0 && (item.zone === room.originZone || graphics.has(item.graphic - 10)));
  if (linkedSwitch) return { reason: 'This room contains an existing switch attachment. Reassign that switch before converting the room.', kind: 'switch', room: document.roomSource.rooms.find(item => item.originZone === linkedSwitch.zone)?.id };
}
export function movingRoomConflict(document: LevelDocument, room: SourceRoom): string | undefined {
  return movingRoomOwnership(document, room)?.reason;
}
