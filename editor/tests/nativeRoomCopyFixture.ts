import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { applyCommand, editRoomAppearance } from '../src/model/commands';
import { roomAppearanceChoices } from '../src/model/roomAppearance';

/** Compiler fixture only: the complete room clipboard/controller authoring UI
 * remains separate work. Preserve wall provenance with a fresh source ID. */
export function nativeRoomCopyFixture(zone = 0) {
  let document = openLevel('Native material copy', originalFiles(1));
  const original = document.roomSource.rooms.find(room => room.originZone === zone)!;
  const dx = 20000 - original.vertices[0]!.x, dz = 20000 - original.vertices[0]!.z;
  const copy = { ...original, id: `room-${document.roomSource.nextId}`, originZone: null,
    vertices: original.vertices.map(point => ({ x: point.x + dx, z: point.z + dz })) };
  document = { ...document, roomSource: { rooms: [...document.roomSource.rooms, copy], nextId: document.roomSource.nextId + 1 } };
  // A one-sided copied opening needs an explicit closure style. Choose real
  // template records, just as the appearance inspector does.
  const choices = roomAppearanceChoices(document, copy);
  document = applyCommand(document, editRoomAppearance(document, { room: copy.id, lowerWall: choices.lower[0]!.offset,
    upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! }));
  return { document, original, copy, dx, dz };
}
