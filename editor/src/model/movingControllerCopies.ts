import type { LevelDocument } from './level';
import { resolveAnimationRoom } from './animationRooms';
import { nativeRoomTranslation } from './roomTranslation';
import { mechanismDependencies } from './mechanismDependencies';
import { BinaryReader } from './binary';
import { assertMechanismActivation } from './mechanismActivation';

/** Explicit source identities, never generated zone/line offsets. Graphics
 * ownership and collision incidence are independent native relationships. */
export interface MovingControllerCopy {
  kind: 'door' | 'lift';
  id: number;
  room: string;
  graphicsRooms: ReadonlyMap<number, string>;
  collisionRooms: ReadonlyMap<number, string>;
  /** Optional complete native settings header. Pointer/room fields are rebuilt. */
  header?: Uint8Array;
}

export function validateMovingControllerCopies(document: LevelDocument, copies: readonly MovingControllerCopy[]): void {
  const assigned = new Set<string>();
  for (const copy of copies) {
    const mechanism = document.mechanisms.find(item => item.kind === copy.kind && item.id === copy.id);
    if (!mechanism) throw new Error('A copied controller has no native template.');
    const dependencies = mechanismDependencies(document, copy.kind, copy.id);
    const controlled = document.roomSource.rooms.find(room => room.id === copy.room);
    if (!controlled || controlled.originZone !== null || controlled.recordZone !== resolveAnimationRoom(document, mechanism.zone, copy.kind === 'lift'))
      throw new Error('A copied controller needs a fresh room identity matching its controlled native template.');
    const identity = `${copy.kind}:${copy.room}`;
    if (assigned.has(identity)) throw new Error('A copied room has several controllers for the same moving plane.');
    assigned.add(identity);
    const translation = nativeRoomTranslation(document, controlled);
    if (!translation) throw new Error('Copied moving controllers require an exact whole-room translation.');
    const checkRoom = (id: string | undefined, template: number | readonly number[], subject: string) => {
      const room = document.roomSource.rooms.find(room => room.id === id);
      const allowed = typeof template === 'number' ? [template] : template;
      if (!room || room.originZone !== null || !allowed.includes(room.recordZone)) throw new Error(`${subject} needs an explicit copied room with the matching native attachment.`);
      const offset = nativeRoomTranslation(document, room);
      if (!offset || offset.x !== translation.x || offset.z !== translation.z) throw new Error(`${subject} must move with the copied controller's room group.`);
    };
    for (const room of dependencies.graphicsRooms) checkRoom(copy.graphicsRooms.get(room), room, `Moving graphics from Zone ${room}`);
    if (copy.graphicsRooms.get(dependencies.room) !== copy.room) throw new Error('The controller room and graphics-room mapping disagree.');
    for (const wall of dependencies.walls) checkRoom(copy.collisionRooms.get(wall.line), wall.collisionRooms, `Moving collision line ${wall.line}`);
    if (copy.header) {
      if (copy.header.length !== 18) throw new Error('Copied controller settings require a complete 18-byte native header.');
      const reader = new BinaryReader(copy.header, 'copied controller settings');
      const bottom = reader.i16(0) / 4, top = reader.i16(2) / 4, current = reader.i16(4) / 4;
      if (reader.i16(0) === 999 || top > bottom || current < top || current > bottom) throw new Error('Copied controller settings have an invalid native travel range.');
      if (current !== mechanism.current) throw new Error('Changing a copied controller initial position requires synchronized source geometry.');
      if (reader.u8(16) > (copy.kind === 'door' ? 5 : 3) || reader.u8(17) > (copy.kind === 'door' ? 1 : 3)) throw new Error('Copied controller settings contain unsupported activation rules.');
      assertMechanismActivation({ kind: copy.kind, direction: reader.i16(6), conditions: reader.u16(14), current, top, bottom });
    }
  }
}
