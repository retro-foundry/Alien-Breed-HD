import type { LevelDocument } from './level';
import { resolveAnimationRoom } from './animationRooms';

export function mechanismsForRoom(document: LevelDocument, room: number) {
  return document.mechanisms.filter(item => resolveAnimationRoom(document, item.zone, item.kind === 'lift') === room);
}

export interface MechanismDependencies {
  room: number;
  surface: { graphic: number; room: number };
  walls: readonly { graphic: number; room: number; line: number; collisionRooms: readonly number[] }[];
  /** Rooms owning live graphics writes, including the controlled volume.
   * Collision-line incidence is separate: a shared boundary can have several
   * owning rooms and must not be guessed from the graphics owner. */
  graphicsRooms: readonly number[];
  keys: readonly number[];
  switches: readonly { number: number; room: number | null }[];
}

/** Anims.s DoorRoutine/LiftRoutine write the controlled zone plane and every
 * listed wall, and use floorline +14 for activation. Switch/key requirements
 * are bits in the shared Conditions word, not attachments to those walls. */
export function mechanismDependencies(document: LevelDocument, kind: 'door' | 'lift', id: number): MechanismDependencies {
  const mechanism = document.mechanisms.find(item => item.kind === kind && item.id === id);
  if (!mechanism) throw new Error('That door or lift no longer exists.');
  const room = resolveAnimationRoom(document, mechanism.zone, kind === 'lift');
  if (room === undefined) throw new Error(`${kind} ${id} has no controlled room.`);
  const graphics = new Map(document.zones.flatMap(zone => [...zone.lower, ...zone.upper].map(record => [record.offset, zone.id] as const)));
  const owner = (graphic: number) => {
    const room = graphics.get(graphic);
    if (room === undefined) throw new Error(`${kind} ${id} has a missing graphics attachment at ${graphic}.`);
    return room;
  };
  const surfaceGraphic = new DataView(document.files.graphics.buffer, document.files.graphics.byteOffset, document.files.graphics.byteLength).getInt32(mechanism.offset + 8);
  const surface = { graphic: surfaceGraphic, room: owner(surfaceGraphic) };
  const walls = mechanism.walls.map(wall => {
    const collisionRooms = document.zones.filter(zone => zone.boundaryIds.includes(wall.line)).map(zone => zone.id);
    if (!document.lines[wall.line] || !collisionRooms.length) throw new Error(`${kind} ${id} has an unowned collision attachment ${wall.line}.`);
    return { graphic: wall.graphic, room: owner(wall.graphic), line: wall.line, collisionRooms };
  });
  return { room, surface, walls,
    graphicsRooms: [...new Set([room, surface.room, ...walls.map(wall => wall.room)])].sort((a, b) => a - b),
    keys: Array.from({ length: 4 }, (_, bit) => bit + 1).filter(number => mechanism.conditions & (1 << (number - 1))),
    switches: Array.from({ length: 8 }, (_, bit) => bit + 1).filter(number => mechanism.conditions & (1 << (number + 3)))
      .map(number => ({ number, room: document.switches[number - 1]?.zone ?? null })) };
}
