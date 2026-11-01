import type { LevelDocument } from './level';
import { checkedInteger } from './binary';
import { nativeRoomTranslation } from './roomTranslation';

/** SwitchRoutine iterates eight fixed slots; each slot owns condition bit 4+n.
 * The graphics owner can differ from the activation room. */
export interface SwitchCopy {
  id: number;
  slot: number;
  room: string;
  graphicsRoom: string;
  bytes?: Uint8Array;
}

export function validateSwitchCopies(document: LevelDocument, copies: readonly SwitchCopy[]): void {
  const slots = new Set<number>();
  const walls = new Set<string>();
  for (const copy of copies) {
    checkedInteger(copy.id, 0, 7, 'Switch template slot'); checkedInteger(copy.slot, 0, 7, 'Copied switch slot');
    const original = document.switches[copy.id], destination = document.switches[copy.slot];
    if (!original || original.zone < 0) throw new Error('A copied switch needs an active native template.');
    const wall = `${original.graphic}:${copy.graphicsRoom}`;
    if (walls.has(wall)) throw new Error('Copied switches need independent graphics attachments.');
    walls.add(wall);
    if (!destination || destination.zone >= 0 || slots.has(copy.slot)) throw new Error('A copied switch needs its own unused native slot. The level supports eight switches.');
    slots.add(copy.slot);
    if (copy.bytes && copy.bytes.length !== 14) throw new Error('Copied switch settings need a complete 14-byte native record.');
    const room = document.roomSource.rooms.find(room => room.id === copy.room);
    const owner = document.zones.find(zone => [...zone.lower, ...zone.upper].some(record => record.offset === original.graphic - 10));
    const graphicsRoom = document.roomSource.rooms.find(room => room.id === copy.graphicsRoom);
    if (!room || room.originZone !== null || room.recordZone !== original.zone || !owner || !graphicsRoom || graphicsRoom.originZone !== null || graphicsRoom.recordZone !== owner.id)
      throw new Error('Copied switch activation and graphics need explicit fresh rooms matching their native owners.');
    const offset = nativeRoomTranslation(document, room), graphicsOffset = nativeRoomTranslation(document, graphicsRoom);
    if (!offset || !graphicsOffset || offset.x !== graphicsOffset.x || offset.z !== graphicsOffset.z) throw new Error('Copied switch graphics and activation rooms must share an exact whole-room translation.');
  }
}

/** Remap only copied internal switch conditions. Keys and external switch
 * requirements keep their original meanings; aliasing one would change a gate. */
export function remapSwitchConditions(conditions: number, slots: ReadonlyMap<number, number>): number {
  checkedInteger(conditions, 0, 65535, 'Native conditions');
  const destinations = new Set<number>();
  for (const [source, target] of slots) {
    checkedInteger(source, 0, 7, 'Source switch slot'); checkedInteger(target, 0, 7, 'Target switch slot');
    if (destinations.has(target)) throw new Error('Copied switches need distinct condition bits.');
    destinations.add(target);
    if (source !== target && !slots.has(target) && (conditions & (1 << (target + 4)))) throw new Error('A copied switch would alias an external switch requirement. Choose another unused condition bit.');
  }
  let result = conditions;
  for (const source of slots.keys()) result &= ~(1 << (source + 4));
  for (const [source, target] of slots) if (conditions & (1 << (source + 4))) result |= 1 << (target + 4);
  return result;
}
