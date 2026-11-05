import { activeNativeSwitches } from './nativeSwitches';
import type { LevelDocument } from './level';
import { checkedInteger } from './binary';
import { nativeRoomTranslation } from './roomTranslation';
import { importedAnimationTables } from './animationBuild';
import type { RoomChange } from './roomSource';

export interface SourceSwitchCopy {
  readonly id: number;
  readonly slot: number;
  readonly room: string;
  readonly graphicsRoom: string;
  readonly bytes: readonly number[];
}

export function restoreSwitchCopies(value: unknown): readonly SourceSwitchCopy[] {
  if (!Array.isArray(value) || value.length > 8) throw new Error('Copied switch source must contain at most eight records.');
  const slots = new Set<number>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).length !== 5
      || Object.keys(item).some(key => !['id', 'slot', 'room', 'graphicsRoom', 'bytes'].includes(key))) throw new Error('Copied switch source has missing or unsupported fields.');
    const id = checkedInteger(item.id, 0, 7, 'Switch template slot'), slot = checkedInteger(item.slot, 0, 7, 'Copied switch slot');
    if (slots.has(slot)) throw new Error('Copied switch source repeats a native destination slot.');
    slots.add(slot);
    for (const field of ['room', 'graphicsRoom']) if (typeof item[field] !== 'string' || !item[field].length || item[field].length > 128) throw new Error('Copied switch room identity is invalid.');
    if (!Array.isArray(item.bytes) || item.bytes.length !== 14) throw new Error('Copied switch source needs every native settings byte.');
    const bytes = Object.freeze(item.bytes.map((byte: number) => checkedInteger(byte, 0, 255, 'Copied switch settings byte')));
    return Object.freeze({ id, slot, room: item.room as string, graphicsRoom: item.graphicsRoom as string, bytes });
  }));
}

export function compileSwitchCopies(copies: readonly SourceSwitchCopy[]): readonly SwitchCopy[] {
  return copies.map(copy => ({ ...copy, bytes: new Uint8Array(copy.bytes) }));
}

export function setRoomSwitchCopies(document: LevelDocument, copies: readonly SwitchCopy[]): RoomChange {
  const tables = importedAnimationTables(document).tables;
  const captured = restoreSwitchCopies(copies.map(copy => {
    const template = tables.switches?.[copy.id];
    if (!template || template.zone < 0) throw new Error('A copied switch needs an active native template.');
    return { ...copy, bytes: [...(copy.bytes ?? template.bytes)] };
  }));
  validateSwitchCopies(document, compileSwitchCopies(captured));
  const before = document.roomSource;
  if (JSON.stringify(before.switchCopies ?? []) === JSON.stringify(captured)) return { label: 'Change copied switches', before, after: before };
  const { switchCopies: _previous, ...unchanged } = before;
  return { label: 'Change copied switches', before, after: Object.freeze({ ...unchanged, ...(captured.length ? { switchCopies: captured } : {}) }) };
}

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
    if (!destination || activeNativeSwitches(document).some(item => item.id === copy.slot) || slots.has(copy.slot)) throw new Error('A copied switch needs its own unused native slot. The level supports eight switches.');
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
