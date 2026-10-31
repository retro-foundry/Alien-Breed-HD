import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import { BinaryWriter } from './writer';

export interface AnimationWall { line: number; graphic: number; base: number }
export interface MovingRoomAnimation { header: Uint8Array; zone: number; graphic: number; walls: readonly AnimationWall[] }
export interface WaterAnimation { header: Uint8Array; entries: readonly { zone: number; graphic: number }[] }
export interface SwitchAnimation { bytes: Uint8Array; zone: number; point: number; graphic: number }
export interface AnimationTables {
  doors: readonly MovingRoomAnimation[] | null;
  lifts: readonly MovingRoomAnimation[] | null;
  water: readonly WaterAnimation[];
  switches: readonly SwitchAnimation[] | null;
  trailing: Uint8Array;
}

/** Current Anims.s/level.c layout, not the older AMOS header variant. Preserve
 * opaque position/rule/runtime bytes; identify authored room/line/point and
 * graphics relationships explicitly for the geometry compiler to remap. */
export function importedAnimationTables(document: LevelDocument): { start: number; tables: AnimationTables } {
  const reader = new BinaryReader(document.files.graphics, 'twolev.graph.bin');
  const present = [0, 4, 8].map(field => reader.i32(field)).filter(offset => offset > 16);
  const start = present.length ? Math.min(...present) : reader.bytes.length;
  let cursor = start;
  const moving = (kind: 'door' | 'lift', offset: number): MovingRoomAnimation[] | null => {
    if (offset <= 16) return null;
    if (cursor !== offset) throw new Error(`The ${kind} animation layout contains unaccounted source data.`);
    const result: MovingRoomAnimation[] = [];
    for (const mechanism of document.mechanisms.filter(item => item.kind === kind)) {
      if (mechanism.offset !== cursor) throw new Error(`The ${kind} animation records are not contiguous.`);
      result.push({ header: reader.bytes.slice(cursor, cursor + 18), zone: mechanism.zone, graphic: reader.i32(cursor + 8),
        walls: mechanism.walls.map(wall => ({ ...wall })) });
      cursor += 18 + mechanism.walls.length * 10;
      if (reader.i16(cursor) !== -1) throw new Error(`The ${kind} wall list has an unsupported terminator.`);
      cursor += 2;
    }
    if (reader.i16(cursor) !== 999) throw new Error(`The ${kind} table has no 999 terminator.`);
    cursor += 2; return result;
  };
  const doors = moving('door', reader.i32(0)), lifts = moving('lift', reader.i32(4)), water: WaterAnimation[] = [];
  if (lifts !== null) for (let group = 0; group < 21; group++) {
    reader.require(cursor, 14);
    const header = reader.bytes.slice(cursor, cursor + 14), entries: { zone: number; graphic: number }[] = [];
    cursor += 14;
    for (;;) {
      const zone = reader.i16(cursor); cursor += 2;
      if (zone === -1) break;
      if (entries.length >= 256 || !document.zones.some(room => room.id === zone)) throw new Error('A water animation has an invalid room list.');
      entries.push({ zone, graphic: reader.i32(cursor) }); cursor += 4;
    }
    water.push({ header, entries });
  }
  let switches: SwitchAnimation[] | null = null;
  const switchOffset = reader.i32(8);
  if (switchOffset > 16) {
    if (cursor !== switchOffset) throw new Error('The switch animation layout contains unaccounted source data.');
    switches = [];
    for (let slot = 0; slot < 8; slot++) {
      reader.require(cursor, 14);
      switches.push({ bytes: reader.bytes.slice(cursor, cursor + 14), zone: reader.i16(cursor), point: reader.i16(cursor + 4), graphic: reader.i32(cursor + 6) });
      cursor += 14;
    }
  }
  return { start, tables: { doors, lifts, water, switches, trailing: reader.bytes.slice(cursor) } };
}

export function buildAnimationTables(tables: AnimationTables, base: number, resolveGraphic: (source: number) => number):
  { bytes: Uint8Array; doors: number; lifts: number; switches: number } {
  checkedInteger(base, 16, 16 * 1024 * 1024, 'Animation table offset');
  const output = new BinaryWriter();
  const pointer = (value: number) => checkedInteger(resolveGraphic(value), 0, 16 * 1024 * 1024 - 1, 'Animation graphics pointer');
  const moving = (records: readonly MovingRoomAnimation[] | null) => {
    if (records === null) return 0;
    checkedInteger(records.length, 0, 256, 'Moving room count');
    const offset = base + output.offset;
    for (const record of records) {
      if (record.header.length !== 18 || new BinaryReader(record.header, 'moving room header').i16(0) === 999)
        throw new Error('A moving room needs an 18-byte header whose height is not the reserved 999 terminator.');
      checkedInteger(record.walls.length, 0, 64, 'Moving wall count');
      const header = output.offset; output.write(record.header);
      output.patchWord(header + 12, checkedInteger(record.zone, 0, 255, 'Controlled room'));
      output.patchLong(header + 8, pointer(record.graphic));
      for (const wall of record.walls) {
        output.word(checkedInteger(wall.line, 0, 32767, 'Animated collision line'));
        output.long(pointer(wall.graphic)); output.long(checkedInteger(wall.base, -2147483648, 2147483647, 'Animated wall texture phase'));
      }
      output.word(-1);
    }
    output.word(999); return offset;
  };
  const doors = moving(tables.doors), lifts = moving(tables.lifts);
  if (tables.water.length !== (tables.lifts === null ? 0 : 21)) throw new Error('A lift table must be followed by exactly 21 water animation groups.');
  for (const group of tables.water) {
    if (group.header.length !== 14) throw new Error('A water animation needs a complete 14-byte header.');
    checkedInteger(group.entries.length, 0, 256, 'Water animation room count');
    output.write(group.header);
    for (const entry of group.entries) { output.word(checkedInteger(entry.zone, 0, 255, 'Water animation room')); output.long(pointer(entry.graphic)); }
    output.word(-1);
  }
  let switches = 0;
  if (tables.switches !== null) {
    checkedInteger(tables.switches.length, 8, 8, 'Switch slot count'); switches = base + output.offset;
    for (const record of tables.switches) {
      if (record.bytes.length !== 14) throw new Error('A switch slot needs a complete 14-byte record.');
      const offset = output.offset; output.write(record.bytes);
      output.patchWord(offset, checkedInteger(record.zone, -32768, 255, 'Switch room'));
      output.patchWord(offset + 4, checkedInteger(record.point, record.zone >= 0 ? 0 : -32768, record.zone >= 0 ? 2046 : 32767, 'Switch point pair'));
      // Inactive graphics fields are stored data, not live relocation targets.
      output.patchLong(offset + 6, record.zone >= 0 ? pointer(record.graphic)
        : checkedInteger(record.graphic, -2147483648, 2147483647, 'Inactive switch graphics field'));
    }
  }
  output.write(tables.trailing);
  checkedInteger(base + output.offset, 0, 16 * 1024 * 1024, 'Graphics file size');
  return { bytes: output.finish(), doors, lifts, switches };
}
