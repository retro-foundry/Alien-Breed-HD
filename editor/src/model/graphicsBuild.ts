import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import { encodePolygonSurface, encodeWallSurface } from './surfaces';
import { BinaryWriter } from './writer';

/** Reassemble current-format graphics streams and relocate every known
 * graphics pointer consumed by Anims.s and level.c. Zone-data offsets remain
 * supplied by the data compiler; material phases and animation bases are not
 * pointers and must never be rebased. Extra slots support the geometry writer.
 */
export function buildGraphicsStreams(document: LevelDocument, order = document.zones.map(zone => zone.id), slots?: number): Uint8Array {
  const reader = new BinaryReader(document.files.graphics, 'twolev.graph.bin');
  const oldTable = reader.i32(12), oldSlots = (oldTable - 16) / 4;
  const count = checkedInteger(slots ?? oldSlots, oldSlots, 256, 'Graphics zone capacity');
  if (order.length !== document.zones.length || new Set(order).size !== order.length || order.some(id => !document.zones.some(zone => zone.id === id)))
    throw new Error('Graphics assembly must include each source zone exactly once.');
  const streams = document.zones.flatMap(zone => ([false, true] as const).flatMap(upper => {
    const start = reader.i32(oldTable + zone.id * 8 + (upper ? 4 : 0));
    if (start === 0) return [];
    const records = upper ? zone.upper : zone.lower;
    const end = records.length ? records.at(-1)!.offset + records.at(-1)!.size + 4 : start + 6;
    if (reader.i16(start) !== zone.id || reader.i32(end - 4) !== -1)
      throw new Error(`Zone ${zone.id} has an unsupported graphics stream header or terminator.`);
    return [{ zone: zone.id, upper, start, end, records }];
  }));
  const sorted = streams.toSorted((a, b) => a.start - b.start), bodyStart = 16 + oldSlots * 12;
  let cursor = bodyStart;
  for (const stream of sorted) {
    if (stream.start !== cursor) throw new Error('The graphics streams contain unaccounted source data. Original files have been kept.');
    cursor = stream.end;
  }
  const tail = cursor, output = new BinaryWriter(), table = 16 + count * 4;
  output.write(reader.bytes.subarray(0, 16)); output.reserve(count * 12); output.patchLong(12, table);
  for (let id = 0; id < count; id++) output.patchLong(16 + id * 4, id < oldSlots ? reader.i32(16 + id * 4) : -1);
  const ranges: { start: number; end: number; destination: number }[] = [];
  for (const id of order) for (const upper of [false, true]) {
    const stream = streams.find(stream => stream.zone === id && stream.upper === upper);
    if (!stream) continue;
    const destination = output.offset;
    output.patchLong(table + id * 8 + (upper ? 4 : 0), destination); output.word(id);
    for (const record of stream.records) output.write(record.wall ? encodeWallSurface(record.wall)
      : record.polygon ? encodePolygonSurface(record.polygon) : reader.bytes.subarray(record.offset, record.offset + record.size));
    output.long(-1);
    if (output.offset - destination !== stream.end - stream.start) throw new Error('A graphics record changed size without a relocation map.');
    ranges.push({ start: stream.start, end: stream.end, destination });
  }
  const destinationTail = output.offset;
  output.write(reader.bytes.subarray(tail));
  ranges.push({ start: tail, end: reader.bytes.length, destination: destinationTail });
  const relocate = (offset: number) => {
    if (offset === 0) return 0;
    const range = ranges.find(range => offset >= range.start && offset < range.end);
    if (!range) throw new Error(`A graphics pointer at ${offset} is outside the compiled streams and animation tables.`);
    return range.destination + offset - range.start;
  };
  for (const header of [0, 4, 8]) {
    const offset = reader.i32(header);
    if (offset > 16) output.patchLong(header, relocate(offset));
  }
  for (const mechanism of document.mechanisms) {
    output.patchLong(relocate(mechanism.offset + 8), relocate(reader.i32(mechanism.offset + 8)));
    mechanism.walls.forEach((wall, index) => output.patchLong(relocate(mechanism.offset + 18 + index * 10 + 2), relocate(wall.graphic)));
  }
  for (const item of document.switches) if (item.zone >= 0)
    output.patchLong(relocate(item.offset + 6), relocate(item.graphic));
  // The lift terminator is immediately followed by all 21 water animation
  // groups: 14-byte header, (zone word + graphics pointer long)*, -1 word.
  const lift = reader.i32(4);
  if (lift > 16) {
    let water = lift;
    for (const mechanism of document.mechanisms.filter(item => item.kind === 'lift')) {
      if (mechanism.offset !== water) throw new Error('The lift animation layout contains unaccounted source data.');
      water += 18 + mechanism.walls.length * 10 + 2;
    }
    if (reader.i16(water) !== 999) throw new Error('The lift animation table has no supported terminator.');
    water += 2;
    for (let group = 0; group < 21; group++) {
      reader.require(water, 14); water += 14;
      let terminated = false;
      for (let entry = 0; entry <= 256; entry++) {
        const zone = reader.i16(water);
        if (zone === -1) { water += 2; terminated = true; break; }
        if (!document.zones.some(item => item.id === zone)) throw new Error('A water animation refers to a missing zone.');
        output.patchLong(relocate(water + 2), relocate(reader.i32(water + 2))); water += 6;
      }
      if (!terminated) throw new Error('A water animation group has no terminator.');
    }
  }
  return output.finish();
}
