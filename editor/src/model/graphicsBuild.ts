import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import { encodePolygonSurface, encodeWallSurface } from './surfaces';
import { BinaryWriter } from './writer';
import { readGraphicRecord, type Graphic } from './graphics';
import { buildAnimationTables, importedAnimationTables, remapAnimationRooms, type AnimationTables } from './animationBuild';
import { relocateRecordPointer } from './graphicsRelocation';
import type { Position } from './geometry';

export interface GraphicsZoneInput { id: number; lower: readonly Uint8Array[]; upper?: readonly Uint8Array[] }
export interface GraphicsRecordInput {
  bytes: Uint8Array;
  sourceOffset?: number;
  /** Full authored lighting span before cutting a new opening. */
  lightingSpan?: readonly [Position, Position];
  /** Explicit lighting for newly authored closing walls. */
  lightingPoint?: number;
}
export interface GraphicsStreamInput { lower: readonly GraphicsRecordInput[]; upper?: readonly GraphicsRecordInput[] }
export interface GraphicsEdits {
  replacements?: ReadonlyMap<number, Uint8Array>;
  additions?: readonly GraphicsZoneInput[];
  streams?: ReadonlyMap<number, GraphicsStreamInput>;
  animations?: AnimationTables;
  /** Source-to-runtime permutation. Explicit animation tables already use
   * target IDs; otherwise imported animation rooms are remapped automatically. */
  roomIds?: ReadonlyMap<number, number>;
}

/** Explicit source identities let complete generated streams retain only the
 * records that still own animation dependencies. Upper-stream absence and an
 * existing empty stream are distinct in the original table. */
export function importedGraphicsStreams(document: LevelDocument): Map<number, GraphicsStreamInput> {
  const reader = new BinaryReader(document.files.graphics, 'twolev.graph.bin'), table = reader.i32(12);
  const records = (list: readonly Graphic[]) => list.map(record => ({ sourceOffset: record.offset,
    bytes: reader.bytes.slice(record.offset, record.offset + record.size) }));
  return new Map(document.zones.map(zone => [zone.id, { lower: records(zone.lower),
    ...(reader.i32(table + zone.id * 8 + 4) ? { upper: records(zone.upper) } : {}) }]));
}

function checkedRecord(bytes: Uint8Array): Graphic {
  const record = readGraphicRecord(new BinaryReader(bytes, 'generated graphics record'), 0, 2048);
  if (record.size !== bytes.length) throw new Error('A generated graphics record contains trailing or incomplete data.');
  return record;
}

/** Reassemble current-format graphics streams and relocate every known
 * graphics pointer consumed by Anims.s and level.c. Zone-data offsets remain
 * supplied by the data compiler; material phases and animation bases are not
 * pointers and must never be rebased. Extra slots support the geometry writer.
 */
export function buildGraphicsStreams(document: LevelDocument, order = document.zones.map(zone => zone.id), slots?: number, edits: GraphicsEdits = {}): Uint8Array {
  const reader = new BinaryReader(document.files.graphics, 'twolev.graph.bin');
  const oldTable = reader.i32(12), oldSlots = (oldTable - 16) / 4;
  const count = checkedInteger(slots ?? Math.max(oldSlots, ...(edits.additions ?? []).map(zone => zone.id + 1)), oldSlots, 256, 'Graphics zone capacity');
  if (edits.roomIds && (edits.roomIds.size !== document.zones.length || [...edits.roomIds.keys()].some(id => !document.zones.some(zone => zone.id === id))
    || new Set(edits.roomIds.values()).size !== document.zones.length)) throw new Error('Graphics room remapping needs each source room exactly once and unique target identities.');
  for (const id of edits.roomIds?.values() ?? []) checkedInteger(id, 0, document.zones.length - 1, 'Generated graphics room');
  const destinationId = (id: number) => edits.roomIds?.get(id) ?? id;
  const destinations = new Set(document.zones.map(zone => destinationId(zone.id)));
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
  const sourceRecords = new Map(streams.flatMap(stream => stream.records.map(record => [record.offset, record] as const)));
  for (const [offset, bytes] of edits.replacements ?? []) {
    const original = sourceRecords.get(offset);
    if (!original) throw new Error(`The replacement graphics record at ${offset} no longer exists.`);
    if (checkedRecord(bytes).type !== original.type) throw new Error('A replacement must retain its graphics opcode; changing record roles needs explicit dependency remapping.');
    const stream = streams.find(stream => stream.records.some(record => record.offset === offset))!;
    if (edits.streams?.has(stream.zone)) throw new Error('Use either complete stream assembly or individual replacements for a room, not both.');
  }
  for (const [id, generated] of edits.streams ?? []) {
    if (!document.zones.some(zone => zone.id === id)) throw new Error('A complete graphics stream refers to a missing source room.');
    if (!Array.isArray(generated.lower)) throw new Error('Complete room graphics need an explicit lower stream.');
    if (generated.upper !== undefined && !Array.isArray(generated.upper)) throw new Error('An upper graphics stream must be a record list or be absent.');
    for (const [upper, records] of [[false, generated.lower], [true, generated.upper]] as const) {
      if (!records) continue;
      checkedInteger(records.length, 0, 4095, 'Graphics record count');
      const original = streams.find(stream => stream.zone === id && stream.upper === upper), identities = new Set<number>();
      for (const item of records) {
        const record = checkedRecord(item.bytes);
        if (item.sourceOffset === undefined) continue;
        const source = original?.records.find(record => record.offset === item.sourceOffset);
        if (!source || identities.has(item.sourceOffset)) throw new Error(`Room ${id} has a missing or duplicate source graphics record identity.`);
        identities.add(item.sourceOffset);
        if (record.type !== source.type) throw new Error('A generated record must retain its source graphics opcode for dependency remapping.');
      }
    }
  }
  const additionIds = new Set<number>();
  for (const zone of edits.additions ?? []) {
    checkedInteger(zone.id, 0, count - 1, 'Additional graphics zone');
    if (destinations.has(zone.id) || additionIds.has(zone.id)) throw new Error('Additional graphics streams must have unique new zone identities.');
    additionIds.add(zone.id);
    for (const records of [zone.lower, zone.upper].filter(records => records !== undefined)) {
      checkedInteger(records.length, 1, 4095, 'Graphics record count'); records.forEach(checkedRecord);
    }
  }
  let cursor = bodyStart;
  for (const stream of sorted) {
    if (stream.start !== cursor) throw new Error('The graphics streams contain unaccounted source data. Original files have been kept.');
    cursor = stream.end;
  }
  const tail = cursor, output = new BinaryWriter(), table = 16 + count * 4;
  output.write(reader.bytes.subarray(0, 16)); output.reserve(count * 12); output.patchLong(12, table);
  for (let id = 0; id < count; id++) output.patchLong(16 + id * 4, !edits.roomIds && id < oldSlots ? reader.i32(16 + id * 4) : -1);
  if (edits.roomIds) for (const zone of document.zones) output.patchLong(16 + destinationId(zone.id) * 4, reader.i32(16 + zone.id * 4));
  const ranges: { start: number; end: number; destination: number; original?: Graphic; generated?: Graphic }[] = [];
  for (const id of order) for (const upper of [false, true]) {
    const stream = streams.find(stream => stream.zone === id && stream.upper === upper);
    const override = edits.streams?.get(id), generated = upper ? override?.upper : override?.lower;
    if (override ? !generated : !stream) continue;
    const destination = output.offset;
    output.patchLong(table + destinationId(id) * 8 + (upper ? 4 : 0), destination); output.word(destinationId(id));
    if (stream) ranges.push({ start: stream.start, end: stream.start + 2, destination });
    const records = generated ?? stream!.records.map(record => ({ sourceOffset: record.offset,
      bytes: edits.replacements?.get(record.offset) ?? (record.wall ? encodeWallSurface(record.wall)
        : record.polygon ? encodePolygonSurface(record.polygon) : reader.bytes.subarray(record.offset, record.offset + record.size)) }));
    for (const item of records) {
      const start = output.offset; output.write(item.bytes);
      if (item.sourceOffset === undefined) continue;
      const record = sourceRecords.get(item.sourceOffset)!;
      ranges.push({ start: record.offset, end: record.offset + record.size, destination: start, original: record, generated: checkedRecord(item.bytes) });
    }
    if (stream) ranges.push({ start: stream.end - 4, end: stream.end, destination: output.offset });
    output.long(-1);
  }
  for (const zone of edits.additions ?? []) for (const [upper, records] of [[false, zone.lower], [true, zone.upper]] as const) {
    if (!records) continue;
    output.patchLong(table + zone.id * 8 + (upper ? 4 : 0), output.offset); output.word(zone.id);
    records.forEach(record => output.write(record)); output.long(-1);
  }
  const relocate = (offset: number) => {
    if (offset === 0) return 0;
    const range = ranges.find(range => offset >= range.start && offset < range.end);
    if (!range) throw new Error(`A graphics pointer at ${offset} is outside the compiled streams and animation tables.`);
    const relative = offset - range.start;
    if (range.original && range.generated) return relocateRecordPointer(range.original, range.generated, range.destination, offset);
    return range.destination + relative;
  };
  const imported = importedAnimationTables(document);
  if (imported.start !== tail) throw new Error('The graphics body and animation tables contain unaccounted source data.');
  const animationInput = edits.animations ?? (edits.roomIds ? remapAnimationRooms(document, imported.tables, edits.roomIds) : imported.tables);
  const animations = buildAnimationTables(animationInput, output.offset, relocate);
  output.write(animations.bytes);
  for (const [field, offset] of [[0, animations.doors], [4, animations.lifts], [8, animations.switches]])
    if (edits.animations !== undefined || reader.i32(field!) > 16) output.patchLong(field!, offset!);
  return output.finish();
}
