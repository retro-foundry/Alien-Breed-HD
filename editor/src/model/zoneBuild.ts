import { BinaryReader, checkedInteger } from './binary';
import { BinaryWriter } from './writer';

export interface ZoneRecordInput {
  id: number;
  header: Uint8Array;
  boundaries: readonly number[];
  cornerWalls: readonly number[];
  visibility: readonly { zone: number; clip: number; mask: number }[];
  rotationPoints: readonly number[];
}
export interface ZoneAssembly { data: Uint8Array; offsets: number[] }

/** Current Defs.i/level.c zone layout, independently checked against all
 * original room blocks. Prefix ends with LEVELSAVE's zone-count word.
 * Border lists precede the 48-byte header; visibility has a long terminator,
 * rotation a word terminator. Header-relative offsets are always regenerated.
 * Callers supply remapped headers and lists, not invented room defaults.
 */
export function buildZoneRecords(prefix: Uint8Array, records: readonly ZoneRecordInput[]): ZoneAssembly {
  checkedInteger(records.length, 1, 256, 'Runtime room count');
  const reader = new BinaryReader(prefix, 'level data prefix'); reader.require(0, 56);
  for (const field of [22, 26, 30, 34, 38, 42, 46, 50])
    checkedInteger(reader.i32(field), 54, prefix.length - 3, 'Preserved table offset');
  const pointCount = checkedInteger(reader.i16(14) + 1, 1, 2048, 'Point table count');
  reader.require(reader.i32(22), pointCount * 8);
  const lineCount = checkedInteger((prefix.length - 2 - reader.i32(26)) / 16, 1, 32768, 'Collision table count');
  const ids = new Set<number>();
  for (const record of records) {
    checkedInteger(record.id, 0, records.length - 1, 'Runtime room identity');
    if (ids.has(record.id)) throw new Error('The room build contains duplicate runtime identities.');
    ids.add(record.id);
    if (record.header.length !== 48) throw new Error(`Room ${record.id} needs a complete 48-byte header.`);
    checkedInteger(record.boundaries.length, 3, 10, 'Room boundary count');
    checkedInteger(record.boundaries.length + record.cornerWalls.length + 2, 5, 256, 'Room collision list size');
    for (const line of [...record.boundaries, ...record.cornerWalls]) checkedInteger(line, 0, lineCount - 1, 'Room collision line');
    checkedInteger(record.visibility.length, 1, records.length, 'Room visibility count');
    const visible = new Set<number>();
    for (const entry of record.visibility) {
      checkedInteger(entry.zone, 0, records.length - 1, 'Visible room');
      if (visible.has(entry.zone)) throw new Error(`Room ${record.id} has duplicate visibility entries.`);
      visible.add(entry.zone);
      checkedInteger(entry.clip, -32768, 32767, 'Clip flag'); checkedInteger(entry.mask, 0, 0xFFFFFFFF, 'Ordering mask');
    }
    if (!visible.has(record.id)) throw new Error(`Room ${record.id} is missing its own visibility entry.`);
    checkedInteger(record.rotationPoints.length, 0, 2048, 'Rotation point count');
    const rotate = new Set<number>();
    for (const point of record.rotationPoints) {
      checkedInteger(point, 0, pointCount - 1, 'Rotation point');
      if (rotate.has(point)) throw new Error(`Room ${record.id} has duplicate rotation points.`);
      rotate.add(point);
    }
  }
  const output = new BinaryWriter(), offsets: number[] = [];
  output.write(prefix); output.patchWord(16, records.length - 1); output.patchWord(prefix.length - 2, records.length);
  for (const record of records) {
    const border = output.offset;
    record.boundaries.forEach(line => output.word(line)); output.word(-1);
    record.cornerWalls.forEach(line => output.word(line)); output.word(-2);
    const header = output.offset; offsets[record.id] = header;
    output.write(record.header); output.patchWord(header, record.id);
    output.patchWord(header + 32, checkedInteger(border - header, -32768, 32767, 'Border-list relative offset'));
    for (const entry of record.visibility) { output.word(entry.zone); output.word(entry.clip); output.long(entry.mask); }
    output.long(-1);
    output.patchWord(header + 34, checkedInteger(output.offset - header, -32768, 32767, 'Rotation-list relative offset'));
    record.rotationPoints.forEach(point => output.word(point)); output.word(-1);
  }
  return { data: output.finish(), offsets };
}
