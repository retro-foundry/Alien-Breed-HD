import { BinaryReader, checkedInteger } from './binary';
import { encodeCollisionSegments, type CollisionSegment } from './collisionBuild';
import type { Position } from './geometry';
import type { LevelDocument, LevelFiles } from './level';
import { BinaryWriter } from './writer';

export interface DataPoint extends Position { lowerBrightness: number; upperBrightness: number }
export interface DataTableInput { points: readonly DataPoint[]; collision: readonly CollisionSegment[] }

export function importedDataTables(document: LevelDocument): DataTableInput {
  const reader = new BinaryReader(document.files.data, 'twolev.bin');
  return { points: document.points.map(point => ({ x: point.x, z: point.z,
    lowerBrightness: reader.i16(document.pointBrightnessOffset + point.id * 4),
    upperBrightness: reader.i16(document.pointBrightnessOffset + point.id * 4 + 2) })),
    collision: document.lines.map(line => ({ start: { x: line.x, z: line.z }, end: { x: line.x + line.dx, z: line.z + line.dz }, neighbour: line.connect })) };
}

export function dataLayout(document: LevelDocument) {
  const reader = new BinaryReader(document.files.data, 'twolev.bin');
  const objectPoints = reader.i32(42), objects = reader.i32(30), points = reader.i32(22), collision = reader.i32(26);
  const firstZone = Math.min(...document.zones.map(zone => zone.exitListOffset));
  const objectCount = reader.i16(20) + 1, pointCount = reader.i16(14) + 1;
  if (objectPoints !== 54 + reader.i16(12) * 8 || objects !== objectPoints + objectCount * 8
    || points !== objects + document.objects.length * 64 + 2 || collision !== points + pointCount * 8
    || firstZone !== collision + document.lines.length * 16 + 2)
    throw new Error('The level tables contain unaccounted source data. Original files have been kept.');
  if (reader.i16(points - 2) !== -1 || reader.i16(firstZone - 2) !== document.zones.length)
    throw new Error('The object terminator or collision-table zone count is not supported.');
  for (const field of [34, 38, 46, 50]) {
    const offset = reader.i32(field);
    if (offset < objects || offset >= points - 2 || (offset - objects) % 64 !== 0)
      throw new Error('An object/shot-pool pointer is outside the preserved object table.');
  }
  return { reader, points, firstZone };
}

/** Assemble the non-room region for a complete geometry build. Point and
 * collision identities may change here: callers must assemble fresh room
 * records, graphics and dependency maps before exposing a playable file.
 * Retain the actual control/object/shot prefix; never invent pool records.
 * The header stores the last zone index, while the trailer stores its count.
 */
export function buildDataPrefix(document: LevelDocument, input: DataTableInput, roomCount: number): Uint8Array {
  const { reader, points } = dataLayout(document);
  checkedInteger(roomCount, 1, 256, 'Runtime room count');
  checkedInteger(input.points.length, 1, 2048, 'Point table size');
  checkedInteger(input.collision.length, 1, 32768, 'Collision table size');
  if (input.collision.some(line => line.neighbour !== -1 && (!Number.isInteger(line.neighbour) || line.neighbour < 0 || line.neighbour >= roomCount)))
    throw new Error('A collision line connects to a missing room. Remap room references before building.');
  const output = new BinaryWriter(); output.write(reader.bytes.subarray(0, points));
  output.patchWord(14, input.points.length - 1);
  output.patchWord(16, roomCount - 1);
  for (const point of input.points) {
    output.word(checkedInteger(point.x, -32768, 32767, 'Point X coordinate'));
    output.word(checkedInteger(point.z, -32768, 32767, 'Point Z coordinate'));
  }
  for (const point of input.points) {
    output.word(checkedInteger(point.lowerBrightness, -32768, 32767, 'Lower point brightness'));
    output.word(checkedInteger(point.upperBrightness, -32768, 32767, 'Upper point brightness'));
  }
  output.patchLong(26, output.offset); output.write(encodeCollisionSegments(input.collision));
  output.word(roomCount);
  return output.finish();
}

/** Retain existing room records and identities while rebuilding tables.
 * Shrinking referenced tables is only supported through the complete-build
 * prefix API, not through this original-room relocation path.
 */
export function buildDataTables(document: LevelDocument, input = importedDataTables(document)): LevelFiles {
  const { reader, firstZone } = dataLayout(document);
  checkedInteger(input.points.length, document.points.length, 2048, 'Point table size');
  checkedInteger(input.collision.length, document.lines.length, 32768, 'Collision table size');
  const prefix = buildDataPrefix(document, input, document.zones.length);
  const output = new BinaryWriter(); output.write(prefix);
  const displacement = prefix.length - firstZone;
  output.write(reader.bytes.subarray(firstZone));
  const graphics = document.files.graphics.slice(), view = new DataView(graphics.buffer);
  const graphicReader = new BinaryReader(graphics, 'twolev.graph.bin'), slots = (graphicReader.i32(12) - 16) / 4;
  for (let id = 0; id < slots; id++) {
    const old = graphicReader.i32(16 + id * 4);
    if (old < 0) continue;
    if (old < firstZone || old + 48 > reader.bytes.length) throw new Error('A zone header points outside the preserved room data.');
    view.setInt32(16 + id * 4, old + displacement);
  }
  return { data: output.finish(), graphics, clips: document.files.clips.slice() };
}
