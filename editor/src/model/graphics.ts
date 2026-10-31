import { BinaryReader, LevelFormatError } from './binary';
import { readWallSurface, readPolygonSurface, type WallSurface, type PolygonSurface } from './surfaces';

export interface Graphic {
  type: number; offset: number; size: number; points: number[];
  height?: number; top?: number; bottom?: number; texture?: number;
  wall?: WallSurface; polygon?: PolygonSurface;
}
/** One current renderer opcode, shared by imported streams and generated
 * record validation. A terminator belongs to a stream, not a record. */
export function readGraphicRecord(reader: BinaryReader, cursor: number, points: number): Graphic {
  const type = reader.i16(cursor), entry: Graphic = { type, offset: cursor, size: 0, points: [] }, payload = cursor + 2;
  let size: number;
  if (type === 0 || type === 13 || type === 5) {
    size = 30; entry.points = [reader.i16(payload), reader.i16(payload + 2)];
    if (type !== 5) {
      entry.wall = readWallSurface(reader, cursor); entry.texture = entry.wall.texture;
      entry.top = entry.wall.topFixed / 256; entry.bottom = entry.wall.bottomFixed / 256;
    }
  } else if (type === 3 || type === 12) size = 2;
  else if (type === 4) size = 4;
  else if (type === 6) { size = 6; entry.points = [reader.i16(payload), reader.i16(payload + 2)]; }
  else if ([1, 2, 7, 8, 9, 10, 11].includes(type)) {
    const sides = reader.i16(payload + 2) + 1;
    if (sides < 0 || sides > 100) throw new LevelFormatError('polygon has an invalid vertex count', reader.name, payload + 2);
    size = 14 + sides * 2; entry.polygon = readPolygonSurface(reader, cursor);
    entry.height = entry.polygon.heightQuarter / 4; entry.points = entry.polygon.points; entry.texture = entry.polygon.texture;
  } else throw new LevelFormatError(`unrecognised graphics opcode ${type}`, reader.name, cursor);
  reader.require(cursor, size);
  if (entry.points.some(point => point < 0 || point >= points)) throw new LevelFormatError('graphic references a point outside the point table', reader.name, cursor);
  entry.size = size; return entry;
}
export function readGraphicsStream(reader: BinaryReader, offset: number, points: number): Graphic[] {
  if (offset === 0) return [];
  reader.require(offset, 2);
  const result: Graphic[] = [];
  let cursor = offset + 2;
  for (let count = 0; count < 4096; count++) {
    const type = reader.i16(cursor);
    if (type < 0) return result;
    const entry = readGraphicRecord(reader, cursor, points);
    result.push(entry); cursor += entry.size;
  }
  throw new LevelFormatError('graphics stream has no terminator', reader.name, offset);
}
