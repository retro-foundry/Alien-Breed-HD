import { BinaryReader, checkedInteger } from './binary';
import { BinaryWriter } from './writer';

export interface WallSurface {
  type: 0 | 13;
  points: [number, number];
  stripStart: number; stripEnd: number;
  horizontalOffset: number; verticalOffset: number;
  texture: number; verticalMask: number; verticalShift: number; horizontalMask: number;
  topFixed: number; bottomFixed: number; brightness: number;
}
export const polygonTypes = [1, 2, 7, 8, 9, 10, 11] as const;
export type PolygonType = typeof polygonTypes[number];
export interface PolygonSurface {
  type: PolygonType;
  heightQuarter: number;
  points: number[];
  closingPoint: number;
  scale: number; texture: number; brightness: number;
}

function point(value: number): number { return checkedInteger(value, 0, 2047, 'Surface point index'); }
function word(value: number, label: string): number { return checkedInteger(value, -32768, 32767, label); }

/** Current renderer.c case 0/13, WallRoutine3.ChipMem.s itsawalldraw.
 * Keep authored phase, sampling masks and fixed-point heights as separate
 * fields. Replacing them with inferred texture dimensions regressed doors. */
export function readWallSurface(reader: BinaryReader, offset: number): WallSurface {
  reader.require(offset, 30);
  const type = reader.i16(offset);
  if (type !== 0 && type !== 13) throw new Error('This graphics record is not a straight wall.');
  const p = offset + 2;
  return { type, points: [reader.i16(p), reader.i16(p + 2)], stripStart: reader.i16(p + 4), stripEnd: reader.i16(p + 6),
    horizontalOffset: reader.i16(p + 8), verticalOffset: reader.i16(p + 10), texture: reader.i16(p + 12),
    verticalMask: reader.u8(p + 14), verticalShift: reader.u8(p + 15), horizontalMask: reader.i16(p + 16),
    topFixed: reader.i32(p + 18), bottomFixed: reader.i32(p + 22), brightness: reader.i16(p + 26) };
}

export function encodeWallSurface(surface: WallSurface): Uint8Array {
  if (surface.type !== 0 && surface.type !== 13) throw new Error('Choose a straight or see-through wall type.');
  checkedInteger(surface.points.length, 2, 2, 'Wall endpoint count');
  const output = new BinaryWriter(); output.word(surface.type);
  surface.points.forEach(value => output.word(point(value)));
  for (const [value, label] of [[surface.stripStart, 'Wall strip start'], [surface.stripEnd, 'Wall strip end'],
    [surface.horizontalOffset, 'Wall horizontal phase'], [surface.verticalOffset, 'Wall vertical phase'], [surface.texture, 'Wall material']] as const)
    output.word(word(value, label));
  output.byte(surface.verticalMask); output.byte(surface.verticalShift); output.word(word(surface.horizontalMask, 'Wall horizontal mask'));
  output.long(checkedInteger(surface.topFixed, -2147483648, 2147483647, 'Wall top'));
  output.long(checkedInteger(surface.bottomFixed, -2147483648, 2147483647, 'Wall bottom'));
  output.word(word(surface.brightness, 'Wall brightness')); return output.finish();
}

/** Current renderer.c floor/roof/water cases and AB3DI.s itsafloordraw.
 * The first trailing word is the closing corner, followed by three material
 * words. Preserve it explicitly rather than treating all eight bytes as padding.
 */
export function readPolygonSurface(reader: BinaryReader, offset: number): PolygonSurface {
  const type = reader.i16(offset);
  if (!(polygonTypes as readonly number[]).includes(type)) throw new Error('This graphics record is not a floor, roof or water polygon.');
  const count = reader.i16(offset + 4) + 1;
  checkedInteger(count, 0, 100, 'Surface corner count'); reader.require(offset, 14 + count * 2);
  const tail = offset + 6 + count * 2;
  return { type: type as PolygonType, heightQuarter: reader.i16(offset + 2),
    points: Array.from({ length: count }, (_, i) => reader.i16(offset + 6 + i * 2)),
    closingPoint: reader.i16(tail), scale: reader.i16(tail + 2), texture: reader.i16(tail + 4), brightness: reader.i16(tail + 6) };
}

export function encodePolygonSurface(surface: PolygonSurface): Uint8Array {
  if (!(polygonTypes as readonly number[]).includes(surface.type)) throw new Error('Choose a supported floor, roof or water polygon type.');
  checkedInteger(surface.points.length, 0, 100, 'Surface corner count');
  const output = new BinaryWriter(); output.word(surface.type); output.word(word(surface.heightQuarter, 'Surface height'));
  output.word(surface.points.length - 1); surface.points.forEach(value => output.word(point(value)));
  output.word(word(surface.closingPoint, 'Closing corner')); output.word(word(surface.scale, 'Surface scale'));
  output.word(word(surface.texture, 'Surface material')); output.word(word(surface.brightness, 'Surface brightness'));
  return output.finish();
}

/** Bind an explicitly selected material record to a generated convex cell.
 * The compiler supplies oriented point IDs; no drawing-direction UI is needed.
 * Empty/legacy degenerate polygons can round-trip but cannot author new rooms. */
export function polygonForCell(template: PolygonSurface, vertices: readonly number[], height: number): PolygonSurface {
  checkedInteger(vertices.length, 3, 10, 'Convex surface corner count');
  const heightQuarter = word(height * 4, 'Surface height in quarter units');
  return { ...template, points: vertices.map(point), closingPoint: vertices[0]!, heightQuarter };
}
