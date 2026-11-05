import { BinaryReader, checkedInteger } from './binary';
import type { DataPoint } from './dataBuild';
import type { Position } from './geometry';
import type { LevelDocument } from './level';

export type PointLightingBinding =
  | { kind: 'copy'; source: number }
  | { kind: 'segment'; from: number; to: number; numerator: number; denominator: number }
  | { kind: 'value'; lowerBrightness: number; upperBrightness: number };

export function resolvePointLighting(document: LevelDocument, binding: PointLightingBinding): Pick<DataPoint, 'lowerBrightness' | 'upperBrightness'> {
  const reader = new BinaryReader(document.files.data, 'source point lighting');
  const read = (id: number, upper: boolean) => {
    checkedInteger(id, 0, document.points.length - 1, 'Source lighting point');
    return reader.i16(document.pointBrightnessOffset + id * 4 + (upper ? 2 : 0));
  };
  const channel = (upper: boolean) => {
    if (binding.kind === 'copy') return read(binding.source, upper);
    if (binding.kind === 'segment') return interpolatePointBrightness(read(binding.from, upper), read(binding.to, upper), binding.numerator, binding.denominator);
    return checkedInteger(upper ? binding.upperBrightness : binding.lowerBrightness, -32768, 32767, 'Explicit point brightness');
  };
  return { lowerBrightness: channel(false), upperBrightness: channel(true) };
}

/** Current AB3DI.s currentPointBrights / level_get_point_brightness encoding:
 * signed low-byte base; high-byte animation selector/blend. Linear sampling
 * is an explicit geometry-authoring operation, not an average of packed words.
 */
export function interpolatePointBrightness(a: number, b: number, numerator: number, denominator: number): number {
  checkedInteger(a, -32768, 32767, 'Point brightness'); checkedInteger(b, -32768, 32767, 'Point brightness');
  checkedInteger(denominator, 1, Number.MAX_SAFE_INTEGER, 'Lighting segment denominator');
  checkedInteger(numerator, 0, denominator, 'Lighting segment numerator');
  if (numerator === 0 || a === b) return a;
  if (numerator === denominator) return b;
  const base = (word: number) => (word & 255) >= 128 ? (word & 255) - 256 : word & 255;
  const flags = (word: number) => word >>> 8 & 255;
  const encoded = (word: number) => base(word) >= 0 && flags(word) !== 0;
  if (encoded(a) !== encoded(b) || (encoded(a) && flags(a) !== flags(b)))
    throw new Error('This wall segment has incompatible point-light animation settings. Open Corner lighting in the room inspector and assign explicit values to the new corner before building.');
  // Integer geometry sampling truncates towards zero; BigInt avoids precision
  // loss for long rational spans. Animation bits never enter the arithmetic.
  const n = BigInt(numerator), d = BigInt(denominator);
  const sampled = Number((BigInt(base(a)) * (d - n) + BigInt(base(b)) * n) / d);
  if (!encoded(a)) return sampled;
  const word = flags(a) * 256 + (sampled & 255); return word >= 32768 ? word - 65536 : word;
}

/** Assign generated point-table lighting from explicit source provenance.
 * Retained original identities keep their exact words. New or moved identities
 * require bindings; neither nearest-coordinate guesses nor zero defaults can
 * silently replace authored animation settings. Coordinates belong to geometry.
 */
export function buildPointLighting(document: LevelDocument, points: readonly Position[],
  bindings: ReadonlyMap<number, PointLightingBinding>): DataPoint[] {
  checkedInteger(points.length, 1, 2048, 'Generated point count');
  for (const id of bindings.keys()) checkedInteger(id, 0, points.length - 1, 'Point lighting target');
  const reader = new BinaryReader(document.files.data, 'source point lighting');
  const read = (id: number, upper: boolean): number => {
    checkedInteger(id, 0, document.points.length - 1, 'Source lighting point');
    return reader.i16(document.pointBrightnessOffset + id * 4 + (upper ? 2 : 0));
  };
  return points.map((point, id) => {
    checkedInteger(point.x, -32768, 32767, 'Generated point X'); checkedInteger(point.z, -32768, 32767, 'Generated point Z');
    const binding = bindings.get(id);
    if (!binding) {
      if (document.points[id]?.x !== point.x || document.points[id]?.z !== point.z)
        throw new Error(`Generated point ${id} at (${point.x}, ${point.z}) has no lighting assignment.`);
      return { x: point.x, z: point.z, lowerBrightness: read(id, false), upperBrightness: read(id, true) };
    }
    return { x: point.x, z: point.z, ...resolvePointLighting(document, binding) };
  });
}
