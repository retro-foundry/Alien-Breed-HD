import type { Graphic } from './graphics';

/** Polygon tails move when the corner count changes; header offsets do not.
 * An attachment inside replaced corner data needs an explicit new identity. */
export function relocateRecordPointer(original: Graphic, generated: Graphic, destination: number, pointer: number): number {
  const relative = pointer - original.offset;
  if (relative < 0 || relative >= original.size) throw new Error('A graphics pointer is outside its source record.');
  if (original.size !== generated.size) {
    if (!original.polygon || !generated.polygon) throw new Error('An unsupported graphics record changed size.');
    const oldTail = 6 + original.points.length * 2, newTail = 6 + generated.points.length * 2;
    if (relative >= oldTail) return destination + newTail + relative - oldTail;
    if (relative >= 6) throw new Error('A graphics pointer targets a regenerated polygon corner array. Remap that dependency before building.');
  }
  return destination + relative;
}
