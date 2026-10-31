/* Tangent-envelope translation of LRBORD (leveld.asc lines 6585–6662).
 * New authoring outlines are normalised/decomposed upstream. Imported runtime
 * outlines retain their original first corner: collinear tangent choices can
 * affect the original clipping cache. Screen Y is negative world Z. */
import type { Position } from './geometry';

export interface PortalEnvelope { points: [Position, Position, Position, Position]; mode: 0 | 1 | 2 }
const same = (a: Position, b: Position) => a.x === b.x && a.z === b.z;
// AMOS integer arithmetic here is full-width. Unlike the Asm bank, these
// statements do not first truncate coordinate differences with SUB.W.
function side(a: Position, b: Position, p: Position): number {
  return (Math.imul(b.x - a.x, -(p.z - a.z)) - Math.imul(-(b.z - a.z), p.x - a.x)) | 0;
}

export function findPortalEnvelope(sourceOutline: readonly Position[], targetOutline: readonly Position[]): PortalEnvelope | null {
  const source = sourceOutline, target = targetOutline;
  if (source.length < 3 || source.length > 10 || target.length < 3 || target.length > 10)
    throw new Error('Visibility compilation requires room boundaries with three to ten corners.');
  function tangent(left: boolean): [number, number] {
    let from = same(source[0]!, target[0]!) ? 1 : 0, to = 0;
    const visited = new Set<string>();
    for (;;) {
      const state = `${from},${to}`;
      if (visited.has(state)) throw new Error('The visibility envelope cannot settle. Check overlapping room outlines.');
      visited.add(state);
      let changed = false;
      const initialTarget = to;
      const targetStep = left ? target.length - 1 : 1;
      for (let next = (to + targetStep) % target.length; next !== initialTarget; next = (next + targetStep) % target.length) {
        const value = side(source[from]!, target[to]!, target[next]!);
        if (left ? value < 0 : value > 0) { to = next; changed = true; }
      }
      const initialSource = from;
      const sourceStep = left ? 1 : source.length - 1;
      for (let next = (from + sourceStep) % source.length; next !== initialSource; next = (next + sourceStep) % source.length) {
        const value = side(target[to]!, source[from]!, source[next]!);
        if (left ? value > 0 : value < 0) { from = next; changed = true; }
      }
      if (!changed) return [from, to];
    }
  }
  const [sl, tl] = tangent(true), [sr, tr] = tangent(false);
  const sourceCollapsed = sl === sr, targetCollapsed = tl === tr;
  // A pair seen only through one point has no positive-width envelope. The
  // AMOS caller left its previous mode in memory in this case; do not carry
  // state from a different room pair into the new compiler.
  if (sourceCollapsed && targetCollapsed) return null;
  return { points: [source[sl]!, target[tl]!, target[tr]!, source[sr]!], mode: sourceCollapsed ? 1 : targetCollapsed ? 2 : 0 };
}
