/* Behavioral translation of leved.amos attached Asm bank 9. See
 * tools/editor_geometry_oracles.py for independently emulated 68020 fixtures.
 * This is the geometry primitive layer, not a completed PVS rebuild. */
import type { Position } from './geometry';

export interface OrderingRoom { id: number; outline: Position[]; neighbours: number[] }
export interface VisibilityRoom { id: number; vertices: number[]; neighbours: number[] }

const word = (value: number) => (value << 16) >> 16;

/** Original SUB.W/MULS.W/SUB.L side test, including signed-word subtraction. */
export function legacySide(a: Position, b: Position, point: Position): number {
  const dx = word(b.x - a.x), dz = word(b.z - a.z);
  return (Math.imul(dx, word(point.z - a.z)) - Math.imul(dz, word(point.x - a.x))) | 0;
}

/** CALCORDER, bank +0x2a6: three bits per target wall, starting at bit 1.
 * Wall type 1 is skipped. Preserve target order: runtime mask bits refer to it. */
export function zoneOrderMask(source: readonly Position[], target: readonly Position[], wallTypes: readonly number[]): number {
  if (source.length < 3 || target.length < 3 || target.length > 10 || target.length !== wallTypes.length)
    throw new Error('Ordering masks require valid rooms with at most ten target walls.');
  let mask = 0;
  for (let i = 0; i < target.length; i++) {
    if (wallTypes[i] === 1) continue;
    const a = target[i]!, b = target[(i + 1) % target.length]!;
    let negative = false, positive = false;
    for (const point of source) {
      const value = legacySide(a, b, point);
      negative ||= value < 0; positive ||= value > 0;
    }
    const bit = 1 + i * 3;
    if (negative && !positive) mask |= 1 << bit;
    if (positive && !negative) mask |= (1 << bit) | (1 << (bit + 1));
  }
  return mask >>> 0;
}

/** LEVELSAVE lines 1188–1228 adapts CALCORDER's cache to the current runtime:
 * source coordinates use screen Y=-world Z; bit 3*i gates an exit whose far
 * room belongs to the source room's visible set. The self entry uses the
 * original compact three-bit groups for connected exits. */
export function runtimeOrderingMask(source: OrderingRoom, target: OrderingRoom, visible: ReadonlySet<number>): number {
  if (target.outline.length !== target.neighbours.length || target.outline.length > 10)
    throw new Error('A room ordering mask does not match its wall/portal list.');
  if (source.id === target.id) {
    const exits = target.neighbours.filter(zone => zone >= 0).length;
    return ((1 << (exits * 3)) - 1) >>> 0;
  }
  const screenCoordinates = (room: OrderingRoom) => room.outline.map(point => ({ x: point.x, z: -point.z }));
  let mask = zoneOrderMask(screenCoordinates(source), screenCoordinates(target), target.neighbours.map(zone => zone < 0 ? 1 : 0));
  target.neighbours.forEach((zone, i) => { if (zone >= 0 && visible.has(zone)) mask |= 1 << (i * 3); });
  return mask >>> 0;
}

/** BETPTS, bank +0x346: classify wall vertices against the four directed
 * portal-envelope lines. Modes 1/2 are the original collapsed-end cases. */
export function classifyPortalPoints(portal: readonly Position[], points: readonly Position[], used: readonly number[], mode: 0 | 1 | 2): number[] {
  if (portal.length !== 4 || points.length !== used.length) throw new Error('Invalid portal geometry inputs.');
  return points.map((point, i) => {
    if (used[i] === 0) return 0;
    const d = portal.map((a, j) => legacySide(a, portal[(j + 1) % 4]!, point) <= 0);
    const a = d[0]!, c = d[2]!;
    if (mode === 0) {
      if (!d.some(Boolean)) return -3;
      if (a) return d[1] || d[3] ? -5 : -4;
      if (c) return d[1] || d[3] ? -1 : -2;
      return 0;
    }
    const middle = mode === 1 ? d[1]! : d[3]!;
    if (!a && !middle && !c) return -3;
    if (a) return middle || c ? 0 : -4;
    if (c) return middle || a ? 0 : -2;
    return 0;
  });
}

/** Bank +0x1f4, called by INITDEFORDER lines 6088–6133. A line between
 * left/right clips makes the pair invisible if no source vertex is strictly
 * to its positive side, or no target vertex is strictly to its negative side. */
export function clipsSeparateRooms(source: readonly Position[], target: readonly Position[], left: readonly Position[], right: readonly Position[]): boolean {
  if (source.length < 3 || target.length < 3) throw new Error('Portal clipping requires two valid rooms.');
  for (const a of left) for (const b of right) {
    if (!source.some(point => legacySide(a, b, point) > 0)) return true;
    if (!target.some(point => legacySide(a, b, point) < 0)) return true;
  }
  return false;
}
