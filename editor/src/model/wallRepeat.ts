import { checkedInteger } from './binary';
import type { Position } from './geometry';
import type { LevelDocument } from './level';
import type { SourceRoom } from './roomSource';
import type { WallSettings } from './wallSettings';
import type { WallSurface } from './surfaces';
import { wallForSpan } from './surfaceGeometry';
import { nativeRoomTranslation } from './roomTranslation';

export const wallRepeatWidths = [2, 4, 8, 16, 32, 64, 128, 256] as const;
export interface WallRepeat { readonly width: number; readonly fit: 'natural' | 'up' | 'down' }
export function restoreWallRepeat(value: unknown): WallRepeat {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== 2
    || !('width' in value) || !wallRepeatWidths.includes(value.width as typeof wallRepeatWidths[number])
    || !('fit' in value) || !['natural', 'up', 'down'].includes(value.fit as string)) throw new Error('Wall repeat needs a width of 2, 4, 8, 16, 32, 64, 128 or 256 columns and a natural, up or down fit.');
  return Object.freeze({ width: value.width as number, fit: value.fit as WallRepeat['fit'] });
}

/** leveld.asc WGPUT q/w and +/-; LEVELSAVE at 1324 and upper equivalent:
 * half-world-unit columns, with original up always adding one whole repeat
 * even on an exact multiple, and down keeping at least one whole repeat. */
export function wallRepeatColumns(length: number, repeat: WallRepeat): number {
  restoreWallRepeat(repeat);
  if (!Number.isFinite(length) || length <= 0) throw new Error('Wall repeat needs a wall with nonzero physical length.');
  let columns = Math.trunc(length / 2);
  if (repeat.fit === 'up') columns = Math.floor(columns / repeat.width) * repeat.width + repeat.width;
  if (repeat.fit === 'down') columns = Math.max(repeat.width, Math.floor(columns / repeat.width) * repeat.width);
  return checkedInteger(columns, 0, 32767, 'Wall repeat texture column count');
}

const distance = (a: Position, b: Position) => Math.hypot(b.x - a.x, b.z - a.z);
const on = (p: Position, a: Position, b: Position) => (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
  && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);

function wallChain(room: SourceRoom, matches: (index: number) => boolean) {
  const count = room.vertices.length;
  const first = room.vertices.findIndex((_, index) => matches(index) && !matches((index + count - 1) % count));
  if (first < 0) throw new Error('Wall repeat lost its source boundary. Restore the outline or reset this wall’s settings.');
  const pieces: { start: Position; end: Position; offset: number; length: number }[] = [];
  let total = 0, ended = false;
  for (let step = 0; step < count; step++) {
    const index = (first + step) % count;
    if (!matches(index)) { ended = true; continue; }
    if (ended) throw new Error('Wall repeat needs one continuous source wall. Reassign the separated wall pieces before applying repeat.');
    const start = room.vertices[index]!, end = room.vertices[(index + 1) % count]!, length = distance(start, end);
    pieces.push({ start, end, offset: total, length }); total += length;
  }
  const at = (point: Position) => {
    const piece = pieces.find(piece => on(point, piece.start, piece.end));
    if (!piece) throw new Error('Wall repeat lost a generated wall endpoint. Restore its source outline or reset the wall’s settings.');
    return piece.offset + distance(piece.start, point);
  };
  return { total, at };
}

/** Bind from the complete logical face, rather than independently fitting
 * each convex cell/portal fragment. Cut endpoints share one texture column. */
export function repeatWallSurface(document: LevelDocument, room: SourceRoom, settings: WallSettings, wall: WallSurface, points: readonly Position[]): WallSurface {
  const repeat = settings.repeat!;
  let length: number, from: number, to: number;
  if ('boundary' in settings) {
    const chain = wallChain(room, index => room.boundaryMaterials?.edges[index] === settings.boundary);
    length = chain.total; from = chain.at(points[wall.points[0]]!); to = chain.at(points[wall.points[1]]!);
    // The authored boundary's material can retain the opposite native winding.
    if (to < from) { from = length - from; to = length - to; }
  } else {
    const template = document.zones.find(zone => zone.id === room.recordZone)!;
    const original = (settings.upper ? template.upper : template.lower).find(record => record.offset === settings.record)?.wall;
    if (!original) throw new Error('Wall repeat has no straight source face. Reset its wall settings.');
    const a = document.points[original.points[0]]!, b = document.points[original.points[1]]!;
    const lineId = template.boundaryIds.find(id => {
      const line = document.lines[id]!; return on(a, line, { x: line.x + line.dx, z: line.z + line.dz }) && on(b, line, { x: line.x + line.dx, z: line.z + line.dz });
    });
    if (lineId !== undefined) {
      const line = document.lines[lineId]!, chain = wallChain(room, index => room.wallSources[index] === lineId);
      const fraction = (p: Position) => ((p.x - line.x) * line.dx + (p.z - line.z) * line.dz) / (line.dx ** 2 + line.dz ** 2);
      const start = fraction(a) * chain.total, end = fraction(b) * chain.total, direction = end > start ? 1 : -1;
      length = Math.abs(end - start); from = direction * (chain.at(points[wall.points[0]]!) - start); to = direction * (chain.at(points[wall.points[1]]!) - start);
    } else {
      const offset = nativeRoomTranslation(document, room);
      if (!offset) throw new Error('Decorative wall repeat needs an explicit translated source face. Restore its geometry before applying repeat.');
      const start = { x: a.x + offset.x, z: a.z + offset.z }, end = { x: b.x + offset.x, z: b.z + offset.z };
      if (!wall.points.every(id => on(points[id]!, start, end))) throw new Error('Decorative wall repeat lost its source span. Reset its wall settings.');
      length = distance(a, b); from = distance(start, points[wall.points[0]]!); to = distance(start, points[wall.points[1]]!);
    }
  }
  const columns = wallRepeatColumns(length, repeat);
  const epsilon = 1e-7;
  if (from < -epsilon || to > length + epsilon || to <= from) throw new Error('Wall repeat has a reversed or out-of-range generated span. Restore the source wall before applying repeat.');
  return wallForSpan({ ...wall, stripStart: 0, stripEnd: columns, horizontalMask: repeat.width - 1 }, length, wall.points, Math.max(0, from), Math.min(length, to));
}
