/* leveld.asc AUTOLINK and CPTNEARTOZONE: visual links follow geometric
 * adjoining outlines, independent of wall types, heights or physical links.
 * Convex clipping extends that source rule to modern concave outlines. */
import { buildRoomEdges, convexParts, type Position } from './geometry';
import type { LevelDocument } from './level';

export function navigationSightlines(document: Pick<LevelDocument, 'roomSource'>) {
  const rooms = document.roomSource.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] }));
  const parts = rooms.flatMap(room => convexParts(room.vertices).map(vertices => ({ room: room.id, vertices })));
  const edges = buildRoomEdges(rooms);
  const cross = (a: Position, b: Position) => a.x * b.z - a.z * b.x;
  return (from: Position, to: Position): boolean => {
    const delta = { x: to.x - from.x, z: to.z - from.z };
    const intervals = parts.flatMap(part => {
      let lo = 0, hi = 1;
      for (let i = 0; i < part.vertices.length; i++) {
        const a = part.vertices[i]!, b = part.vertices[(i + 1) % part.vertices.length]!;
        const edge = { x: b.x - a.x, z: b.z - a.z };
        const start = cross(edge, { x: from.x - a.x, z: from.z - a.z }), end = start + cross(edge, delta);
        if (start > 0 && end > 0) return [];
        if (start <= 0 && end <= 0) continue;
        const at = start / (start - end);
        if (start > 0) lo = Math.max(lo, at); else hi = Math.min(hi, at);
        if (lo >= hi) return [];
      }
      return lo < hi ? [{ room: part.room, lo, hi }] : [];
    });
    const cuts = [...new Set([0, 1, ...intervals.flatMap(interval => [interval.lo, interval.hi])])].sort((a, b) => a - b);
    let previous: string | undefined;
    for (let i = 0; i < cuts.length - 1; i++) {
      const middle = (cuts[i]! + cuts[i + 1]!) / 2;
      const covering = intervals.filter(interval => interval.lo <= middle && interval.hi >= middle);
      const current = covering.find(interval => interval.room === previous)?.room ?? covering[0]?.room;
      if (current === undefined) return false;
      if (previous !== undefined && current !== previous) {
        const t = cuts[i]!;
        const adjacent = edges.some(edge => {
          if (edge.room !== previous || edge.neighbour !== current) return false;
          const direction = { x: edge.end.x - edge.start.x, z: edge.end.z - edge.start.z }, determinant = cross(delta, direction);
          if (determinant === 0) return false;
          const offset = { x: edge.start.x - from.x, z: edge.start.z - from.z };
          const at = cross(offset, direction) / determinant, along = cross(offset, delta) / determinant;
          return Math.abs(at - t) < 1e-10 && along >= 0 && along <= 1;
        });
        if (!adjacent) return false; // corner contact alone does not join rooms
      }
      previous = current;
    }
    return previous !== undefined;
  };
}
