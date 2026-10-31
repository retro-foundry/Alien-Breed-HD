import { checkedInteger } from './binary';
import type { Position } from './geometry';
import type { RoomTopology } from './roomTopology';
import { BinaryWriter } from './writer';

export interface CollisionSegment { start: Position; end: Position; neighbour: number }
export interface CollisionGeometry {
  bytes: Uint8Array;
  /** One primary collision-line index per convex-cell boundary segment. */
  boundaries: number[][];
  cornerWalls: number[][];
}
export interface CollisionBoundary { points: readonly number[]; lines: readonly number[]; neighbours: readonly number[] }

/** LEVELSAVE's PW/PCW pass and ZoneBorders writer (leveld.asc 1044–1102).
 * Extra walls belong to adjoining solid corners; using every visible wall
 * here would incorrectly fence players into otherwise valid portal rooms. */
export function buildCornerWallLists(boundaries: readonly CollisionBoundary[]): number[][] {
  const wallFrom = new Map<number, number>(), previous = new Map<number, number>(), next = new Map<number, number>();
  for (const boundary of boundaries) {
    if (boundary.points.length < 3 || boundary.points.length !== boundary.lines.length || boundary.points.length !== boundary.neighbours.length)
      throw new Error('A corner-wall boundary is incomplete.');
    boundary.points.forEach((point, index) => {
      if (boundary.neighbours[index] !== -1) return;
      const end = boundary.points[(index + 1) % boundary.points.length]!;
      wallFrom.set(point, boundary.lines[index]!); previous.set(end, point); next.set(point, end);
    });
  }
  return boundaries.map(boundary => {
    const walls: number[] = [];
    boundary.points.forEach((point, index) => {
      if (!wallFrom.has(point)) return;
      if (boundary.points[(index + boundary.points.length - 1) % boundary.points.length] !== previous.get(point)) {
        const prior = previous.get(point), wall = prior === undefined ? undefined : wallFrom.get(prior);
        if (wall === undefined) throw new Error('A solid corner has no incoming wall. Check disconnected or overlapping room outlines.');
        walls.push(wall);
      }
      if (boundary.points[(index + 1) % boundary.points.length] !== next.get(point)) walls.push(wallFrom.get(point)!);
    });
    return walls;
  });
}

/** LEVELSAVE, leveld.asc 979–1038; consumed by ObjectMove.s and movement.c's
 * get_floorline_shift_offsets. Bytes 12/13 are signed displacement components,
 * not a word angle. AMOS conversion truncates towards zero: independently
 * checked against every shipped floor line, including non-axis-aligned walls.
 * Flags at 14 are runtime touch flags and start at zero in generated files.
 */
export function encodeCollisionSegments(segments: readonly CollisionSegment[]): Uint8Array {
  const output = new BinaryWriter();
  for (const { start, end, neighbour } of segments) {
    for (const point of [start, end]) {
      checkedInteger(point.x, -32768, 32767, 'Collision X coordinate');
      checkedInteger(point.z, -32768, 32767, 'Collision Z coordinate');
    }
    const dx = checkedInteger(end.x - start.x, -32768, 32767, 'Wall X span');
    const dz = checkedInteger(end.z - start.z, -32768, 32767, 'Wall Z span');
    const magnitude = Math.hypot(dx, dz);
    const length = checkedInteger(Math.trunc(magnitude), 1, 32767, 'Wall length');
    checkedInteger(neighbour, -1, 255, 'Connected room');
    const normalX = Math.trunc((dz - dx) * 20 / magnitude);
    const normalZ = Math.trunc(-(dx + dz) * 20 / magnitude);
    output.word(start.x); output.word(start.z); output.word(dx); output.word(dz);
    output.word(neighbour); output.word(length);
    output.byte(normalX & 255); output.byte(normalZ & 255); output.word(0);
  }
  return output.finish();
}

/** Collision table, primary boundaries and adjoining solid-corner walls for
 * generated convex cells. These lists have different runtime purposes. */
export function compileCollisionGeometry(topology: RoomTopology): CollisionGeometry {
  const segments: CollisionSegment[] = [], boundaries: number[][] = [];
  if (topology.cells.length > 256) throw new Error('The collision table exceeds the engine limit of 256 rooms.');
  for (const cell of topology.cells) {
    if (cell.vertices.length < 3 || cell.vertices.length !== cell.neighbours.length)
      throw new Error(`Room ${cell.sourceId} has an incomplete collision boundary.`);
    boundaries.push(cell.vertices.map((point, index) => {
      const start = topology.points[point], end = topology.points[cell.vertices[(index + 1) % cell.vertices.length]!];
      if (!start || !end) throw new Error(`Room ${cell.sourceId} references a missing collision point.`);
      const neighbour = cell.neighbours[index]!;
      if (neighbour >= topology.cells.length) throw new Error(`Room ${cell.sourceId} connects to a missing room.`);
      const id = checkedInteger(segments.length, 0, 32767, 'Collision line index');
      segments.push({ start, end, neighbour }); return id;
    }));
  }
  const cornerWalls = buildCornerWallLists(topology.cells.map((cell, index) => ({ points: cell.vertices,
    lines: boundaries[index]!, neighbours: cell.neighbours })));
  return { bytes: encodeCollisionSegments(segments), boundaries, cornerWalls };
}
