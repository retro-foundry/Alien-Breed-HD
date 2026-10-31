/* Convert legacy runtime boundaries into convex visibility cells. Concave
 * originals retain their zone identities and original game bytes; internal
 * cell edges exist only for sightline computation. New authored topology is
 * compiled into its own convex runtime zones by the geometry compiler. */
import { buildRoomEdges, convexParts, type Position } from './geometry';
import type { LevelDocument } from './level';
import type { VisibilityRoom } from './visibilityGeometry';

export function visibilityCells(document: LevelDocument): { points: Position[]; rooms: VisibilityRoom[]; owners: Map<number, number> } {
  const anchors = document.zones.flatMap(zone => zone.boundaryIds.map(id => { const line = document.lines[id]!; return { x: line.x, z: line.z }; }));
  const cells = document.zones.flatMap(zone => convexParts(zone.boundaryIds.map(id => {
    const line = document.lines[id]!; return { x: line.x, z: line.z };
  })).map(vertices => ({ owner: zone.id, vertices })));
  const edges = buildRoomEdges(cells.map((cell, id) => ({ id: String(id), vertices: cell.vertices })), anchors);
  const byCoordinate = new Map<string, number>();
  for (const point of document.points) { const key = `${point.x},${point.z}`; if (!byCoordinate.has(key)) byCoordinate.set(key, point.id); }
  const pointId = (point: Position) => {
    const id = byCoordinate.get(`${point.x},${point.z}`);
    if (id === undefined) throw new Error(`A visibility corner at ${point.x}, ${point.z} is missing from the original point table.`);
    return id;
  };
  const contains = (point: Position, line: LevelDocument['lines'][number]) =>
    (point.x - line.x) * line.dz === (point.z - line.z) * line.dx
    && point.x >= Math.min(line.x, line.x + line.dx) && point.x <= Math.max(line.x, line.x + line.dx)
    && point.z >= Math.min(line.z, line.z + line.dz) && point.z <= Math.max(line.z, line.z + line.dz);
  const owners = new Map(cells.map((cell, id) => [id, cell.owner]));
  const rooms = cells.map((cell, id): VisibilityRoom => {
    const boundary = edges.filter(edge => edge.room === String(id)), source = document.zones.find(zone => zone.id === cell.owner)!;
    return { id, vertices: boundary.map(edge => pointId(edge.start)), neighbours: boundary.map(edge => {
      const original = source.boundaryIds.map(id => document.lines[id]!).find(line => contains(edge.start, line) && contains(edge.end, line));
      if (original?.connect === -1) return -1;
      if (edge.neighbour === null) throw new Error(`Zone ${source.id} has an unmatched visibility portal.`);
      const neighbour = Number(edge.neighbour), destination = owners.get(neighbour)!;
      if (original && destination !== original.connect) throw new Error(`Zone ${source.id} has a portal whose boundary disagrees with destination ${original.connect}.`);
      if (!original && destination !== cell.owner) throw new Error(`Zone ${source.id} has an internal visibility edge outside its source boundary.`);
      return neighbour;
    }) };
  });
  return { points: document.points, rooms, owners };
}
