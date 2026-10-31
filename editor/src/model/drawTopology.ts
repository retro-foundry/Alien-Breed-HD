import type { RoomSource } from './roomSource';
import type { RoomTopology } from './roomTopology';

/** Recover authored draw-volume boundaries from convex collision cells.
 * Internal decomposition edges have no physical wall or separate draw volume.
 * Runtime draw identities stay explicit in the exported group setup. */
export function buildDrawTopology(source: RoomSource, topology: RoomTopology) {
  const owners = new Map<number, number>();
  const cells = source.rooms.flatMap(room => {
    const members = topology.cells.filter(cell => cell.sourceId === room.id);
    if (!members.length) return [];
    members.forEach(cell => owners.set(cell.id, members[0]!.id));
    if (members.length === 1) return members;
    const ids = new Set(members.map(cell => cell.id));
    const edges = members.flatMap(cell => cell.vertices.flatMap((a, index) => ids.has(cell.neighbours[index]!) ? [] : [{ a,
      b: cell.vertices[(index + 1) % cell.vertices.length]!, neighbour: cell.neighbours[index]!, source: cell.sourceLines[index]! }]));
    const byStart = new Map(edges.map(edge => [edge.a, edge]));
    if (byStart.size !== edges.length) throw new Error(`Room ${room.id} has an ambiguous draw boundary.`);
    const first = edges.find(edge => topology.points[edge.a]!.x === room.vertices[0]!.x && topology.points[edge.a]!.z === room.vertices[0]!.z);
    if (!first) throw new Error(`Room ${room.id} lost its draw boundary start.`);
    const ordered: typeof edges = []; let edge = first;
    do {
      ordered.push(edge); const next = byStart.get(edge.b);
      if (!next || ordered.length > edges.length) throw new Error(`Room ${room.id} has an incomplete draw boundary.`);
      edge = next;
    } while (edge !== first);
    if (ordered.length !== edges.length) throw new Error(`Room ${room.id} has disconnected draw boundaries.`);
    return [{ ...members[0]!, vertices: ordered.map(edge => edge.a), neighbours: ordered.map(edge => edge.neighbour), sourceLines: ordered.map(edge => edge.source) }];
  });
  return { topology: { ...topology, cells }, owners };
}
