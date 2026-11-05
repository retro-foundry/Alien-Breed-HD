import type { LevelDocument } from './level';
import { compileRoomTopology } from './roomTopology';
import { buildDrawTopology } from './drawTopology';
import type { Position } from './geometry';
import { barrierBlocksWalking } from './importedBarriers';

export interface RoomConnectionEdge { room: string; start: Position; end: Position; neighbour: string | null; barrier?: true }

/** Use the export compiler's exterior boundaries and closed-wall rules.
 * Internal convex decomposition seams are not authored zone connections. */
export function roomConnectionEdges(document: LevelDocument): readonly RoomConnectionEdge[] {
  const topology = compileRoomTopology(document, document.roomSource);
  const draw = buildDrawTopology(document.roomSource, topology).topology;
  const owners = new Map(draw.cells.map(cell => [cell.id, cell.sourceId]));
  return draw.cells.flatMap(cell => cell.vertices.map((id, index) => {
    const neighbour = cell.neighbours[index]! < 0 ? null : owners.get(cell.neighbours[index]!);
    if (neighbour === undefined) throw new Error('A zone connection has no exterior room identity. Rebuild the room outline.');
    return {
      room: cell.sourceId, start: draw.points[id]!, end: draw.points[cell.vertices[(index + 1) % cell.vertices.length]!]!,
      neighbour,
      ...(neighbour && barrierBlocksWalking(document, document.roomSource.rooms.find(room => room.id === cell.sourceId)!, document.roomSource.rooms.find(room => room.id === neighbour)!, draw.points[id]!, draw.points[cell.vertices[(index + 1) % cell.vertices.length]!]!) ? { barrier: true as const } : {}),
    };
  }));
}
