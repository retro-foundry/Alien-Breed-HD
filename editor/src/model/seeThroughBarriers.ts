import { BinaryReader } from './binary';
import { sourceBoundaryIndex } from './boundaryMaterials';
import type { CellGraphics } from './cellGraphics';
import { buildDrawTopology } from './drawTopology';
import type { Position } from './geometry';
import { readGraphicRecord } from './graphics';
import type { LevelDocument } from './level';
import { portalBarrierAt } from './portalStates';
import { barrierBlocksWalking } from './importedBarriers';
import { roomSettings } from './roomHeights';
import type { RoomTopology } from './roomTopology';
import { wallForSpan } from './surfaceGeometry';
import { encodeWallSurface } from './surfaces';

const on = (p: Position, a: Position, b: Position) => (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
  && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);

/** AMOS WT 3/4: visible neighbour, solid collision line (-1), opcode 13.
 * Visibility keeps the original topology; only the collision projection closes.
 */
export function barrierCollisionTopology(document: LevelDocument, topology: RoomTopology): RoomTopology {
  return { ...topology, cells: topology.cells.map(cell => {
    const room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!;
    return { ...cell, neighbours: cell.neighbours.map((id, edge) => id >= 0 && barrierBlocksWalking(document, room,
      document.roomSource.rooms.find(room => room.id === topology.cells.find(other => other.id === id)!.sourceId)!,
      topology.points[cell.vertices[edge]!]!, topology.points[cell.vertices[(edge + 1) % cell.vertices.length]!]!) ? -1 : id) };
  }) };
}

export function authorSeeThroughBarriers(document: LevelDocument, graphics: CellGraphics): CellGraphics {
  if (!document.roomSource.rooms.some(room => room.portals?.some(state => state.barrier))) return graphics;
  const draw = buildDrawTopology(document.roomSource, graphics.topology), streams = new Map(graphics.streams), references = new Map(graphics.pointReferences);
  for (const cell of graphics.topology.cells) {
    const room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!;
    const owner = draw.topology.cells.find(item => item.id === draw.owners.get(cell.id))!;
    const edges = owner.neighbours.flatMap((id, edge) => {
      if (id < 0) return [];
      const a = owner.vertices[edge]!, b = owner.vertices[(edge + 1) % owner.vertices.length]!, other = draw.topology.cells.find(other => other.id === id)!;
      return portalBarrierAt(room, graphics.topology.points[a]!, graphics.topology.points[b]!, other.sourceId) ? [{ a, b }] : [];
    });
    if (!edges.length) continue;
    const appearance = document.roomAppearances?.find(item => item.room === room.id);
    if (!appearance) throw new Error(`Room ${room.id} needs a wall appearance and lighting for its see-through barrier.`);
    const settings = roomSettings(document, room), stream = streams.get(cell.id)!;
    const layer = (items: typeof stream.lower, upper: boolean) => {
      const active = !upper || settings.upper.length && settings.upperFloor > settings.upperCeiling;
      if (!active) return items;
      const offset = upper ? room.upperStorey ? room.upperStorey.wall : appearance.upperWall : appearance.lowerWall;
      const material = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === offset)?.wall;
      if (!material) throw new Error(`Room ${room.id} needs a ${upper ? 'upper' : 'lower'} wall style for its see-through barrier.`);
      const retained = items.filter(item => {
        const wall = readGraphicRecord(new BinaryReader(item.bytes, 'barrier wall replacement'), 0, graphics.topology.points.length).wall;
        return !wall || !edges.some(({ a, b }) => wall.points.every(id => on(graphics.topology.points[id]!, graphics.topology.points[a]!, graphics.topology.points[b]!)));
      });
      const additions = edges.map(({ a, b }) => {
        const start = graphics.topology.points[a]!, end = graphics.topology.points[b]!;
        const originalStart = document.points[material.points[0]]!, originalEnd = document.points[material.points[1]]!;
        const boundaryIndex = sourceBoundaryIndex(room, start, end);
        if (boundaryIndex < 0) throw new Error('A see-through barrier lost its source boundary identity. Restore its room outline.');
        const identity = room.boundaryMaterials!.edges[boundaryIndex]!;
        const first = room.vertices.findIndex((_, index) => room.boundaryMaterials!.edges[index] === identity && room.boundaryMaterials!.edges[(index + room.vertices.length - 1) % room.vertices.length] !== identity);
        if (first < 0) throw new Error('A see-through barrier has no ordered source edge chain. Restore its outline.');
        let offset = 0;
        for (let index = first; index !== boundaryIndex; index = (index + 1) % room.vertices.length) {
          if (room.boundaryMaterials!.edges[index] !== identity) throw new Error('A see-through barrier has a disconnected source edge chain. Restore its outline.');
          const from = room.vertices[index]!, to = room.vertices[(index + 1) % room.vertices.length]!;
          offset += Math.hypot(to.x - from.x, to.z - from.z);
        }
        offset += Math.hypot(start.x - room.vertices[boundaryIndex]!.x, start.z - room.vertices[boundaryIndex]!.z);
        const span = wallForSpan({ ...material, type: 13, topFixed: (upper ? settings.upperCeiling : settings.ceiling) * 256,
          bottomFixed: (upper ? settings.upperFloor : settings.floor) * 256 }, Math.hypot(originalEnd.x - originalStart.x, originalEnd.z - originalStart.z), [a, b], offset, offset + Math.hypot(end.x - start.x, end.z - start.z));
        return { bytes: encodeWallSurface(span), boundaryIndex, lightingPoint: upper && room.upperStorey ? room.upperStorey.lightingPoint : appearance.lightingPoint };
      });
      // Match original LEVELSAVE's wall batch before floor/ceiling commands.
      const insertion = retained.findIndex(item => readGraphicRecord(new BinaryReader(item.bytes, 'barrier insertion'), 0, graphics.topology.points.length).polygon);
      const at = insertion < 0 ? retained.length : insertion;
      return [...retained.slice(0, at), ...additions, ...retained.slice(at)];
    };
    const lower = layer(stream.lower, false), upper = stream.upper ? layer(stream.upper, true) : undefined;
    streams.set(cell.id, { lower, ...(upper ? { upper } : {}) });
    references.set(cell.id, [...new Set([...lower, ...(upper ?? [])].flatMap(item => readGraphicRecord(new BinaryReader(item.bytes, 'barrier point references'), 0, graphics.topology.points.length).points))]);
  }
  return { ...graphics, streams, pointReferences: references };
}
