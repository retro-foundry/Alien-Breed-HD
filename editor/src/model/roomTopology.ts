import { buildRoomEdges, convexParts, type Position } from './geometry';
import type { LevelDocument } from './level';
import { sourceWallForEdge, type RoomSource, type SourceRoom } from './roomSource';
import type { VisibilityRoom } from './visibilityGeometry';
import { restoreRoomHeights, type RoomHeights } from './roomHeights';
import { hasNativeWallMaterials } from './roomMaterials';
import { restoreRoomProperties, type RoomProperties } from './roomProperties';

export interface CompiledRoomCell extends VisibilityRoom {
  sourceId: string;
  recordZone: number;
  originZone: number | null;
  heights?: RoomHeights;
  properties?: RoomProperties;
  /** Original wall records supply materials; null denotes a new/internal edge. */
  sourceLines: (number | null)[];
}
export interface RoomTopology {
  points: Position[];
  cells: CompiledRoomCell[];
  sourceCells: Map<string, number[]>;
}

/** Compile source outlines into the current engine's convex portal cells.
 * This is the topology stage, not a game-file writer: graphics relocation,
 * dependency remapping and collision normals must be compiled separately.
 * Existing closed boundaries must not become portals merely because another
 * imported zone shares their coordinates (visibilityImport.ts's same rule).
 */
export function compileRoomTopology(document: LevelDocument, source: RoomSource): RoomTopology {
  if (!source.rooms.length) throw new Error('Create a room before building the level.');
  const records = new Map(document.zones.map(zone => [zone.id, zone]));
  const origins = new Set<number>(), ids = new Set<string>();
  const pieces: { room: SourceRoom; vertices: Position[] }[] = [];
  for (const room of source.rooms) {
    if (ids.has(room.id)) throw new Error('Each source room must have a unique identity.');
    ids.add(room.id);
    if (!records.has(room.recordZone)) throw new Error(`Room ${room.id} refers to missing surface record ${room.recordZone}.`);
    if (room.heights) restoreRoomHeights(room.heights, records.get(room.recordZone)!);
    if (room.properties) restoreRoomProperties(room.properties);
    if (room.wallSources.length !== room.vertices.length || room.wallSources.some(line => line !== null && !records.get(room.recordZone)!.boundaryIds.includes(line)))
      throw new Error(`Room ${room.id} has invalid wall-source references. Restore or reassign its wall materials before building.`);
    if (room.originZone !== null) {
      if (!records.has(room.originZone) || origins.has(room.originZone)) throw new Error(`Room ${room.id} has an invalid or duplicate original zone reference.`);
      origins.add(room.originZone);
    }
    for (const vertices of convexParts(room.vertices)) {
      pieces.push({ room, vertices });
      if (pieces.length > 256) throw new Error('This level needs more than the engine limit of 256 convex rooms. Simplify or divide the level.');
    }
  }
  const edges = buildRoomEdges(pieces.map((piece, id) => ({ id: String(id), vertices: piece.vertices })), source.rooms.flatMap(room => [...room.vertices]));
  const points: Position[] = document.points.map(({ x, z }) => ({ x, z }));
  const coordinates = new Map<string, number>();
  points.forEach((point, id) => { const key = `${point.x},${point.z}`; if (!coordinates.has(key)) coordinates.set(key, id); });
  const pointId = (point: Position) => {
    const key = `${point.x},${point.z}`, existing = coordinates.get(key);
    if (existing !== undefined) return existing;
    if (points.length >= 2048) throw new Error('This level exceeds the engine limit of 2,048 points. Remove unused geometry before building.');
    const id = points.length; points.push({ ...point }); coordinates.set(key, id); return id;
  };
  const sourceCells = new Map<string, number[]>();
  const cells = pieces.map(({ room }, id): CompiledRoomCell => {
    const boundary = edges.filter(edge => edge.room === String(id));
    if (boundary.length > 10) throw new Error(`Room ${room.id} has more than ten wall segments after joining its neighbours. Divide this outline before building.`);
    const sourceLines = boundary.map(edge => sourceWallForEdge(room, edge.start, edge.end));
    const neighbours = boundary.map((edge, index) => {
      if (edge.neighbour === null) return -1;
      const destination = Number(edge.neighbour), other = pieces[destination]!.room;
      const line = sourceLines[index] ?? null;
      if (line !== null && document.lines[line]!.connect === -1 && hasNativeWallMaterials(other.originZone, other.wallSources) && other.id !== room.id) return -1;
      return destination;
    });
    sourceCells.set(room.id, [...(sourceCells.get(room.id) ?? []), id]);
    return { id, sourceId: room.id, recordZone: room.recordZone, originZone: room.originZone, ...(room.heights ? { heights: room.heights } : {}), ...(room.properties ? { properties: room.properties } : {}),
      vertices: boundary.map(edge => pointId(edge.start)), neighbours, sourceLines };
  });
  // A one-sided closed portal is not a playable join. Report it at the source
  // boundary instead of emitting mismatched collision/visibility tables.
  for (const cell of cells) cell.neighbours.forEach((neighbour, index) => {
    if (neighbour < 0) return;
    const target = cells[neighbour]!, a = cell.vertices[index]!, b = cell.vertices[(index + 1) % cell.vertices.length]!;
    const reverse = target.vertices.findIndex((point, i) => point === b && target.vertices[(i + 1) % target.vertices.length] === a);
    if (reverse < 0 || target.neighbours[reverse] !== cell.id) throw new Error(`Rooms ${cell.sourceId} and ${target.sourceId} disagree about their shared doorway.`);
  });
  return { points, cells, sourceCells };
}
