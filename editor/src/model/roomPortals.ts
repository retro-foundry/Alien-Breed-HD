import { buildRoomEdges, type Position } from './geometry';
import { restoreBoundaryMaterials, sourceBoundaryIndex } from './boundaryMaterials';
import { restorePortalStates } from './portalStates';
import { barrierBlocksWalking } from './importedBarriers';
import type { LevelDocument } from './level';
import type { SourceRoom } from './roomSource';
import { assertStaticRoom } from './roomStoreys';
import { roomConnectionEdges } from './roomConnections';
import { setRoomAppearance, type RoomAppearance } from './roomAppearance';
import { buildPlayableLevel } from './playableBuild';
import type { EditCommand } from './commands';

function boundaries(room: SourceRoom) {
  return room.boundaryMaterials ?? restoreBoundaryMaterials({ nextId: room.vertices.length,
    edges: room.vertices.map((_, index) => index), overrides: [] }, room.vertices.length);
}

export interface RoomPortalTarget {
  neighbour: string; edges: readonly number[]; otherEdges: readonly number[];
  segments: readonly { start: Position; end: Position }[]; open: boolean | null; authored: boolean; barrier: boolean;
}

/** A logical boundary can be split at T-junctions or inserted corners. Preserve
 * its stable identity and update every reciprocal segment in the same command. */
export function roomPortalTargets(document: LevelDocument, id: string): readonly RoomPortalTarget[] {
  const room = document.roomSource.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select a room before inspecting wall connections.');
  const raw = buildRoomEdges(document.roomSource.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  const own = boundaries(room), compiled = roomConnectionEdges(document);
  const targets: RoomPortalTarget[] = [];
  for (const neighbour of new Set(raw.filter(edge => edge.room === id && edge.neighbour !== null).map(edge => edge.neighbour!))) {
    const other = document.roomSource.rooms.find(room => room.id === neighbour)!, theirs = boundaries(other);
    const pairs = raw.filter(edge => edge.room === id && edge.neighbour === neighbour).map(edge => ({ ...edge,
      a: sourceBoundaryIndex(room, edge.start, edge.end), b: sourceBoundaryIndex(other, edge.start, edge.end) }));
    const handled = new Set<number>();
    for (const pair of pairs) {
      if (handled.has(own.edges[pair.a]!)) continue;
      const a = new Set([own.edges[pair.a]!]), b = new Set<number>();
      let previous = -1;
      while (previous !== a.size + b.size) {
        previous = a.size + b.size;
        for (const part of pairs) if (a.has(own.edges[part.a]!) || b.has(theirs.edges[part.b]!)) { a.add(own.edges[part.a]!); b.add(theirs.edges[part.b]!); }
      }
      a.forEach(value => handled.add(value));
      const segments = pairs.filter(part => a.has(own.edges[part.a]!));
      const states = compiled.filter(edge => edge.room === id && segments.some(part => sourceBoundaryIndex({ vertices: [part.start, part.end] }, edge.start, edge.end) >= 0)).map(edge => edge.neighbour === neighbour);
      if (!states.length) throw new Error('A shared wall has no compiled exterior segments. Rebuild its room outline.');
      targets.push({ neighbour, edges: [...new Set(segments.map(part => part.a))], otherEdges: [...new Set(segments.map(part => part.b))], segments,
        barrier: segments.every(part => barrierBlocksWalking(document, room, other, part.start, part.end)),
        open: states.every(Boolean) ? true : states.every(value => !value) ? false : null,
        authored: !!room.portals?.some(item => a.has(item.boundary) && item.neighbour === neighbour) });
    }
  }
  return targets;
}

/** AMOS DEFWALL toggles ordinary WT 0/1 boundaries, excluding controlled walls.
 * Current export uses reciprocal collision neighbours and rebuilds real wall
 * bands, visibility and clips through the existing source compiler. */
export function setRoomPortal(document: LevelDocument, id: string, edge: number, neighbour: string, open: boolean | 'barrier' | null,
  appearances: readonly RoomAppearance[] = []): EditCommand {
  const room = document.roomSource.rooms.find(room => room.id === id), other = document.roomSource.rooms.find(room => room.id === neighbour);
  if (!room || !other || room.id === other.id) throw new Error('Choose two different adjoining rooms.');
  assertStaticRoom(document, room); assertStaticRoom(document, other);
  const target = roomPortalTargets(document, id).find(target => target.neighbour === neighbour && target.edges.includes(edge));
  if (!target) throw new Error('These rooms no longer share this wall. Select a current wall connection.');
  const update = (room: SourceRoom, indices: readonly number[], neighbour: string) => {
    const boundaryMaterials = boundaries(room), ids = new Set(indices.map(index => boundaryMaterials.edges[index]!));
    const retained = (room.portals ?? []).filter(item => item.neighbour !== neighbour || !ids.has(item.boundary));
    const portals = restorePortalStates([...retained, ...(open === null ? [] : [...ids].map(boundary => ({ boundary, neighbour, open: open === 'barrier' ? true : open, ...(open === 'barrier' ? { barrier: true } : {}) })))], boundaryMaterials.nextId);
    const { portals: _previous, ...unchanged } = room;
    return Object.freeze({ ...unchanged, boundaryMaterials, ...(portals.length ? { portals } : {}) });
  };
  const updated = new Map([[id, update(room, target.edges, neighbour)], [neighbour, update(other, target.otherEdges, id)]]);
  const roomSource = Object.freeze({ ...document.roomSource, rooms: Object.freeze(document.roomSource.rooms.map(room => updated.get(room.id) ?? room)) });
  let working = { ...document, roomSource };
  for (const appearance of appearances) {
    if (!updated.has(appearance.room)) throw new Error('Wall repair appearance must belong to one of the two adjoining rooms.');
    working = { ...working, roomAppearances: setRoomAppearance(working, appearance) };
  }
  buildPlayableLevel(working, -1);
  return { label: open === null ? 'Restore automatic wall connection' : open === 'barrier' ? 'Create see-through barrier' : open ? 'Open wall connection' : 'Close wall connection', patches: [],
    roomSource: { before: document.roomSource, after: working.roomSource }, roomAppearances: { before: document.roomAppearances, after: working.roomAppearances } };
}
