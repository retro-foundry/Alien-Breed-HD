import { checkedInteger } from './binary';
import { sourceBoundaryIndex } from './boundaryMaterials';
import type { Position } from './geometry';
import type { SourceRoom } from './roomSource';
import type { RoomSource } from './roomSource';
import { buildRoomEdges } from './geometry';

export interface PortalState { readonly boundary: number; readonly neighbour: string; readonly open: boolean; readonly barrier?: true }

export function restorePortalStates(value: unknown, nextBoundary: number | undefined): readonly PortalState[] {
  if (!Array.isArray(value) || value.length > 8192 || nextBoundary === undefined) throw new Error('Wall connections require boundary identities and a bounded override list.');
  const seen = new Set<string>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).some(key => !['boundary', 'neighbour', 'open', 'barrier'].includes(key)) || typeof item.boundary !== 'number'
      || typeof item.neighbour !== 'string' || !item.neighbour || item.neighbour.length > 128 || typeof item.open !== 'boolean') throw new Error('Invalid saved wall connection.');
    if ('barrier' in item && (item.barrier !== true || item.open !== true)) throw new Error('A see-through barrier needs an open visibility connection.');
    const boundary = checkedInteger(item.boundary, 0, nextBoundary - 1, 'Wall connection boundary'), key = `${boundary}:${item.neighbour}`;
    if (seen.has(key)) throw new Error('A wall connection has duplicate states.'); seen.add(key);
    return Object.freeze({ boundary, neighbour: item.neighbour, open: item.open, ...(item.barrier ? { barrier: true as const } : {}) });
  }));
}

export function portalBarrierAt(room: SourceRoom, start: Position, end: Position, neighbour: string): boolean {
  const index = sourceBoundaryIndex(room, start, end);
  return index >= 0 && !!room.portals?.some(item => item.boundary === room.boundaryMaterials?.edges[index] && item.neighbour === neighbour && item.barrier);
}

export function portalStateAt(room: SourceRoom, start: Position, end: Position, neighbour: string): boolean | undefined {
  if (!room.portals?.length || !room.boundaryMaterials) return;
  const index = sourceBoundaryIndex(room, start, end);
  if (index < 0) return;
  return room.portals.find(item => item.boundary === room.boundaryMaterials!.edges[index] && item.neighbour === neighbour)?.open;
}

/** A shaft/skylight replaces one outline with several source rooms. Route
 * outside neighbours' explicit states to the actual new boundary owners. */
export function partitionPortalStates(source: RoomSource, replaced: string, members: readonly string[]): RoomSource {
  if (!source.rooms.some(room => room.portals?.some(state => state.neighbour === replaced))) return source;
  const ids = new Set(members), edges = buildRoomEdges(source.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  const rooms = source.rooms.map(room => {
    if (ids.has(room.id) || !room.portals?.some(state => state.neighbour === replaced)) return room;
    const states = room.portals.flatMap(state => state.neighbour !== replaced ? [state] : edges.filter(edge => edge.room === room.id && edge.neighbour !== null && ids.has(edge.neighbour)
      && room.boundaryMaterials?.edges[sourceBoundaryIndex(room, edge.start, edge.end)] === state.boundary).map(edge => ({ ...state, neighbour: edge.neighbour! })));
    const portals = restorePortalStates([...new Map(states.map(state => [`${state.boundary}:${state.neighbour}`, state])).values()], room.boundaryMaterials?.nextId);
    const { portals: _previous, ...unchanged } = room;
    return Object.freeze({ ...unchanged, ...(portals.length ? { portals } : {}) });
  });
  return Object.freeze({ ...source, rooms: Object.freeze(rooms) });
}
