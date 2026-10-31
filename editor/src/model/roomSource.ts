import { convexParts, normaliseRoomOutline, validateRoomInteriors, type Position } from './geometry';
import type { LevelDocument } from './level';
import { checkedInteger } from './binary';

/** Editable room identities survive compiler zone renumbering. The imported
 * record remains the authority for materials/mechanisms until those authoring
 * adapters are implemented; it is never reconstructed from guessed defaults. */
export interface SourceRoom {
  readonly id: string;
  readonly recordZone: number;
  readonly originZone: number | null;
  readonly vertices: readonly Position[];
  readonly wallSources: readonly (number | null)[];
}
export interface RoomSource { readonly rooms: readonly SourceRoom[]; readonly nextId: number }
export interface RoomChange { readonly label: string; readonly before: RoomSource; readonly after: RoomSource }
export interface RoomHistory { readonly source: RoomSource; readonly past: readonly RoomChange[]; readonly future: readonly RoomChange[] }
export interface RoomDraft { readonly base: RoomSource; readonly roomId: string | null; readonly recordZone: number; readonly vertices: readonly Position[] }

const samePoint = (a: Position, b: Position) => a.x === b.x && a.z === b.z;
const onSegment = (point: Position, a: Position, b: Position) => (point.x - a.x) * (b.z - a.z) === (point.z - a.z) * (b.x - a.x)
  && point.x >= Math.min(a.x, b.x) && point.x <= Math.max(a.x, b.x) && point.z >= Math.min(a.z, b.z) && point.z <= Math.max(a.z, b.z);

/** A split boundary retains the material/closure identity of its source wall.
 * Internal decomposition edges and genuinely new walls have no such identity. */
export function sourceWallForEdge(room: SourceRoom, start: Position, end: Position): number | null {
  const index = room.vertices.findIndex((a, i) => onSegment(start, a, room.vertices[(i + 1) % room.vertices.length]!)
    && onSegment(end, a, room.vertices[(i + 1) % room.vertices.length]!));
  return index < 0 ? null : room.wallSources[index]!;
}

function frozenRoom(id: string, recordZone: number, vertices: readonly Position[], originZone: number | null,
  wallSources: readonly (number | null)[] = vertices.map(() => null)): SourceRoom {
  if (wallSources.length !== vertices.length) throw new Error(`Room ${id} has an incomplete wall-source list.`);
  const outline = normaliseRoomOutline(vertices);
  const walls = outline.map((a, i) => {
    const b = outline[(i + 1) % outline.length]!;
    const index = vertices.findIndex((start, j) => {
      const end = vertices[(j + 1) % vertices.length]!;
      return (samePoint(a, start) && samePoint(b, end)) || (samePoint(a, end) && samePoint(b, start));
    });
    if (index < 0) throw new Error(`Room ${id} lost a wall identity while normalizing its outline.`);
    return wallSources[index]!;
  });
  return Object.freeze({ id, recordZone, originZone, vertices: Object.freeze(outline.map(point => Object.freeze(point))), wallSources: Object.freeze(walls) });
}
function source(rooms: readonly SourceRoom[], nextId: number): RoomSource {
  return Object.freeze({ rooms: Object.freeze([...rooms]), nextId });
}

type RoomRecords = Pick<LevelDocument, 'zones' | 'lines'>;
export function importRoomSource(document: RoomRecords): RoomSource {
  return source(document.zones.map(zone => frozenRoom(`room-${zone.id}`, zone.id,
    zone.boundaryIds.map(id => { const line = document.lines[id]!; return { x: line.x, z: line.z }; }), zone.id, zone.boundaryIds)), Math.max(-1, ...document.zones.map(zone => zone.id)) + 1);
}

export function roomSourceSnapshot(current: RoomSource) {
  return { format: 'alien-breed-room-source', version: 2, nextId: current.nextId, rooms: current.rooms };
}

export function roomSourceMatchesLevel(document: LevelDocument): boolean {
  return JSON.stringify(document.roomSource.rooms) === JSON.stringify(importRoomSource(document).rooms);
}

/** All editable source must match before using the imported runtime files. */
export function sourceMatchesLevel(document: LevelDocument): boolean {
  return !document.sourceStarts?.length && !document.sourceObjects?.length && roomSourceMatchesLevel(document);
}

/** Restore committed editable geometry, never a partially drawn gesture or
 * generated runtime cells. Original binary records remain the material and
 * dependency baseline; missing templates cannot be silently substituted. */
export function restoreRoomSource(document: RoomRecords, value: unknown): RoomSource {
  if (!value || typeof value !== 'object' || !('format' in value) || value.format !== 'alien-breed-room-source'
    || !('version' in value) || ![1, 2].includes(value.version as number) || !('rooms' in value) || !Array.isArray(value.rooms)
    || !('nextId' in value) || typeof value.nextId !== 'number') throw new Error('Unsupported or damaged room source.');
  if (Object.keys(value).some(key => !['format', 'version', 'rooms', 'nextId'].includes(key))) throw new Error('Room source contains unsupported fields; it has not been replaced or discarded.');
  const nextId = checkedInteger(value.nextId, 0, Number.MAX_SAFE_INTEGER - 1, 'Next room identity');
  const records = new Set(document.zones.map(zone => zone.id)), ids = new Set<string>(), origins = new Set<number>();
  const rooms = value.rooms.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('id' in item) || typeof item.id !== 'string' || !item.id || item.id.length > 128 || ids.has(item.id))
      throw new Error('Room source contains an invalid or duplicate room identity.');
    ids.add(item.id);
    if (Object.keys(item).some(key => !['id', 'recordZone', 'originZone', 'vertices', ...(value.version === 2 ? ['wallSources'] : [])].includes(key))) throw new Error(`Room ${item.id} contains unsupported source fields.`);
    const numericId = /^room-(0|[1-9]\d*)$/.exec(item.id)?.[1];
    if (numericId !== undefined && Number(numericId) >= nextId) throw new Error('The next room identity would reuse an existing room.');
    if (!('recordZone' in item) || typeof item.recordZone !== 'number' || !records.has(item.recordZone)) throw new Error(`Room ${item.id} has a missing surface record.`);
    if (!('originZone' in item) || (item.originZone !== null && (typeof item.originZone !== 'number' || !records.has(item.originZone) || origins.has(item.originZone))))
      throw new Error(`Room ${item.id} has a missing or duplicate original-zone reference.`);
    const origin = item.originZone as number | null;
    if (origin !== null) origins.add(origin);
    if (!('vertices' in item) || !Array.isArray(item.vertices) || item.vertices.length > 2048) throw new Error(`Room ${item.id} has an invalid corner list.`);
    const vertices = item.vertices.map((point: unknown) => {
      if (!point || typeof point !== 'object' || !('x' in point) || !('z' in point) || typeof point.x !== 'number' || typeof point.z !== 'number')
        throw new Error(`Room ${item.id} contains an invalid corner.`);
      if (Object.keys(point).some(key => key !== 'x' && key !== 'z')) throw new Error(`Room ${item.id} contains unsupported corner fields.`);
      return { x: point.x, z: point.z };
    });
    const template = document.zones.find(zone => zone.id === item.recordZone)!;
    let walls: (number | null)[];
    if (value.version === 2) {
      if (!('wallSources' in item) || !Array.isArray(item.wallSources) || item.wallSources.length !== vertices.length)
        throw new Error(`Room ${item.id} has an incomplete wall-source list.`);
      walls = item.wallSources.map((line: unknown) => {
        if (line !== null && (typeof line !== 'number' || !Number.isInteger(line) || !template.boundaryIds.includes(line)))
          throw new Error(`Room ${item.id} refers to a wall outside its settings template.`);
        return line as number | null;
      });
    } else {
      // Older saves have no edge identity. Recover only exact geometric
      // evidence; never guess which material belonged to an already moved edge.
      walls = vertices.map((start, i) => {
        if (origin === null) return null;
        const end = vertices[(i + 1) % vertices.length]!;
        return template.boundaryIds.find(id => {
          const line = document.lines[id]!, a = { x: line.x, z: line.z }, b = { x: line.x + line.dx, z: line.z + line.dz };
          return onSegment(start, a, b) && onSegment(end, a, b);
        }) ?? null;
      });
    }
    return frozenRoom(item.id, item.recordZone, vertices, origin, walls);
  });
  return source(rooms, nextId);
}

/** Drafts are separate from committed source: Escape/tool changes simply drop
 * the draft. A record zone must be selected explicitly when creating a room. */
export function beginRoomDraft(document: LevelDocument, current: RoomSource, recordZone: number, roomId: string | null = null): RoomDraft {
  if (!document.zones.some(zone => zone.id === recordZone)) throw new Error('Choose an existing room whose surface settings can be used.');
  const room = roomId === null ? undefined : current.rooms.find(room => room.id === roomId);
  if (roomId !== null && !room) throw new Error('That room no longer exists.');
  if (room && room.recordZone !== recordZone) throw new Error('The selected room does not use that source record.');
  return Object.freeze({ base: current, roomId, recordZone, vertices: room?.vertices ?? Object.freeze([]) });
}

export function setDraftCorners(draft: RoomDraft, vertices: readonly Position[]): RoomDraft {
  // Partial gestures are allowed, but invalid coordinates and unbounded input
  // are rejected before copying. Full polygon validation happens on commit.
  if (vertices.length > 2048) throw new Error('A room has too many corners for the game engine.');
  for (const point of vertices) if (!Number.isInteger(point.x) || !Number.isInteger(point.z)
    || point.x < -32768 || point.x > 32767 || point.z < -32768 || point.z > 32767)
    throw new Error('Room corners must sit on whole coordinates within the level bounds.');
  return Object.freeze({ ...draft, vertices: Object.freeze(vertices.map(point => Object.freeze({ ...point }))) });
}

export function commitRoomDraft(current: RoomSource, draft: RoomDraft): RoomChange {
  if (current !== draft.base) throw new Error('The rooms changed while this outline was being drawn. No changes were saved.');
  const existing = draft.roomId === null ? undefined : current.rooms.find(room => room.id === draft.roomId);
  if (!existing) checkedInteger(current.nextId, 0, Number.MAX_SAFE_INTEGER - 2, 'Next room identity');
  if (draft.roomId !== null && !existing) throw new Error('That room no longer exists.');
  const id = existing?.id ?? `room-${current.nextId}`;
  const walls = draft.vertices.map((start, i) => existing ? sourceWallForEdge(existing, start, draft.vertices[(i + 1) % draft.vertices.length]!) : null);
  const room = frozenRoom(id, draft.recordZone, draft.vertices, existing?.originZone ?? null, walls);
  if (existing && existing.recordZone === room.recordZone && existing.vertices.length === room.vertices.length
    && existing.vertices.every((point, i) => point.x === room.vertices[i]!.x && point.z === room.vertices[i]!.z))
    return Object.freeze({ label: 'Edit room outline', before: current, after: current });
  // Enforce the actual convex-cell corner contract before accepting a command.
  // Concavity and winding remain editor responsibilities, not user settings.
  convexParts(room.vertices);
  const rooms = existing ? current.rooms.map(value => value.id === id ? room : value) : [...current.rooms, room];
  validateRoomInteriors(rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  return Object.freeze({ label: existing ? 'Edit room outline' : 'Create room', before: current,
    after: source(rooms, current.nextId + (existing ? 0 : 1)) });
}

/** Original authoring points are shared by adjoining rooms. Move the whole
 * incident portal fan atomically, including a corner inserted at a T join. */
export function moveSourceCorner(current: RoomSource, roomId: string, index: number, point: Position): RoomChange {
  const room = current.rooms.find(room => room.id === roomId), original = room?.vertices[index];
  if (!room || !original) throw new Error('That room corner no longer exists.');
  if (original.x === point.x && original.z === point.z) return Object.freeze({ label: 'Move shared room corner', before: current, after: current });
  const edges = validateRoomInteriors(current.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  const same = (value: Position) => value.x === original.x && value.z === original.z;
  const affected = new Set([roomId]);
  for (;;) {
    const count = affected.size;
    for (const edge of edges) if (affected.has(edge.room) && edge.neighbour !== null && (same(edge.start) || same(edge.end))) affected.add(edge.neighbour);
    if (count === affected.size) break;
  }
  const rooms = current.rooms.map(room => {
    if (!affected.has(room.id)) return room;
    const boundary = edges.filter(edge => edge.room === room.id);
    return frozenRoom(room.id, room.recordZone, boundary.map(edge => same(edge.start) ? point : edge.start), room.originZone,
      boundary.map(edge => sourceWallForEdge(room, edge.start, edge.end)));
  });
  validateRoomInteriors(rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  return Object.freeze({ label: 'Move shared room corner', before: current, after: source(rooms, current.nextId) });
}

export function removeSourceRooms(current: RoomSource, ids: readonly string[]): RoomChange {
  const selected = new Set(ids);
  if (!selected.size || [...selected].some(id => !current.rooms.some(room => room.id === id))) throw new Error('Select existing rooms to delete.');
  return Object.freeze({ label: 'Delete rooms', before: current, after: source(current.rooms.filter(room => !selected.has(room.id)), current.nextId) });
}

export function executeRoomChange(history: RoomHistory, change: RoomChange): RoomHistory {
  if (history.source !== change.before) throw new Error('The rooms changed before this edit could be applied. No changes were saved.');
  if (change.before === change.after) return history;
  return { source: change.after, past: [...history.past, change], future: [] };
}
export function undoRoomChange(history: RoomHistory): RoomHistory {
  const change = history.past.at(-1);
  return change ? { source: change.before, past: history.past.slice(0, -1), future: [...history.future, change] } : history;
}
export function redoRoomChange(history: RoomHistory): RoomHistory {
  const change = history.future.at(-1);
  return change ? { source: change.after, past: [...history.past, change], future: history.future.slice(0, -1) } : history;
}
