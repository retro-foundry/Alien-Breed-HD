import { restoreSourceMovingRoom, type SourceMovingRoom } from './sourceMovingRoom';
import { convexParts, normaliseRoomOutline, validateRoomInteriors, type Position } from './geometry';
import type { LevelDocument } from './level';
import { checkedInteger } from './binary';
import { restoreRoomHeights, roomHeightFields, type RoomHeights } from './roomHeights';
import { restoreRoomProperties, type RoomProperties } from './roomProperties';
import { restoreControllerCopies, type SourceControllerCopy } from './movingControllerCopies';
import { restoreSwitchCopies, type SourceSwitchCopy } from './switchCopies';
import { restoreWallMaterials, type WallMaterial } from './wallMaterials';
import { restorePolygonMaterials, type PolygonMaterial } from './polygonMaterials';
import { restoreBoundaryMaterials, remapBoundaryMaterials, type BoundaryMaterials } from './boundaryMaterials';

/** Editable room identities survive compiler zone renumbering. The imported
 * record remains the authority for materials/mechanisms until those authoring
 * adapters are implemented; it is never reconstructed from guessed defaults. */
export interface SourceRoom {
  readonly id: string;
  readonly recordZone: number;
  readonly originZone: number | null;
  readonly vertices: readonly Position[];
  readonly wallSources: readonly (number | null)[];
  readonly wallMaterials?: readonly WallMaterial[];
  readonly polygonMaterials?: readonly PolygonMaterial[];
  readonly boundaryMaterials?: BoundaryMaterials;
  readonly moving?: SourceMovingRoom;
  readonly heights?: RoomHeights;
  readonly properties?: RoomProperties;
}
export interface RoomSource { readonly rooms: readonly SourceRoom[]; readonly nextId: number; readonly controllerCopies?: readonly SourceControllerCopy[]; readonly switchCopies?: readonly SourceSwitchCopy[] }
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
  wallSources: readonly (number | null)[] = vertices.map(() => null), heights?: RoomHeights, properties?: RoomProperties, materials: Pick<SourceRoom, 'wallMaterials' | 'boundaryMaterials' | 'polygonMaterials' | 'moving'> = {}): SourceRoom {
  const { wallMaterials, boundaryMaterials, polygonMaterials, moving } = materials;
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
  const boundary = boundaryMaterials && remapBoundaryMaterials({ vertices, boundaryMaterials },
    outline.map((start, index) => ({ start, end: outline[(index + 1) % outline.length]! })));
  return Object.freeze({ id, recordZone, originZone, vertices: Object.freeze(outline.map(point => Object.freeze(point))), wallSources: Object.freeze(walls), ...(heights ? { heights } : {}), ...(properties ? { properties } : {}), ...(wallMaterials?.length ? { wallMaterials: restoreWallMaterials(wallMaterials) } : {}), ...(boundary ? { boundaryMaterials: boundary } : {}), ...(polygonMaterials?.length ? { polygonMaterials: restorePolygonMaterials(polygonMaterials) } : {}), ...(moving ? { moving: restoreSourceMovingRoom(moving) } : {}) });
}
function source(rooms: readonly SourceRoom[], nextId: number, plans?: Pick<RoomSource, 'controllerCopies' | 'switchCopies'>): RoomSource {
  return Object.freeze({ rooms: Object.freeze([...rooms]), nextId,
    ...(plans?.controllerCopies?.length ? { controllerCopies: plans.controllerCopies } : {}),
    ...(plans?.switchCopies?.length ? { switchCopies: plans.switchCopies } : {}) });
}

type RoomRecords = Pick<LevelDocument, 'zones' | 'lines'>;
export function importRoomSource(document: RoomRecords): RoomSource {
  return source(document.zones.map(zone => frozenRoom(`room-${zone.id}`, zone.id,
    zone.boundaryIds.map(id => { const line = document.lines[id]!; return { x: line.x, z: line.z }; }), zone.id, zone.boundaryIds)), Math.max(-1, ...document.zones.map(zone => zone.id)) + 1);
}

export function roomSourceSnapshot(current: RoomSource) {
  const moving = current.rooms.some(room => room.moving);
  const polygons = current.rooms.some(room => room.polygonMaterials?.length);
  const boundary = current.rooms.some(room => room.boundaryMaterials);
  const painted = moving || polygons || boundary || current.rooms.some(room => room.wallMaterials?.length);
  return { format: 'alien-breed-room-source', version: moving ? 10 : polygons ? 9 : boundary ? 8 : painted ? 7 : current.switchCopies?.length ? 6 : current.controllerCopies?.length ? 5 : current.rooms.some(room => room.properties) ? 4 : current.rooms.some(room => room.heights) ? 3 : 2, nextId: current.nextId, rooms: current.rooms,
    ...(painted || current.controllerCopies?.length || current.switchCopies?.length ? { controllerCopies: restoreControllerCopies(current.controllerCopies ?? []) } : {}),
    ...(painted || current.switchCopies?.length ? { switchCopies: restoreSwitchCopies(current.switchCopies ?? []) } : {}) };
}

export function roomSourceMatchesLevel(document: LevelDocument): boolean {
  return !document.roomSource.controllerCopies?.length && !document.roomSource.switchCopies?.length && JSON.stringify(document.roomSource.rooms) === JSON.stringify(importRoomSource(document).rooms);
}

/** All editable source must match before using the imported runtime files. */
export function sourceMatchesLevel(document: LevelDocument): boolean {
  return !document.sourceStarts?.length && !document.sourceObjects?.length && !document.sourceTeleports?.length && !document.roomAppearances?.length && roomSourceMatchesLevel(document);
}

/** Restore committed editable geometry, never a partially drawn gesture or
 * generated runtime cells. Original binary records remain the material and
 * dependency baseline; missing templates cannot be silently substituted. */
export function restoreRoomSource(document: RoomRecords, value: unknown): RoomSource {
  if (!value || typeof value !== 'object' || !('format' in value) || value.format !== 'alien-breed-room-source'
    || !('version' in value) || ![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].includes(value.version as number) || !('rooms' in value) || !Array.isArray(value.rooms)
    || !('nextId' in value) || typeof value.nextId !== 'number') throw new Error('Unsupported or damaged room source.');
  if (Object.keys(value).some(key => !['format', 'version', 'rooms', 'nextId', ...((value.version as number) >= 5 ? ['controllerCopies'] : []), ...((value.version as number) >= 6 ? ['switchCopies'] : [])].includes(key))) throw new Error('Room source contains unsupported fields; it has not been replaced or discarded.');
  if ((value.version as number) >= 5 && !('controllerCopies' in value)) throw new Error('Room source is missing its copied controller records.');
  if ((value.version as number) >= 6 && !('switchCopies' in value)) throw new Error('Room source is missing its copied switch records.');
  const controllerCopies = (value.version as number) >= 5 && 'controllerCopies' in value ? restoreControllerCopies(value.controllerCopies) : undefined;
  const switchCopies = (value.version as number) >= 6 && 'switchCopies' in value ? restoreSwitchCopies(value.switchCopies) : undefined;
  const nextId = checkedInteger(value.nextId, 0, Number.MAX_SAFE_INTEGER - 1, 'Next room identity');
  const records = new Set(document.zones.map(zone => zone.id)), ids = new Set<string>(), origins = new Set<number>();
  const rooms = value.rooms.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('id' in item) || typeof item.id !== 'string' || !item.id || item.id.length > 128 || ids.has(item.id))
      throw new Error('Room source contains an invalid or duplicate room identity.');
    ids.add(item.id);
    if (Object.keys(item).some(key => !['id', 'recordZone', 'originZone', 'vertices', ...(value.version !== 1 ? ['wallSources'] : []), ...((value.version as number) >= 3 ? ['heights'] : []), ...((value.version as number) >= 4 ? ['properties'] : []), ...((value.version as number) >= 7 ? ['wallMaterials'] : []), ...((value.version as number) >= 8 ? ['boundaryMaterials'] : []), ...((value.version as number) >= 9 ? ['polygonMaterials'] : []), ...((value.version as number) >= 10 ? ['moving'] : [])].includes(key))) throw new Error(`Room ${item.id} contains unsupported source fields.`);
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
    if (value.version !== 1) {
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
    return frozenRoom(item.id, item.recordZone, vertices, origin, walls, 'heights' in item ? restoreRoomHeights(item.heights, template) : undefined,
      'properties' in item ? restoreRoomProperties(item.properties) : undefined, { wallMaterials: 'wallMaterials' in item ? restoreWallMaterials(item.wallMaterials) : undefined, boundaryMaterials: 'boundaryMaterials' in item ? restoreBoundaryMaterials(item.boundaryMaterials, vertices.length) : undefined, polygonMaterials: 'polygonMaterials' in item ? restorePolygonMaterials(item.polygonMaterials) : undefined, moving: 'moving' in item ? restoreSourceMovingRoom(item.moving) : undefined });
  });
  return source(rooms, nextId, { controllerCopies, switchCopies });
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

/** Validate a generated group together, without repeatedly rebuilding all existing rooms. */
export function addSourceRooms(document: RoomRecords, current: RoomSource,
  drafts: readonly { recordZone: number; vertices: readonly Position[]; heights?: RoomHeights }[]): RoomChange {
  if (!drafts.length) return { label: 'Create rooms', before: current, after: current };
  checkedInteger(current.rooms.length + drafts.length, 1, 256, 'Source room count');
  checkedInteger(current.nextId, 0, Number.MAX_SAFE_INTEGER - drafts.length - 1, 'Next room identity');
  const added = drafts.map((draft, index) => {
    const template = document.zones.find(zone => zone.id === draft.recordZone);
    if (!template) throw new Error('Choose an existing room material template.');
    const requested = draft.heights ? restoreRoomHeights(draft.heights, template) : undefined;
    const heights = requested && roomHeightFields.some(field => requested[field] !== template[field]) ? requested : undefined;
    const room = frozenRoom(`room-${current.nextId + index}`, draft.recordZone, draft.vertices, null, undefined, heights);
    convexParts(room.vertices);
    return room;
  });
  const rooms = [...current.rooms, ...added];
  validateRoomInteriors(rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  return { label: 'Create rooms', before: current, after: source(rooms, current.nextId + added.length, current) };
}

export function commitRoomDraft(current: RoomSource, draft: RoomDraft): RoomChange {
  if (current !== draft.base) throw new Error('The rooms changed while this outline was being drawn. No changes were saved.');
  const existing = draft.roomId === null ? undefined : current.rooms.find(room => room.id === draft.roomId);
  if (!existing) checkedInteger(current.nextId, 0, Number.MAX_SAFE_INTEGER - 2, 'Next room identity');
  if (draft.roomId !== null && !existing) throw new Error('That room no longer exists.');
  const id = existing?.id ?? `room-${current.nextId}`;
  const walls = draft.vertices.map((start, i) => existing ? sourceWallForEdge(existing, start, draft.vertices[(i + 1) % draft.vertices.length]!) : null);
  const room = frozenRoom(id, draft.recordZone, draft.vertices, existing?.originZone ?? null, walls, existing?.heights, existing?.properties, { ...existing, boundaryMaterials: existing && remapBoundaryMaterials(existing, draft.vertices.map((start, index) => ({ start, end: draft.vertices[(index + 1) % draft.vertices.length]! }))) });
  if (existing && existing.recordZone === room.recordZone && existing.vertices.length === room.vertices.length
    && existing.vertices.every((point, i) => point.x === room.vertices[i]!.x && point.z === room.vertices[i]!.z))
    return Object.freeze({ label: 'Edit room outline', before: current, after: current });
  // Enforce the actual convex-cell corner contract before accepting a command.
  // Concavity and winding remain editor responsibilities, not user settings.
  convexParts(room.vertices);
  const rooms = existing ? current.rooms.map(value => value.id === id ? room : value) : [...current.rooms, room];
  validateRoomInteriors(rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  return Object.freeze({ label: existing ? 'Edit room outline' : 'Create room', before: current,
    after: source(rooms, current.nextId + (existing ? 0 : 1), current) });
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
      boundary.map(edge => sourceWallForEdge(room, edge.start, edge.end)), room.heights, room.properties, { ...room, boundaryMaterials: remapBoundaryMaterials(room, boundary) });
  });
  validateRoomInteriors(rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  return Object.freeze({ label: 'Move shared room corner', before: current, after: source(rooms, current.nextId, current) });
}

export function removeSourceRooms(current: RoomSource, ids: readonly string[]): RoomChange {
  const selected = new Set(ids);
  if (!selected.size || [...selected].some(id => !current.rooms.some(room => room.id === id))) throw new Error('Select existing rooms to delete.');
  return Object.freeze({ label: 'Delete rooms', before: current, after: source(current.rooms.filter(room => !selected.has(room.id)), current.nextId,
    { controllerCopies: current.controllerCopies?.filter(copy => !selected.has(copy.room)), switchCopies: current.switchCopies?.filter(copy => !selected.has(copy.room)) }) });
}

/** Translate all selected boundaries simultaneously. Shared portal vertices,
 * including inserted T-junctions, follow through their incident room fan;
 * unrelated corners of neighbouring rooms remain anchored. */
export function translateSourceRooms(current: RoomSource, ids: readonly string[], x: number, z: number): RoomChange {
  const selected = new Set(ids);
  if (!selected.size || selected.size !== ids.length || ids.some(id => !current.rooms.some(room => room.id === id))) throw new Error('Select different existing rooms to move.');
  checkedInteger(x, -65535, 65535, 'Room X offset'); checkedInteger(z, -65535, 65535, 'Room Z offset');
  if (!x && !z) return { label: 'Move rooms', before: current, after: current };
  const edges = validateRoomInteriors(current.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  const key = (point: Position) => `${point.x},${point.z}`;
  const moved = new Map<string, Set<string>>();
  for (const edge of edges) if (selected.has(edge.room)) {
    const points = moved.get(edge.room) ?? new Set<string>(); points.add(key(edge.start)); points.add(key(edge.end)); moved.set(edge.room, points);
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const edge of edges) if (edge.neighbour !== null) {
      for (const point of [edge.start, edge.end]) if (moved.get(edge.room)?.has(key(point))) {
        const points = moved.get(edge.neighbour) ?? new Set<string>();
        if (!points.has(key(point))) { points.add(key(point)); moved.set(edge.neighbour, points); changed = true; }
      }
    }
  }
  const rooms = current.rooms.map(room => {
    const points = moved.get(room.id);
    if (!points?.size) return room;
    const boundary = edges.filter(edge => edge.room === room.id);
    return frozenRoom(room.id, room.recordZone, boundary.map(edge => points.has(key(edge.start)) ? { x: edge.start.x + x, z: edge.start.z + z } : edge.start),
      room.originZone, boundary.map(edge => sourceWallForEdge(room, edge.start, edge.end)), room.heights, room.properties, { ...room, boundaryMaterials: remapBoundaryMaterials(room, boundary) });
  });
  validateRoomInteriors(rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  return { label: `Move ${ids.length} rooms`, before: current, after: source(rooms, current.nextId, current) };
}

export function setRoomHeights(document: Pick<LevelDocument, 'zones'>, current: RoomSource, id: string, value: unknown): RoomChange {
  const room = current.rooms.find(room => room.id === id);
  if (!room) throw new Error('Choose an existing room before changing heights.');
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const heights = restoreRoomHeights(value, template);
  const selected = roomHeightFields.every(field => heights[field] === template[field]) ? undefined : heights;
  if (JSON.stringify(selected) === JSON.stringify(room.heights)) return { label: 'Change room heights', before: current, after: current };
  const { heights: _previous, ...unchanged } = room;
  const updated = Object.freeze({ ...unchanged, ...(selected ? { heights: selected } : {}) });
  return { label: 'Change room heights', before: current, after: source(current.rooms.map(item => item.id === id ? updated : item), current.nextId, current) };
}

export function setRoomProperties(current: RoomSource, id: string, value: unknown | null): RoomChange {
  const room = current.rooms.find(room => room.id === id);
  if (!room) throw new Error('Choose an existing room before changing its properties.');
  const properties = value === null ? undefined : restoreRoomProperties(value);
  if (JSON.stringify(properties) === JSON.stringify(room.properties)) return { label: 'Change room properties', before: current, after: current };
  const { properties: _previous, ...unchanged } = room;
  const updated = Object.freeze({ ...unchanged, ...(properties ? { properties } : {}) });
  return { label: properties ? 'Change room properties' : 'Inherit room properties', before: current,
    after: source(current.rooms.map(item => item.id === id ? updated : item), current.nextId, current) };
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
