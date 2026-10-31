import { convexParts, normaliseRoomOutline, type Position } from './geometry';
import type { LevelDocument } from './level';

/** Editable room identities survive compiler zone renumbering. The imported
 * record remains the authority for materials/mechanisms until those authoring
 * adapters are implemented; it is never reconstructed from guessed defaults. */
export interface SourceRoom {
  readonly id: string;
  readonly recordZone: number;
  readonly vertices: readonly Position[];
}
export interface RoomSource { readonly rooms: readonly SourceRoom[]; readonly nextId: number }
export interface RoomChange { readonly label: string; readonly before: RoomSource; readonly after: RoomSource }
export interface RoomHistory { readonly source: RoomSource; readonly past: readonly RoomChange[]; readonly future: readonly RoomChange[] }
export interface RoomDraft { readonly base: RoomSource; readonly roomId: string | null; readonly recordZone: number; readonly vertices: readonly Position[] }

function frozenRoom(id: string, recordZone: number, vertices: readonly Position[]): SourceRoom {
  return Object.freeze({ id, recordZone, vertices: Object.freeze(normaliseRoomOutline(vertices).map(point => Object.freeze(point))) });
}
function source(rooms: readonly SourceRoom[], nextId: number): RoomSource {
  return Object.freeze({ rooms: Object.freeze([...rooms]), nextId });
}

export function importRoomSource(document: LevelDocument): RoomSource {
  return source(document.zones.map(zone => frozenRoom(`room-${zone.id}`, zone.id,
    zone.boundaryIds.map(id => { const line = document.lines[id]!; return { x: line.x, z: line.z }; }))), document.zones.length);
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
  if (draft.roomId !== null && !existing) throw new Error('That room no longer exists.');
  const id = existing?.id ?? `room-${current.nextId}`;
  const room = frozenRoom(id, draft.recordZone, draft.vertices);
  // Enforce the actual convex-cell corner contract before accepting a command.
  // Concavity and winding remain editor responsibilities, not user settings.
  convexParts(room.vertices);
  const rooms = existing ? current.rooms.map(value => value.id === id ? room : value) : [...current.rooms, room];
  return Object.freeze({ label: existing ? 'Edit room outline' : 'Create room', before: current,
    after: source(rooms, current.nextId + (existing ? 0 : 1)) });
}

export function removeSourceRooms(current: RoomSource, ids: readonly string[]): RoomChange {
  const selected = new Set(ids);
  if (!selected.size || [...selected].some(id => !current.rooms.some(room => room.id === id))) throw new Error('Select existing rooms to delete.');
  return Object.freeze({ label: 'Delete rooms', before: current, after: source(current.rooms.filter(room => !selected.has(room.id)), current.nextId) });
}

export function executeRoomChange(history: RoomHistory, change: RoomChange): RoomHistory {
  if (history.source !== change.before) throw new Error('The rooms changed before this edit could be applied. No changes were saved.');
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
