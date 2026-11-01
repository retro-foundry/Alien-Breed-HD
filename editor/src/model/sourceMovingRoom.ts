import { validateWallMaterialSampling } from './wallMaterials';
import { movingRoomConflict } from './movingRoomEligibility';
import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { RoomChange, SourceRoom } from './roomSource';
import type { AuthoredMovingRoom } from './authoredMovingRoom';
import { assertMechanismRange } from './mechanismActivation';

/** Stable source settings: native header bytes and selected source material,
 * never generated room IDs, record indices or graphics/collision offsets. */
export interface SourceMovingRoom { readonly kind: 'door' | 'lift'; readonly header: readonly number[]; readonly wall: number }
export function restoreSourceMovingRoom(value: unknown): SourceMovingRoom {
  if (!value || typeof value !== 'object' || Object.keys(value).length !== 3
    || Object.keys(value).some(key => !['kind', 'header', 'wall'].includes(key))) throw new Error('Invalid authored door/lift settings.');
  const item = value as Record<string, unknown>;
  if (item.kind !== 'door' && item.kind !== 'lift') throw new Error('Choose a door or lift.');
  if (!Array.isArray(item.header) || item.header.length !== 18) throw new Error('Authored moving settings need a complete native header.');
  const header = Object.freeze(item.header.map(byte => checkedInteger(byte, 0, 255, 'Moving settings byte')));
  const reader = new BinaryReader(new Uint8Array(header), 'moving settings');
  assertMechanismRange({ kind: item.kind, direction: reader.i16(6), conditions: reader.u16(14), current: reader.i16(4) / 4 }, reader.i16(2) / 4, reader.i16(0) / 4);
  if (reader.u8(16) > (item.kind === 'door' ? 5 : 3) || reader.u8(17) > (item.kind === 'door' ? 1 : 3)) throw new Error('Unsupported door/lift trigger rule.');
  return Object.freeze({ kind: item.kind, header, wall: checkedInteger(item.wall as number, 0, 16777215, 'Moving wall source material') });
}
export function sourceMovingSettings(value: SourceMovingRoom) {
  const reader = new BinaryReader(new Uint8Array(value.header), 'moving settings');
  return { kind: value.kind, top: reader.i16(2) / 4, bottom: reader.i16(0) / 4, current: reader.i16(4) / 4,
    conditions: reader.u16(14), openRule: reader.u8(16), closeRule: reader.u8(17) };
}
export function compileSourceMovingRoom(document: LevelDocument, room: SourceRoom): AuthoredMovingRoom | undefined {
  if (!room.moving) return;
  const value = restoreSourceMovingRoom(room.moving);
  const record = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === value.wall && record.wall);
  if (!record?.wall) throw new Error(`Room ${room.id} has lost its door/lift wall material. Choose its material again.`);
  validateWallMaterialSampling(record.wall, record.wall);
  const [a, b] = record.wall.points.map(id => document.points[id]!);
  return { room: room.id, kind: value.kind, header: new Uint8Array(value.header), wall: { surface: record.wall, length: Math.hypot(b!.x - a!.x, b!.z - a!.z) } };
}
export function setSourceMovingRoom(document: LevelDocument, id: string, value: SourceMovingRoom | null): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select an existing room to create a door or lift.');
  const conflict = value !== null && movingRoomConflict(document, room);
  if (conflict) throw new Error(conflict);
  const moving = value === null ? undefined : restoreSourceMovingRoom(value);
  if (moving) compileSourceMovingRoom(document, { ...room, moving });
  if (JSON.stringify(room.moving) === JSON.stringify(moving)) return { label: 'Change door/lift', before, after: before };
  const { moving: _old, ...rest } = room;
  const updated = Object.freeze({ ...rest, ...(moving ? { moving } : {}) });
  return { label: moving ? `Make ${moving.kind}` : 'Remove door/lift', before, after: Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === id ? updated : item)) }) };
}


export interface MovingRoomDraft { kind: 'door' | 'lift'; top: number; bottom: number; current: number; wall: number; conditions: number; openRule: number; closeRule: number }
/** level.c's 18-byte native source header: quarter-height words, stationary
 * initial direction, generated pointer/room fields, requirements and rules. */
export function movingRoomFromDraft(draft: MovingRoomDraft): SourceMovingRoom {
  const header = new Uint8Array(18), view = new DataView(header.buffer);
  for (const [offset, value] of [[0, draft.bottom], [2, draft.top], [4, draft.current]]) view.setInt16(offset!, checkedInteger(value! * 4, -32768, 32767, 'Moving destination in quarter units'));
  view.setUint16(14, checkedInteger(draft.conditions, 0, 65535, 'Mechanism requirements'));
  view.setUint8(16, checkedInteger(draft.openRule, 0, draft.kind === 'door' ? 5 : 3, 'Opening rule'));
  view.setUint8(17, checkedInteger(draft.closeRule, 0, draft.kind === 'door' ? 1 : 3, 'Closing rule'));
  return restoreSourceMovingRoom({ kind: draft.kind, wall: draft.wall, header: [...header] });
}


/** Mechanism faces can use any existing native wall material, even when the
 * controlled room contains only portals and has no wall draw record itself. */
export function movingWallChoices(document: LevelDocument, room: SourceRoom) {
  const all = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).filter(record => record.wall);
  const own = document.zones.find(zone => zone.id === room.recordZone)!.lower.filter(record => record.wall);
  const selected = all.find(record => record.offset === room.moving?.wall);
  const preferred = selected && !own.includes(selected) ? [selected, ...own] : [...own];
  const materialKey = (record: typeof all[number]) => {
    const { points, topFixed: _top, bottomFixed: _bottom, stripEnd, ...material } = record.wall!;
    const [a, b] = points.map(id => document.points[id]!);
    return JSON.stringify([material, (stripEnd - material.stripStart) / Math.hypot(b!.x - a!.x, b!.z - a!.z)]);
  };
  const seen = new Set(preferred.map(materialKey));
  return [...preferred, ...all.filter(record => { const key = materialKey(record); if (seen.has(key)) return false; seen.add(key); return true; })];
}
