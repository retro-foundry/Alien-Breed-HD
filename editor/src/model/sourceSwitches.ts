import { checkedInteger } from './binary';
import { restoreBoundaryMaterials } from './boundaryMaterials';
import type { LevelDocument } from './level';
import type { RoomChange, RoomSource, SourceRoom } from './roomSource';
import { roomSettings } from './roomHeights';
import { mechanismsForRoom } from './mechanismDependencies';
import { copiedMechanismSettings } from './movingControllerCopies';
import { sourceMovingSettings } from './sourceMovingRoom';
import { activeNativeSwitches, restoreRemovedNativeSwitches } from './nativeSwitches';

export interface SourceSwitch { readonly slot: number; readonly boundary: number; readonly automatic: boolean; readonly native?: number }

export function restoreSourceSwitches(value: unknown): readonly SourceSwitch[] {
  if (!Array.isArray(value) || value.length > 8) throw new Error('A room can contain at most eight switches.');
  const slots = new Set<number>(), boundaries = new Set<number>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).some(key => !['slot', 'boundary', 'automatic', 'native'].includes(key)) || !('slot' in item) || !('boundary' in item) || !('automatic' in item) || typeof item.automatic !== 'boolean') throw new Error('A switch needs its slot, wall identity and reset mode.');
    const slot = checkedInteger(item.slot, 0, 7, 'Switch slot'), boundary = checkedInteger(item.boundary, 0, 16777215, 'Switch wall identity');
    if (slots.has(slot) || boundaries.has(boundary)) throw new Error('Switches need independent numbered slots and walls.');
    slots.add(slot); boundaries.add(boundary);
    return Object.freeze({ slot, boundary, automatic: item.automatic, ...('native' in item ? { native: checkedInteger(item.native, 0, 7, 'Native switch template') } : {}) });
  }));
}

export function availableSwitchSlots(document: LevelDocument): number[] {
  const occupied = new Set([...activeNativeSwitches(document).map(item => item.id),
    ...(document.roomSource.switchCopies ?? []).map(item => item.slot), ...document.roomSource.rooms.flatMap(room => room.switches?.map(item => item.slot) ?? [])]);
  return Array.from({ length: 8 }, (_, slot) => slot).filter(slot => !occupied.has(slot));
}

export function switchRequirements(document: LevelDocument, slot: number): boolean {
  const masks = [...document.mechanisms.map(item => item.conditions),
    ...(document.roomSource.controllerCopies ?? []).map(item => copiedMechanismSettings(item).conditions),
    ...document.roomSource.rooms.flatMap(room => room.moving ? [sourceMovingSettings(room.moving).conditions] : [])];
  return masks.some(mask => (mask & (1 << (slot + 4))) !== 0);
}

export function setSourceSwitch(document: LevelDocument, id: string, slot: number, value: { edge: number; automatic: boolean } | null): RoomChange {
  checkedInteger(slot, 0, 7, 'Switch slot');
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select a room for this switch.');
  const existing = room.switches?.find(item => item.slot === slot);
  if (!existing && !value) throw new Error('That authored switch no longer exists.');
  if (!existing && !availableSwitchSlots(document).includes(slot)) throw new Error('That switch number is already used. The level supports eight switches.');
  if (!value && switchRequirements(document, slot)) throw new Error(`Switch ${slot + 1} is required by a door or lift. Clear that requirement before removing it.`);
  let boundaryMaterials = room.boundaryMaterials;
  if (value) {
    checkedInteger(value.edge, 0, room.vertices.length - 1, 'Switch wall edge');
    if (typeof value.automatic !== 'boolean') throw new Error('Choose a switch reset mode.');
    if (room.moving || room.originZone !== null && mechanismsForRoom(document, room.originZone).length
      || document.roomSource.controllerCopies?.some(copy => copy.room === room.id)) throw new Error('Choose a static room for this switch.');
    const settings = roomSettings(document, room);
    if (settings.floor - settings.ceiling < 64) throw new Error('The switch panel needs at least 64 units of lower-storey headroom.');
    boundaryMaterials ??= restoreBoundaryMaterials({ nextId: room.vertices.length, edges: room.vertices.map((_, index) => index), overrides: [] }, room.vertices.length);
  }
  const remaining = (room.switches ?? []).filter(item => item.slot !== slot);
  const switches = restoreSourceSwitches(value ? [...remaining, { ...existing, slot, boundary: boundaryMaterials!.edges[value.edge]!, automatic: value.automatic }] : remaining);
  const { switches: _previous, ...retained } = room;
  const updated: SourceRoom = Object.freeze({ ...retained, ...(boundaryMaterials ? { boundaryMaterials } : {}), ...(switches.length ? { switches } : {}) });
  return { label: value ? existing ? `Edit switch ${slot + 1}` : `Create switch ${slot + 1}` : `Remove switch ${slot + 1}`, before,
    after: Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === id ? updated : item)) }) };
}

/** Adopt the same native condition number into an independently authored wall
 * panel. The original binary record stays available to copies and Undo. */
export function reassignNativeSwitch(document: LevelDocument, slot: number, target: { room: string; edge: number } | null): RoomChange {
  checkedInteger(slot, 0, 7, 'Switch slot');
  const original = activeNativeSwitches(document).find(item => item.id === slot);
  if (!original) throw new Error('That active native switch no longer exists.');
  if (!target && switchRequirements(document, slot)) throw new Error(`Switch ${slot + 1} is required by a door or lift. Clear that requirement before removing it.`);
  const before = document.roomSource;
  const removedNativeSwitches = restoreRemovedNativeSwitches([...(before.removedNativeSwitches ?? []), slot]);
  let after: RoomSource = Object.freeze({ ...before, removedNativeSwitches });
  if (target) {
    after = setSourceSwitch({ ...document, roomSource: after }, target.room, slot, { edge: target.edge, automatic: original.type !== 0 }).after;
    after = Object.freeze({ ...after, rooms: Object.freeze(after.rooms.map(room => room.id !== target.room ? room : Object.freeze({ ...room,
      switches: restoreSourceSwitches(room.switches!.map(item => item.slot === slot ? { ...item, native: slot } : item)) }))) });
  }
  return { label: target ? `Reassign switch ${slot + 1}` : `Remove switch ${slot + 1}`, before, after };
}

export function moveSourceSwitch(document: LevelDocument, slot: number, target: { room: string; edge: number }): RoomChange {
  const before = document.roomSource, owner = before.rooms.find(room => room.switches?.some(item => item.slot === slot));
  const existing = owner?.switches?.find(item => item.slot === slot);
  if (!owner || !existing) throw new Error('That authored switch no longer exists.');
  if (owner.id === target.room) return setSourceSwitch(document, owner.id, slot, { edge: target.edge, automatic: existing.automatic });
  const temporary = { ...before, rooms: before.rooms.map(room => {
    if (room.id !== owner.id) return room;
    const { switches, ...retained } = room, remaining = switches!.filter(item => item.slot !== slot);
    return Object.freeze({ ...retained, ...(remaining.length ? { switches: Object.freeze(remaining) } : {}) });
  }) };
  const created = setSourceSwitch({ ...document, roomSource: temporary }, target.room, slot, { edge: target.edge, automatic: existing.automatic }).after;
  const after = Object.freeze({ ...created, rooms: Object.freeze(created.rooms.map(room => room.id !== target.room ? room : Object.freeze({ ...room,
    switches: restoreSourceSwitches(room.switches!.map(item => item.slot !== slot ? item : { ...existing, boundary: item.boundary })) }))) });
  return { label: `Move switch ${slot + 1}`, before, after };
}
