import { checkedInteger } from './binary';
import { restoreBoundaryMaterials } from './boundaryMaterials';
import type { LevelDocument } from './level';
import type { RoomChange, SourceRoom } from './roomSource';
import { roomSettings } from './roomHeights';
import { mechanismsForRoom } from './mechanismDependencies';
import { copiedMechanismSettings } from './movingControllerCopies';
import { sourceMovingSettings } from './sourceMovingRoom';

export interface SourceSwitch { readonly slot: number; readonly boundary: number; readonly automatic: boolean }

export function restoreSourceSwitches(value: unknown): readonly SourceSwitch[] {
  if (!Array.isArray(value) || value.length > 8) throw new Error('A room can contain at most eight switches.');
  const slots = new Set<number>(), boundaries = new Set<number>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).length !== 3 || !('slot' in item) || !('boundary' in item) || !('automatic' in item) || typeof item.automatic !== 'boolean') throw new Error('A switch needs its slot, wall identity and reset mode.');
    const slot = checkedInteger(item.slot, 0, 7, 'Switch slot'), boundary = checkedInteger(item.boundary, 0, 16777215, 'Switch wall identity');
    if (slots.has(slot) || boundaries.has(boundary)) throw new Error('Switches need independent numbered slots and walls.');
    slots.add(slot); boundaries.add(boundary);
    return Object.freeze({ slot, boundary, automatic: item.automatic });
  }));
}

export function availableSwitchSlots(document: LevelDocument): number[] {
  const occupied = new Set([...document.switches.filter(item => item.zone >= 0).map(item => item.id),
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
    if (room.moving || mechanismsForRoom(document, room.recordZone).length) throw new Error('Choose a static room for this switch.');
    const settings = roomSettings(document, room);
    if (settings.floor - settings.ceiling < 64) throw new Error('The switch panel needs at least 64 units of lower-storey headroom.');
    boundaryMaterials ??= restoreBoundaryMaterials({ nextId: room.vertices.length, edges: room.vertices.map((_, index) => index), overrides: [] }, room.vertices.length);
  }
  const remaining = (room.switches ?? []).filter(item => item.slot !== slot);
  const switches = restoreSourceSwitches(value ? [...remaining, { slot, boundary: boundaryMaterials!.edges[value.edge]!, automatic: value.automatic }] : remaining);
  const { switches: _previous, ...retained } = room;
  const updated: SourceRoom = Object.freeze({ ...retained, ...(boundaryMaterials ? { boundaryMaterials } : {}), ...(switches.length ? { switches } : {}) });
  return { label: value ? existing ? `Edit switch ${slot + 1}` : `Create switch ${slot + 1}` : `Remove switch ${slot + 1}`, before,
    after: Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === id ? updated : item)) }) };
}
