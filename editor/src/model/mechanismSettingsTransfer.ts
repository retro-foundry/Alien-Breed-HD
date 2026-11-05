import { BinaryReader, checkedInteger } from './binary';
import { editRoomSource, numberPatch, type EditCommand } from './commands';
import type { LevelDocument } from './level';
import { assertMechanismRange } from './mechanismActivation';
import { compileControllerCopies, setRoomControllerCopies } from './movingControllerCopies';
import { setSourceMovingRoom } from './sourceMovingRoom';
import { resolveAnimationRoom } from './animationRooms';

export type MechanismOwner = { kind: 'door' | 'lift' } & (
  { type: 'native'; id: number } | { type: 'copy' | 'authored'; room: string });
export interface MechanismSettings {
  readonly kind: 'door' | 'lift'; readonly top: number; readonly bottom: number;
  readonly conditions: number; readonly openRule: number; readonly closeRule: number;
}
export interface MechanismTransferOptions { readonly range: boolean; readonly rules: boolean; readonly requirements: boolean }

export const mechanismOwnerKey = (owner: MechanismOwner) => `${owner.type}:${owner.kind}:${owner.type === 'native' ? owner.id : owner.room}`;
export function mechanismSettingsOwners(document: LevelDocument, kind: 'door' | 'lift') {
  return [
    ...document.mechanisms.filter(item => item.kind === kind).map(item => {
      const zone = resolveAnimationRoom(document, item.zone, kind === 'lift');
      if (zone === undefined) throw new Error(`Original ${kind} ${item.id} has no controlled room.`);
      return { owner: { type: 'native', kind, id: item.id } as MechanismOwner, label: `Original ${kind} ${item.id} · Zone ${zone}` };
    }),
    ...(document.roomSource.controllerCopies ?? []).filter(item => item.kind === kind).map(item => ({ owner: { type: 'copy', kind, room: item.room } as MechanismOwner, label: `Copied ${kind} · Room ${item.room.replace(/^room-/, '')}` })),
    ...document.roomSource.rooms.filter(room => room.moving?.kind === kind).map(room => ({ owner: { type: 'authored', kind, room: room.id } as MechanismOwner, label: `Authored ${kind} · Room ${room.id.replace(/^room-/, '')}` })),
  ];
}

function readHeader(document: LevelDocument, owner: MechanismOwner): Uint8Array {
  if (owner.type === 'native') {
    const item = document.mechanisms.find(item => item.kind === owner.kind && item.id === owner.id);
    if (!item) throw new Error('That original mechanism no longer exists.');
    return document.files.graphics.slice(item.offset, item.offset + 18);
  }
  const item = owner.type === 'copy'
    ? document.roomSource.controllerCopies?.find(item => item.kind === owner.kind && item.room === owner.room)
    : document.roomSource.rooms.find(room => room.id === owner.room)?.moving;
  if (!item || item.kind !== owner.kind) throw new Error('That source mechanism no longer exists.');
  return new Uint8Array(item.header);
}

/** Capture supported settings, never pointers, live state or attachment IDs.
 * leveld.asc DEFDOOR/DEFLIFT </> copies activation settings between groups.
 * Current native headers expose ranges/rules/requirements separately; range
 * and requirement transfer are explicit additional authoring choices. */
export function captureMechanismSettings(document: LevelDocument, owner: MechanismOwner): MechanismSettings {
  const reader = new BinaryReader(readHeader(document, owner), 'mechanism settings');
  return Object.freeze({ kind: owner.kind, bottom: reader.i16(0) / 4, top: reader.i16(2) / 4,
    conditions: reader.u16(14), openRule: reader.u8(16), closeRule: reader.u8(17) });
}

export function transferMechanismSettings(document: LevelDocument, target: MechanismOwner, value: MechanismSettings, options: MechanismTransferOptions): EditCommand {
  if (value.kind !== target.kind) throw new Error('Copy settings between two doors or two lifts. Their native trigger rules differ.');
  if (!Object.values(options).some(Boolean)) throw new Error('Choose at least one group of settings to copy.');
  const header = readHeader(document, target), view = new DataView(header.buffer);
  if (options.range) {
    view.setInt16(0, checkedInteger(value.bottom * 4, -32768, 32767, 'Lower destination in quarter units'));
    view.setInt16(2, checkedInteger(value.top * 4, -32768, 32767, 'Upper destination in quarter units'));
  }
  if (options.requirements) view.setUint16(14, checkedInteger(value.conditions, 0, 65535, 'Mechanism requirements'));
  if (options.rules) {
    view.setUint8(16, checkedInteger(value.openRule, 0, target.kind === 'door' ? 5 : 3, 'Opening rule'));
    view.setUint8(17, checkedInteger(value.closeRule, 0, target.kind === 'door' ? 1 : 3, 'Closing rule'));
  }
  assertMechanismRange({ kind: target.kind, current: view.getInt16(4) / 4, direction: view.getInt16(6), conditions: view.getUint16(14) }, view.getInt16(2) / 4, view.getInt16(0) / 4);
  const label = `Copy ${target.kind} settings`;
  if (target.type === 'native') {
    const item = document.mechanisms.find(item => item.kind === target.kind && item.id === target.id)!;
    return { label, patches: [
      ...(options.range ? [numberPatch(document, 'graphics', item.offset, view.getInt16(0), 2), numberPatch(document, 'graphics', item.offset + 2, view.getInt16(2), 2)] : []),
      ...(options.requirements ? [numberPatch(document, 'graphics', item.offset + 14, view.getUint16(14), 2, false)] : []),
      ...(options.rules ? [numberPatch(document, 'graphics', item.offset + 16, view.getUint8(16), 1, false), numberPatch(document, 'graphics', item.offset + 17, view.getUint8(17), 1, false)] : []),
    ] };
  }
  if (target.type === 'copy') {
    const copies = compileControllerCopies(document.roomSource.controllerCopies ?? []).map(item => item.room === target.room && item.kind === target.kind ? { ...item, header } : item);
    return { ...editRoomSource(setRoomControllerCopies(document, copies)), label };
  }
  const moving = document.roomSource.rooms.find(room => room.id === target.room)!.moving!;
  return { ...editRoomSource(setSourceMovingRoom(document, target.room, { ...moving, header: [...header] })), label };
}
