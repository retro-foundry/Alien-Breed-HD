import { BinaryReader, checkedInteger } from './binary';
import { parseLevel, sameBytes, zoneContainsPoint, type LevelDocument, type LevelFiles } from './level';

export interface BytePatch { file: keyof LevelFiles; offset: number; before: Uint8Array; after: Uint8Array }
export interface EditCommand { label: string; patches: BytePatch[] }
export interface History { document: LevelDocument; past: EditCommand[]; future: EditCommand[] }

export function numberPatch(document: LevelDocument, file: keyof LevelFiles, offset: number, value: number, size: 1 | 2 | 4, signed = true): BytePatch {
  const reader = new BinaryReader(document.files[file], file);
  reader.require(offset, size);
  const limit = 2 ** (size * 8 - (signed ? 1 : 0));
  checkedInteger(value, signed ? -limit : 0, limit - 1, 'Value');
  const after = new Uint8Array(size), view = new DataView(after.buffer);
  if (size === 1) view.setUint8(0, value);
  else if (size === 2) view.setUint16(0, value);
  else view.setUint32(0, value);
  return { file, offset, before: reader.bytes.slice(offset, offset + size), after };
}

export function applyCommand(document: LevelDocument, command: EditCommand, reverse = false): LevelDocument {
  const files = { ...document.files };
  const changed = new Set<keyof LevelFiles>();
  for (const patch of command.patches) {
    if (!changed.has(patch.file)) { files[patch.file] = files[patch.file].slice(); changed.add(patch.file); }
    const before = reverse ? patch.after : patch.before, after = reverse ? patch.before : patch.after;
    const reader = new BinaryReader(files[patch.file], patch.file);
    reader.require(patch.offset, before.length);
    if (before.length !== after.length || !sameBytes(reader.bytes.subarray(patch.offset, patch.offset + before.length), before))
      throw new Error('The document changed before this edit could be applied. No changes were saved.');
    files[patch.file].set(after, patch.offset);
  }
  return parseLevel(document.name, files, document.encoded, document.original);
}

export function execute(history: History, command: EditCommand): History {
  if (command.patches.every(patch => sameBytes(patch.before, patch.after))) return history;
  const document = applyCommand(history.document, command);
  return { document, past: [...history.past, command], future: [] };
}

export function undo(history: History): History {
  const command = history.past.at(-1);
  if (!command) return history;
  return { document: applyCommand(history.document, command, true), past: history.past.slice(0, -1), future: [...history.future, command] };
}

export function redo(history: History): History {
  const command = history.future.at(-1);
  if (!command) return history;
  return { document: applyCommand(history.document, command), past: [...history.past, command], future: history.future.slice(0, -1) };
}

export type ZoneProperty = 'brightness' | 'upperBrightness' | 'floorSound' | 'upperFloorSound';
export function setZoneProperty(document: LevelDocument, zoneId: number, property: ZoneProperty, value: number): EditCommand {
  const zone = document.zones.find(zone => zone.id === zoneId);
  if (!zone) throw new Error('That zone no longer exists.');
  const fields: Record<ZoneProperty, { offset: number; size: 1 | 2 | 4; signed: boolean }> = {
    brightness: { offset: 22, size: 2, signed: true }, upperBrightness: { offset: 24, size: 2, signed: true },
    floorSound: { offset: 44, size: 2, signed: true },
    upperFloorSound: { offset: 46, size: 2, signed: true },
  };
  const field = fields[property];
  return { label: `Change zone ${zoneId} ${property}`, patches: [numberPatch(document, 'data', zone.offset + field.offset, value, field.size, field.signed)] };
}

export function setTeleport(document: LevelDocument, zoneId: number, destination: number, x: number, z: number): EditCommand {
  const zone = document.zones.find(zone => zone.id === zoneId);
  if (!zone) throw new Error('That zone no longer exists.');
  if (destination !== -1) {
    const target = document.zones.find(zone => zone.id === destination);
    if (!target) throw new Error('Choose a valid destination zone, or disable the teleporter.');
    if (!zoneContainsPoint(document, target, x, z)) throw new Error('Place the teleporter destination inside its selected zone.');
  }
  return { label: `Change zone ${zoneId} teleporter`, patches: [
    numberPatch(document, 'data', zone.offset + 38, destination, 2),
    numberPatch(document, 'data', zone.offset + 40, x, 2), numberPatch(document, 'data', zone.offset + 42, z, 2),
  ] };
}

export function moveObject(document: LevelDocument, objectId: number, x: number, z: number, zoneId: number): EditCommand {
  const object = document.objects.find(object => object.id === objectId);
  if (!object) throw new Error('That object no longer exists.');
  const zone = document.zones.find(zone => zone.id === zoneId);
  if (!zone || !zoneContainsPoint(document, zone, x, z)) throw new Error('Place the object inside its selected zone.');
  return { label: `Move object ${objectId}`, patches: [
    numberPatch(document, 'data', object.pointOffset, Math.round(x * 65536), 4),
    numberPatch(document, 'data', object.pointOffset + 4, Math.round(z * 65536), 4),
    numberPatch(document, 'data', object.offset + 12, zoneId, 2),
    numberPatch(document, 'data', object.offset + 26, zoneId, 2),
  ] };
}

export function removeObject(document: LevelDocument, objectId: number): EditCommand {
  const object = document.objects.find(object => object.id === objectId);
  if (!object) throw new Error('That object no longer exists.');
  return { label: `Delete object ${objectId}`, patches: [
    numberPatch(document, 'data', object.offset + 12, -1, 2), numberPatch(document, 'data', object.offset + 26, -1, 2),
  ] };
}

export function setStart(document: LevelDocument, player: 1 | 2, x: number, z: number, zoneId: number): EditCommand {
  const zone = document.zones.find(zone => zone.id === zoneId);
  if (!zone || !zoneContainsPoint(document, zone, x, z)) throw new Error('Place the player start inside its selected zone.');
  const offset = player === 1 ? 0 : 6;
  const patches = [numberPatch(document, 'data', offset, x, 2), numberPatch(document, 'data', offset + 2, z, 2),
    numberPatch(document, 'data', offset + 4, zoneId, 2)];
  // The multiplayer avatar uses a separate object record and object-point slot.
  const data = new BinaryReader(document.files.data, 'twolev.bin');
  const avatar = data.i32(player === 1 ? 46 : 50), cid = data.i16(avatar);
  const pointOffset = document.objectPointsOffset + cid * 8;
  patches.push(numberPatch(document, 'data', avatar + 12, zoneId, 2), numberPatch(document, 'data', avatar + 26, zoneId, 2),
    numberPatch(document, 'data', pointOffset, x * 65536, 4), numberPatch(document, 'data', pointOffset + 4, z * 65536, 4));
  return { label: `Move player ${player} start`, patches };
}

export function setMechanismRules(document: LevelDocument, kind: 'door' | 'lift', id: number, conditions: number, openRule: number, closeRule: number): EditCommand {
  const mechanism = document.mechanisms.find(mechanism => mechanism.kind === kind && mechanism.id === id);
  if (!mechanism) throw new Error('That mechanism no longer exists.');
  return { label: `Change ${kind} ${id} rules`, patches: [
    numberPatch(document, 'graphics', mechanism.offset + 14, conditions, 2, false),
    numberPatch(document, 'graphics', mechanism.offset + 16, openRule, 1, false),
    numberPatch(document, 'graphics', mechanism.offset + 17, closeRule, 1, false),
  ] };
}
