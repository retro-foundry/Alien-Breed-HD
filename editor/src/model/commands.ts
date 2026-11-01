import { BinaryReader, checkedInteger } from './binary';
import { parseLevel, sameBytes, zoneContainsPoint, type LevelDocument, type LevelFiles } from './level';
import type { RoomChange, RoomSource } from './roomSource';
import { placeSourceStart, type SourceStart } from './sourceStarts';
import { placeSourceObject, objectPlayer, type SourceObject } from './sourceObjects';
import { placeSourceTeleport, type SourceTeleport } from './sourceTeleports';
import { setRoomAppearance, type RoomAppearance } from './roomAppearance';
import { nativeWallSheets, restoreWallOverrides, type WallOverride } from './wallOverrides';

export interface BytePatch { file: keyof LevelFiles; offset: number; before: Uint8Array; after: Uint8Array }
interface Change<T> { before: T; after: T }
export interface EditCommand {
  label: string; patches: BytePatch[];
  playtestSlot?: Change<number>; exitZone?: Change<number | null>; roomSource?: Change<RoomSource>;
  sourceStarts?: Change<readonly SourceStart[] | undefined>; sourceObjects?: Change<readonly SourceObject[] | undefined>;
  sourceTeleports?: Change<readonly SourceTeleport[] | undefined>; roomAppearances?: Change<readonly RoomAppearance[] | undefined>;
  wallOverrides?: Change<readonly WallOverride[] | undefined>;
  files?: Change<LevelFiles>;
}
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
  let playtestSlot = document.playtestSlot;
  let exitZone = document.exitZone;
  let roomSource = document.roomSource;
  let sourceStarts = document.sourceStarts;
  let sourceObjects = document.sourceObjects;
  let sourceTeleports = document.sourceTeleports;
  let roomAppearances = document.roomAppearances;
  let wallOverrides = document.wallOverrides;
  if (command.wallOverrides) {
    const { before, after } = command.wallOverrides;
    if (wallOverrides !== (reverse ? after : before)) throw new Error('The wall assets changed before this edit. No changes were saved.');
    wallOverrides = reverse ? before : after;
  }
  if (command.roomAppearances) {
    const { before, after } = command.roomAppearances;
    if (roomAppearances !== (reverse ? after : before)) throw new Error('The room appearance changed before this edit could be applied. No changes were saved.');
    roomAppearances = reverse ? before : after;
  }
  if (command.sourceTeleports) {
    const { before, after } = command.sourceTeleports;
    if (sourceTeleports !== (reverse ? after : before)) throw new Error('The teleporter links changed before this edit could be applied. No changes were saved.');
    sourceTeleports = reverse ? before : after;
  }
  if (command.sourceObjects) {
    const { before, after } = command.sourceObjects;
    if (sourceObjects !== (reverse ? after : before)) throw new Error('The object placements changed before this edit could be applied. No changes were saved.');
    sourceObjects = reverse ? before : after;
  }
  if (command.sourceStarts) {
    const { before, after } = command.sourceStarts;
    if (sourceStarts !== (reverse ? after : before)) throw new Error('The player starts changed before this edit could be applied. No changes were saved.');
    sourceStarts = reverse ? before : after;
  }
  if (command.roomSource) {
    const { before, after } = command.roomSource;
    if (roomSource !== (reverse ? after : before)) throw new Error('The room source changed before this edit could be applied. No changes were saved.');
    roomSource = reverse ? before : after;
  }
  if (command.exitZone) {
    const { before, after } = command.exitZone;
    if (exitZone !== (reverse ? after : before)) throw new Error('The exit zone changed before this edit could be applied. No changes were saved.');
    exitZone = reverse ? before : after;
  }
  if (command.playtestSlot) {
    const { before, after } = command.playtestSlot;
    if (playtestSlot !== (reverse ? after : before)) throw new Error('The project slot changed before this edit could be applied. No changes were saved.');
    playtestSlot = checkedInteger(reverse ? before : after, 0, 15, 'Game level slot');
  }
  let files = { ...document.files };
  if (command.files) {
    if (command.patches.length) throw new Error('A table rewrite cannot also contain byte patches.');
    const expected = reverse ? command.files.after : command.files.before;
    if ((['data', 'graphics', 'clips'] as const).some(file => !sameBytes(files[file], expected[file])))
      throw new Error('The level tables changed before this edit. No changes were saved.');
    files = { ...(reverse ? command.files.before : command.files.after) };
  }
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
  const result = parseLevel(document.name, files, document.encoded, document.original, playtestSlot, exitZone, roomSource, document.drawGroups);
  return { ...result, ...(sourceStarts === undefined ? {} : { sourceStarts }), ...(sourceObjects === undefined ? {} : { sourceObjects }),
    ...(sourceTeleports === undefined ? {} : { sourceTeleports }), ...(roomAppearances === undefined ? {} : { roomAppearances }),
    ...(wallOverrides === undefined ? {} : { wallOverrides }) };
}

export function execute(history: History, command: EditCommand): History {
  const changes = [command.playtestSlot, command.exitZone, command.roomSource, command.sourceStarts, command.sourceObjects, command.sourceTeleports, command.roomAppearances, command.wallOverrides, command.files];
  if (command.patches.every(patch => sameBytes(patch.before, patch.after)) && changes.every(change => !change || change.before === change.after)) return history;
  const document = applyCommand(history.document, command);
  return { document, past: [...history.past, command], future: [] };
}

export function undo(history: History): History {
  const command = history.past.at(-1);
  if (!command) return history;
  return { document: applyCommand(history.document, command, true), past: history.past.slice(0, -1), future: [...history.future, command] };
}

export function replaceWallAsset(document: LevelDocument, texture: number, bytes: Uint8Array | null, sourcePng?: Uint8Array): EditCommand {
  if (!Number.isInteger(texture) || !nativeWallSheets[texture]) throw new Error('Choose an existing native wall material slot.');
  const old = document.wallOverrides?.find(asset => asset.texture === texture);
  if ((bytes === null && !old) || (bytes && old && sameBytes(bytes, old.bytes)
    && (sourcePng === undefined || sourcePng === old.sourcePng || (sourcePng && old.sourcePng && sameBytes(sourcePng, old.sourcePng)))))
    return { label: 'Replace wall material', patches: [] };
  const remaining = (document.wallOverrides ?? []).filter(asset => asset.texture !== texture);
  const next = bytes ? [...remaining, { texture, bytes, ...(sourcePng ? { sourcePng } : {}) }] : remaining;
  return { label: bytes ? 'Replace wall material' : 'Restore original wall material', patches: [],
    wallOverrides: { before: document.wallOverrides, after: next.length ? restoreWallOverrides(next) : undefined } };
}

export function redo(history: History): History {
  const command = history.future.at(-1);
  if (!command) return history;
  return { document: applyCommand(history.document, command), past: [...history.past, command], future: history.future.slice(0, -1) };
}

export function setPlaytestSlot(document: LevelDocument, slot: number): EditCommand {
  checkedInteger(slot, 0, 15, 'Game level slot');
  return { label: `Use game level slot ${slot + 1}`, patches: [], playtestSlot: { before: document.playtestSlot, after: slot } };
}

export function setExitZone(document: LevelDocument, zone: number | null): EditCommand {
  if (zone !== null && zone !== -1 && !document.zones.some(item => item.id === zone)) throw new Error('Choose an existing exit zone.');
  return { label: zone === null ? 'Use original slot exit' : zone === -1 ? 'Disable level exit' : `Exit at zone ${zone}`, patches: [], exitZone: { before: document.exitZone, after: zone } };
}

export function editRoomSource(change: RoomChange): EditCommand {
  return { label: change.label, patches: [], roomSource: { before: change.before, after: change.after } };
}

export function setSourceStart(document: LevelDocument, player: 1 | 2, room: string, x: number, z: number): EditCommand {
  return { label: `Move player ${player} start`, patches: [], sourceStarts: {
    before: document.sourceStarts, after: placeSourceStart(document, player, room, x, z),
  } };
}

export function setSourceObject(document: LevelDocument, id: number, room: string | null, x: number, z: number): EditCommand {
  return { label: room === null ? `Delete object ${id}` : `Move object ${id}`, patches: [], sourceObjects: {
    before: document.sourceObjects, after: placeSourceObject(document, id, room, x, z),
  } };
}

export function setSourceTeleport(document: LevelDocument, room: string, destination: string | null, x: number, z: number): EditCommand {
  return { label: `Change room ${room.replace(/^room-/, '')} teleporter`, patches: [], sourceTeleports: {
    before: document.sourceTeleports, after: placeSourceTeleport(document, room, destination, x, z),
  } };
}

export function editRoomAppearance(document: LevelDocument, appearance: RoomAppearance): EditCommand {
  return { label: 'Change room appearance', patches: [], roomAppearances: {
    before: document.roomAppearances, after: setRoomAppearance(document, appearance),
  } };
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
  const room = document.roomSource.rooms.find(room => room.originZone === zoneId);
  if (room && document.sourceTeleports?.some(item => item.room === room.id)) {
    const target = destination === -1 ? null : document.roomSource.rooms.find(room => room.originZone === destination);
    if (target === undefined) throw new Error('Choose an existing editable destination room, or disable the teleporter.');
    return setSourceTeleport(document, room.id, target === null ? null : target.id, x, z);
  }
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
  assertNotAuthoredAvatar(document, object.offset);
  if (document.sourceObjects?.some(item => item.id === objectId)) {
    const room = document.roomSource.rooms.find(room => room.originZone === zoneId);
    if (!room) throw new Error('Choose an existing editable room for this object.');
    return setSourceObject(document, objectId, room.id, x, z);
  }
  const zone = document.zones.find(zone => zone.id === zoneId);
  if (!zone || !zoneContainsPoint(document, zone, x, z)) throw new Error('Place the object inside its selected zone.');
  if (object.upper && !zone.upper.length) throw new Error('This object is on the upper floor. Choose a zone with an upper floor before moving it.');
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
  if (objectPlayer(document, object) !== undefined) throw new Error('Player avatars cannot be deleted. Select the player in Starts to edit its position.');
  assertNotAuthoredAvatar(document, object.offset);
  const authored = document.sourceObjects?.find(item => item.id === objectId);
  if (authored) return setSourceObject(document, objectId, null, authored.x, authored.z);
  return { label: `Delete object ${objectId}`, patches: [
    numberPatch(document, 'data', object.offset + 12, -1, 2), numberPatch(document, 'data', object.offset + 26, -1, 2),
  ] };
}

function assertNotAuthoredAvatar(document: LevelDocument, offset: number): void {
  const reader = new BinaryReader(document.files.data, 'player avatar');
  if (document.sourceStarts?.some(start => reader.i32(start.player === 1 ? 46 : 50) === offset))
    throw new Error('This avatar follows an authored player start. Select the player in Starts to move it.');
}

export function setStart(document: LevelDocument, player: 1 | 2, x: number, z: number, zoneId: number): EditCommand {
  if (document.sourceStarts?.some(start => start.player === player)) {
    const room = document.roomSource.rooms.find(room => room.originZone === zoneId);
    if (!room) throw new Error('Choose an existing editable room for the player start.');
    return setSourceStart(document, player, room.id, x, z);
  }
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
