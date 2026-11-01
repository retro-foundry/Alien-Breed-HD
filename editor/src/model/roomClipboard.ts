import type { LevelDocument } from './level';
import { applyCommand, type EditCommand } from './commands';
import { checkedInteger } from './binary';
import { restoreRoomSource, roomSourceSnapshot } from './roomSource';
import { validateRoomInteriors } from './geometry';
import { roomSettings, roomHeightFields } from './roomHeights';
import { captureRoomProperties } from './roomProperties';
import { roomAppearanceChoices, setRoomAppearance } from './roomAppearance';
import { editableTeleport, placeSourceTeleport } from './sourceTeleports';
import { editableObjects, objectIsActive, objectPlayer } from './sourceObjects';
import { createObject, objectTemplates } from './objectAuthoring';
import { copyRoomControllers } from './roomClipboardControllers';
import { buildPlayableLevel } from './playableBuild';
import { roomCopyDependencies } from './roomCopyDependencies';

export interface RoomClipboard { readonly source: LevelDocument; readonly ids: readonly string[]; readonly requestedIds?: readonly string[] }

export function copyRooms(source: LevelDocument, ids: readonly string[]): RoomClipboard {
  if (!ids.length || ids.length !== new Set(ids).size || ids.some(id => !source.roomSource.rooms.some(room => room.id === id))) throw new Error('Select one or more different existing rooms to copy.');
  return Object.freeze({ source, ids: Object.freeze([...ids]) });
}

export function copyRoomGroup(source: LevelDocument, ids: readonly string[]): RoomClipboard {
  const requested = copyRooms(source, ids);
  return Object.freeze({ ...copyRooms(source, roomCopyDependencies(source, ids)), requestedIds: requested.ids });
}

/** Assemble privately, then validate the complete native artifact. No source,
 * objects, links or history change on a rejected copy. Starts and the global
 * exit remain project configuration rather than duplicated room content. */
export function pasteRooms(document: LevelDocument, clipboard: RoomClipboard, x: number, z: number): { ids: string[]; objects: number[]; command: EditCommand } {
  const source = clipboard.source;
  if (source.original !== document.original) throw new Error('Copied rooms belong to another project. Cross-project resource transfer is not supported yet.');
  copyRooms(source, clipboard.ids);
  checkedInteger(x, -65535, 65535, 'Room paste X offset'); checkedInteger(z, -65535, 65535, 'Room paste Z offset');
  checkedInteger(document.roomSource.rooms.length + clipboard.ids.length, 1, 256, 'Pasted room count');
  checkedInteger(document.roomSource.nextId, 0, Number.MAX_SAFE_INTEGER - clipboard.ids.length - 1, 'Next room identity');
  const mapping = new Map(clipboard.ids.map((id, index) => [id, `room-${document.roomSource.nextId + index}`]));
  const added = clipboard.ids.map(id => {
    const room = source.roomSource.rooms.find(room => room.id === id)!, settings = roomSettings(source, room);
    return { ...room, id: mapping.get(id)!, originZone: null, vertices: room.vertices.map(point => ({ x: point.x + x, z: point.z + z })),
      heights: Object.fromEntries(roomHeightFields.map(field => [field, settings[field]])), properties: captureRoomProperties(settings) };
  });
  const snapshot = roomSourceSnapshot(document.roomSource);
  const boundary = added.some(room => room.boundaryMaterials);
  const painted = boundary || added.some(room => room.wallMaterials?.length);
  const roomSource = restoreRoomSource(document, { ...snapshot, version: Math.max(snapshot.version, boundary ? 8 : painted ? 7 : 4),
    ...(painted ? { controllerCopies: snapshot.controllerCopies ?? [], switchCopies: snapshot.switchCopies ?? [] } : {}),
    rooms: [...document.roomSource.rooms, ...added], nextId: document.roomSource.nextId + added.length });
  validateRoomInteriors(roomSource.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  let working: LevelDocument = { ...document, roomSource };
  for (const [id, mapped] of mapping) {
    const room = source.roomSource.rooms.find(room => room.id === id)!;
    const appearance = source.roomAppearances?.find(item => item.room === id);
    const choices = roomAppearanceChoices(source, room);
    const selected = appearance ? { ...appearance, room: mapped } : choices.lower.length && (choices.template.upper.length === 0 || choices.upper.length) ? { room: mapped, lowerWall: choices.lower[0]!.offset,
      upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! } : undefined;
    if (selected) working = { ...working, roomAppearances: setRoomAppearance(working, selected) };
    const link = editableTeleport(source, room);
    if (link.destination === undefined) throw new Error(`Copied room ${id} has a missing teleport destination. Repair it before pasting.`);
    const internal = link.destination !== null && mapping.has(link.destination);
    working = { ...working, sourceTeleports: placeSourceTeleport(working, mapped,
      link.destination === null ? null : mapping.get(link.destination) ?? link.destination, link.x + (internal ? x : 0), link.z + (internal ? z : 0)) };
  }
  working = copyRoomControllers(working, source, mapping);
  const templates = new Set(objectTemplates(source).map(object => object.id)), objects: number[] = [];
  for (const object of editableObjects(source)) {
    const room = object.room ?? source.roomSource.rooms.find(room => room.originZone === object.zone)?.id;
    if (!room || !mapping.has(room) || !objectIsActive(object) || objectPlayer(source, object) !== undefined) continue;
    if (!templates.has(object.id)) throw new Error(`Object ${object.id} in the copied group is a reserved or unsupported source record. No rooms were pasted.`);
    const proposal = createObject(working, { template: object.id, room: mapping.get(room)!, x: object.x + x, z: object.z + z, y: object.y, upper: object.upper }, source);
    working = applyCommand(working, proposal.command); objects.push(proposal.id);
  }
  buildPlayableLevel(working, -1);
  return { ids: [...mapping.values()], objects, command: { label: `Paste ${added.length} rooms`, patches: [],
    roomSource: { before: document.roomSource, after: working.roomSource }, files: { before: document.files, after: working.files },
    sourceObjects: { before: document.sourceObjects, after: working.sourceObjects }, sourceTeleports: { before: document.sourceTeleports, after: working.sourceTeleports },
    roomAppearances: { before: document.roomAppearances, after: working.roomAppearances } } };
}
