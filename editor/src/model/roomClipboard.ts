import { importedAnimationTables } from './animationBuild';
import { pasteNavigation } from './navigation';
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
    return { ...room, ...(room.cornerLights ? { cornerLights: room.cornerLights.map(light => ({ ...light, x: light.x + x, z: light.z + z })) } : {}), ...(room.portals ? { portals: room.portals.map(state => ({ ...state, neighbour: mapping.get(state.neighbour) ?? state.neighbour })) } : {}), id: mapping.get(id)!, originZone: null, vertices: room.vertices.map(point => ({ x: point.x + x, z: point.z + z })),
      heights: Object.fromEntries(roomHeightFields.map(field => [field, settings[field]])), properties: captureRoomProperties(settings) };
  });
  const snapshot = roomSourceSnapshot(document.roomSource);
  const cornerLights = added.some(room => room.cornerLights?.length);
  const wallRepeat = added.some(room => room.wallSettings?.some(item => item.repeat));
  const wallSettings = added.some(room => room.wallSettings?.length);
  const switches = added.some(room => room.switches?.length);
  const water = added.some(room => room.waterSurface !== undefined);
  const waterGroups = [...(document.roomSource.waterGroups ?? [])];
  const nativeWater = importedAnimationTables(document).tables.water;
  for (const room of added) {
    const surface = room.waterSurface;
    if (!surface || surface.group === null || waterGroups.some(group => group.id === surface.group) || nativeWater[surface.group]?.entries.length) continue;
    const group = source.roomSource.waterGroups?.find(group => group.id === surface.group);
    if (!group) throw new Error('The copied water animation group is missing. Restore it before pasting.');
    waterGroups.push(group);
  }
  const moving = added.some(room => room.moving);
  const polygons = added.some(room => room.polygonMaterials?.length);
  const boundary = added.some(room => room.boundaryMaterials);
  const portals = added.some(room => room.portals?.length);
  const authored = added.some(room => room.upperStorey !== undefined || room.sky || room.planeSources);
  const painted = added.some(room => room.retiredCopiedSwitches?.length) || cornerLights || wallSettings || water || authored || moving || polygons || boundary || added.some(room => room.wallMaterials?.length);
  const roomSource = restoreRoomSource(document, { ...snapshot, version: Math.max(snapshot.version, added.some(room => room.retiredCopiedSwitches?.length || room.switches?.some(item => item.settings !== undefined)) ? 23 : added.some(room => room.switches?.some(item => item.native !== undefined)) ? 22 : added.some(room => room.portals?.some(item => item.barrier)) ? 21 : cornerLights ? 20 : wallRepeat ? 19 : wallSettings ? 18 : switches ? 17 : water || waterGroups.length ? 16 : portals ? 13 : authored ? 12 : moving ? 10 : polygons ? 9 : boundary ? 8 : painted ? 7 : 4),
    ...(painted ? { controllerCopies: snapshot.controllerCopies ?? [], switchCopies: snapshot.switchCopies ?? [] } : {}),
    ...(waterGroups.length ? { waterGroups } : {}), rooms: [...document.roomSource.rooms, ...added], nextId: document.roomSource.nextId + added.length });
  validateRoomInteriors(roomSource.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  let working: LevelDocument = { ...document, roomSource };
  const navigation = pasteNavigation(working, source, mapping, x, z);
  if (navigation) working = { ...working, roomSource: { ...working.roomSource, navigation } };
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
