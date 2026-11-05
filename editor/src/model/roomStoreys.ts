import { mechanismsForRoom } from './mechanismDependencies';
import { roomPlaneRecords } from './roomPlanes';
import { checkedInteger } from './binary';
import type { LevelDocument } from './level';
import { normaliseRoomOutline } from './geometry';
import { roomSettings, restoreRoomHeights } from './roomHeights';
import type { RoomChange, SourceRoom } from './roomSource';
import { editableObjects, objectIsActive } from './sourceObjects';

export interface UpperStorey { readonly wall: number; readonly lightingPoint: number; readonly floorMaterial: number; readonly ceilingMaterial: number }

export function restoreStorey(value: unknown): UpperStorey | false {
  if (value === false) return false;
  if (!value || typeof value !== 'object' || Object.keys(value).length !== 4 || !('wall' in value) || typeof value.wall !== 'number' || !('lightingPoint' in value) || typeof value.lightingPoint !== 'number' || !('floorMaterial' in value) || typeof value.floorMaterial !== 'number' || !('ceilingMaterial' in value) || typeof value.ceilingMaterial !== 'number') throw new Error('An upper storey needs wall, lighting, floor and ceiling material references.');
  return Object.freeze({ wall: checkedInteger(value.wall, 0, 16777215, 'Upper wall style'), lightingPoint: checkedInteger(value.lightingPoint, 0, 2047, 'Upper lighting point'), floorMaterial: checkedInteger(value.floorMaterial, 0, 16777215, 'Upper floor material'), ceilingMaterial: checkedInteger(value.ceilingMaterial, 0, 16777215, 'Upper ceiling material') });
}

/** AMOS DEFUPPER* / LEVELSAVE: two independent air bands, graphics streams,
 * and floor/roof polygons. New upper rooms copy explicitly selected existing
 * wall sampling and explicit normal floor/roof material references. */
export function storeyMaterials(document: LevelDocument, room: SourceRoom) {
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const outline = JSON.stringify(normaliseRoomOutline(template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z }))));
  const whole = (type: number) => roomPlaneRecords(document, room, false).find(record => record.type === type && record.polygon
    && (room.planeSources?.some(item => !item.upper && item.record === record.offset) || JSON.stringify(normaliseRoomOutline(record.points.map(id => ({ x: document.points[id]!.x, z: document.points[id]!.z })))) === outline));
  const ownWalls = template.lower.filter(record => record.wall && record.type === 0);
  const otherWalls = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).filter(record => record.wall && record.type === 0 && !ownWalls.some(own => own.offset === record.offset));
  return { walls: [...ownWalls, ...otherWalls], floor: whole(1), ceiling: whole(2) };
}

export function assertStaticRoom(document: LevelDocument, room: SourceRoom) {
  if (room.switches?.length || room.moving || mechanismsForRoom(document, room.recordZone).length > 0
    || document.switches.some(item => item.zone === room.recordZone && item.zone >= 0)
    || document.roomSource.controllerCopies?.some(item => item.room === room.id))
    throw new Error('This room belongs to a door, lift or switch. Remove its mechanism before changing storeys or ceiling openings.');
}

export function setUpperStorey(document: LevelDocument, id: string, value: { floor: number; ceiling: number; wall: number; lightingPoint: number; floorMaterial?: number; ceilingMaterial?: number } | null): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select a room before changing its storeys.');
  assertStaticRoom(document, room);
  const settings = roomSettings(document, room);
  if (!value && editableObjects(document).some(object => objectIsActive(object) && object.upper && (object.room === id || object.room === undefined && object.zone === room.originZone)))
    throw new Error('Move or delete the objects on this upper storey before removing it.');
  if (value && settings.upper.length > 0 && settings.upperFloor > settings.upperCeiling) throw new Error('This room already has an upper storey. Edit its heights below.');
  const materials = storeyMaterials(document, room);
  const records = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]);
  const floorMaterial = value?.floorMaterial ?? materials.floor?.offset, ceilingMaterial = value?.ceilingMaterial ?? materials.ceiling?.offset;
  if (value && (!records.some(record => record.offset === floorMaterial && record.type === 1) || !records.some(record => record.offset === ceilingMaterial && record.type === 2)
    || !materials.walls.some(record => record.offset === value.wall) || !document.points.some(point => point.id === value.lightingPoint)))
    throw new Error('Choose normal upper floor and ceiling patterns, a straight wall style and an existing lighting point.');
  const upperStorey = value ? restoreStorey({ wall: value.wall, lightingPoint: value.lightingPoint, floorMaterial, ceilingMaterial }) : false;
  const heights = restoreRoomHeights({ floor: settings.floor, ceiling: settings.ceiling,
    upperFloor: value?.floor ?? settings.ceiling, upperCeiling: value?.ceiling ?? settings.ceiling }, document.zones.find(zone => zone.id === room.recordZone)!, upperStorey);
  const lowerRecords = new Set([...roomPlaneRecords(document, room, false), ...settings.lower].map(item => item.offset));
  const upperRecords = new Set([...roomPlaneRecords(document, room, true), ...settings.upper].map(item => item.offset));
  const { polygonMaterials, wallMaterials, wallSettings, planeSources, cornerLights, ...retained } = room;
  const keptSettings = wallSettings?.filter(item => !item.upper);
  const keptLights = cornerLights?.filter(item => item.record === undefined || !upperRecords.has(item.record) || lowerRecords.has(item.record));
  const updated = Object.freeze({ ...retained, upperStorey, heights,
    sky: Object.freeze([value ? false : room.sky?.[0] ?? false, false]),
    ...(planeSources ? { planeSources: Object.freeze(planeSources.filter(item => !item.upper)) } : {}),
    ...(polygonMaterials ? { polygonMaterials: Object.freeze(polygonMaterials.filter(item => !item.upper)) } : {}),
    ...(keptSettings?.length ? { wallSettings: Object.freeze(keptSettings) } : {}),
    ...(wallMaterials ? { wallMaterials: Object.freeze(wallMaterials.filter(item => !item.upper)) } : {}),
    ...(keptLights?.length ? { cornerLights: Object.freeze(keptLights) } : {}),
    ...(room.boundaryMaterials ? { boundaryMaterials: { ...room.boundaryMaterials, overrides: room.boundaryMaterials.overrides.filter(item => !item.upper) } } : {}) });
  return { label: value ? 'Add upper storey' : 'Remove upper storey', before, after: Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === id ? updated : item)) }) };
}

/** RGPUT's NONE (SKY), LEVELSAVE's BACK flag and renderer opcode 12. This
 * changes saved graphics, independently of the editor's visibility checkbox. */
export function setSkyCeiling(document: LevelDocument, id: string, upper: boolean, open: boolean): RoomChange {
  const before = document.roomSource, room = before.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select a room before changing its ceiling.');
  assertStaticRoom(document, room);
  const settings = roomSettings(document, room), hasUpper = settings.upper.length > 0 && settings.upperFloor > settings.upperCeiling;
  if (upper && !hasUpper) throw new Error('Add an upper storey before opening its ceiling.');
  if (!upper && hasUpper && open) throw new Error('Open the upper ceiling to the sky, or use Create opening between storeys for a hole in the lower ceiling.');
  if (!open && !roomPlaneRecords(document, room, upper).some(record => [2, 9, 11].includes(record.type))) throw new Error('This imported room has no ceiling material. Choose Create ceiling under Textures first.');
  const sky = [room.sky?.[0] ?? false, room.sky?.[1] ?? false]; sky[upper ? 1 : 0] = open;
  const ceilings = new Set(roomPlaneRecords(document, room, upper).filter(item => [2, 9, 11].includes(item.type)).map(item => item.offset));
  const otherStorey = new Set(roomPlaneRecords(document, room, !upper).map(item => item.offset));
  const { cornerLights: _lights, ...retained } = room;
  const lights = room.cornerLights?.filter(light => !open || light.record === undefined || !ceilings.has(light.record) || otherStorey.has(light.record));
  const updated = Object.freeze({ ...retained, sky: Object.freeze(sky), ...(lights?.length ? { cornerLights: Object.freeze(lights) } : {}) });
  return { label: open ? 'Open ceiling to sky' : 'Restore ceiling', before, after: Object.freeze({ ...before, rooms: Object.freeze(before.rooms.map(item => item.id === id ? updated : item)) }) };
}

export function ceilingIsSky(document: LevelDocument, room: SourceRoom, upper: boolean): boolean {
  if (room.sky?.[upper ? 1 : 0] !== undefined) return room.sky[upper ? 1 : 0]!;
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const records = upper ? template.upper : template.lower;
  return !roomPlaneRecords(document, room, upper).some(record => [2, 9, 11].includes(record.type)) && (template.backdrop || records.some(record => record.type === 12));
}
