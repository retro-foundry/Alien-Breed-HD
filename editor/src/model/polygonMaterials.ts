import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { RoomChange, RoomSource } from './roomSource';
import type { PlayableLevel } from './playableBuild';
import type { NativeSurfacePick } from './engineSurfacePick';
import type { CellGraphics } from './cellGraphics';
import { resolveEngineSurface } from './engineSurfacePick';
import { readGraphicRecord } from './graphics';
import { encodePolygonSurface, polygonPlane, polygonMaterialBank, polygonMaterialType, type PolygonMaterialMode } from './surfaces';

export interface PolygonMaterial { readonly record: number; readonly upper: boolean; readonly texture: number; readonly scale: number; readonly mode?: PolygonMaterialMode }
export interface PolygonMaterialTarget { readonly room: string; readonly record: number; readonly upper: boolean }
export const usesFloorBank = (type: number) => [1, 2].includes(type);
export const isPaintablePolygon = (type: number) => polygonPlane(type) !== undefined;
const key = (item: Pick<PolygonMaterial, 'record' | 'upper'>) => `${item.upper}:${item.record}`;

export function restorePolygonMaterials(value: unknown): readonly PolygonMaterial[] {
  if (!Array.isArray(value) || value.length > 8192) throw new Error('Invalid source floor/ceiling material list.');
  const seen = new Set<string>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || typeof item.upper !== 'boolean'
      || Object.keys(item).some(field => !['record', 'upper', 'texture', 'scale', 'mode'].includes(field))) throw new Error('Invalid source floor/ceiling material.');
    // renderer.c FloorLine addresses (u & 0xFC) | (v & 0xFC00), plus
    // WHICHTILE, in io.c's 65536-byte floortile bank. Check every sample.
    if ('mode' in item && !['normal', 'bump', 'smooth'].includes(item.mode as string)) throw new Error('Choose a native normal, bump or smooth material mode.');
    const material = Object.freeze({ ...('mode' in item ? { mode: item.mode as PolygonMaterialMode } : {}), record: checkedInteger(item.record, 0, 16777215, 'Polygon source record'), upper: item.upper,
      texture: checkedInteger(item.texture, 0, 65535 - 0xFCFC, 'Floor bank tile offset'),
      scale: checkedInteger(item.scale, -32768, 32767, 'Floor texture scale') });
    if (seen.has(key(material))) throw new Error('A source polygon has duplicate material overrides.');
    seen.add(key(material)); return material;
  }));
}

export function polygonMaterialTarget(built: PlayableLevel, zone: number, pick: NativeSurfacePick): PolygonMaterialTarget | undefined {
  const surface = resolveEngineSurface(built.document, zone, pick);
  if (!surface || !isPaintablePolygon(surface.type)) return;
  const room = built.topology ? built.source.roomSource.rooms.find(room => room.id === built.topology!.cells.find(cell => cell.id === zone)?.sourceId)
    : built.source.roomSource.rooms.find(room => room.originZone === zone);
  if (!room) return;
  const record = built.topology ? built.surfaceSources?.find(item => item.room === zone && item.upper === pick.upper && item.offset === pick.offset)?.sourceOffset : pick.offset;
  const template = built.source.zones.find(zone => zone.id === room.recordZone);
  if (record === undefined || !(pick.upper ? template?.upper : template?.lower)?.some(item => item.offset === record && isPaintablePolygon(item.type))) return;
  return Object.freeze({ room: room.id, upper: pick.upper, record });
}

export function polygonMaterialCurrent(document: LevelDocument, target: PolygonMaterialTarget) {
  const room = document.roomSource.rooms.find(room => room.id === target.room)!;
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  return room.polygonMaterials?.find(item => key(item) === key(target))
    ?? (target.upper ? template.upper : template.lower).find(item => item.offset === target.record)?.polygon;
}

export function polygonMaterialMode(document: LevelDocument, target: PolygonMaterialTarget): PolygonMaterialMode {
  const room = document.roomSource.rooms.find(room => room.id === target.room)!;
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const polygon = (target.upper ? template.upper : template.lower).find(item => item.offset === target.record)!.polygon!;
  return room.polygonMaterials?.find(item => key(item) === key(target))?.mode ?? polygonMaterialBank(polygon.type);
}

export function setPolygonMaterial(document: LevelDocument, target: PolygonMaterialTarget, value: Pick<PolygonMaterial, 'texture' | 'scale' | 'mode'> | null): RoomChange {
  const source = document.roomSource, room = source.rooms.find(room => room.id === target.room);
  const template = document.zones.find(zone => zone.id === room?.recordZone);
  const original = (target.upper ? template?.upper : template?.lower)?.find(item => item.offset === target.record)?.polygon;
  if (!room || !original || !isPaintablePolygon(original.type)) throw new Error('Choose an existing floor or ceiling material surface.');
  let materials = (room.polygonMaterials ?? []).filter(item => key(item) !== key(target));
  if (value !== null) {
    const material = restorePolygonMaterials([{ record: target.record, upper: target.upper, texture: value.texture, scale: value.scale, ...(value.mode ? { mode: value.mode } : {}) }])[0]!;
    materials = room.polygonMaterials?.some(item => key(item) === key(target))
      ? room.polygonMaterials.map(item => key(item) === key(target) ? material : item) : [...materials, material];
  }
  const restored = restorePolygonMaterials(materials);
  if (JSON.stringify(restored) === JSON.stringify(room.polygonMaterials ?? [])) return { label: 'Paint floor/ceiling material', before: source, after: source };
  const { polygonMaterials: _previous, ...unchanged } = room;
  const updated = Object.freeze({ ...unchanged, ...(restored.length ? { polygonMaterials: restored } : {}) });
  return { label: value === null ? 'Reset floor/ceiling material' : 'Paint floor/ceiling material', before: source,
    after: Object.freeze({ ...source, rooms: Object.freeze(source.rooms.map(item => item.id === room.id ? updated : item)) }) };
}

export function clearRoomPolygonMaterials(document: LevelDocument, id: string): RoomChange {
  const source = document.roomSource, room = source.rooms.find(room => room.id === id);
  if (!room) throw new Error('The painted room no longer exists.');
  if (!room.polygonMaterials?.length) return { label: 'Reset painted room surfaces', before: source, after: source };
  const { polygonMaterials: _previous, ...unchanged } = room;
  return { label: 'Reset painted room surfaces', before: source,
    after: Object.freeze({ ...source, rooms: Object.freeze(source.rooms.map(item => item.id === id ? Object.freeze(unchanged) : item)) }) };
}

/** Change the tile offset and authored scale, leaving type, plane, footprint,
 * lighting and source identity available to controller relocation. */
export function paintCellPolygons(source: RoomSource, graphics: CellGraphics): CellGraphics {
  if (!source.rooms.some(room => room.polygonMaterials?.length)) return graphics;
  const applied = new Set<string>(), streams = new Map(graphics.streams);
  for (const cell of graphics.topology.cells) {
    const room = source.rooms.find(room => room.id === cell.sourceId)!, stream = streams.get(cell.id)!;
    const layer = (records: typeof stream.lower, upper: boolean) => records.map(item => {
      const material = room.polygonMaterials?.find(material => material.upper === upper && material.record === item.sourceOffset);
      if (!material) return item;
      restorePolygonMaterials([material]);
      const polygon = readGraphicRecord(new BinaryReader(item.bytes, 'painted polygon'), 0, 2048).polygon;
      if (!polygon || !isPaintablePolygon(polygon.type)) throw new Error(`Room ${room.id} has an incompatible painted polygon. Reset its surface material before building.`);
      applied.add(`${room.id}:${key(material)}`);
      return { ...item, bytes: encodePolygonSurface({ ...polygon, texture: material.texture, scale: material.scale, ...(material.mode ? { type: polygonMaterialType(polygon.type, material.mode) } : {}) }) };
    });
    streams.set(cell.id, { lower: layer(stream.lower, false), ...(stream.upper ? { upper: layer(stream.upper, true) } : {}) });
  }
  for (const room of source.rooms) for (const material of room.polygonMaterials ?? []) if (!applied.has(`${room.id}:${key(material)}`))
    throw new Error(`Room ${room.id} no longer draws painted surface ${material.record}. Restore its source or reset its surface materials before building.`);
  return { ...graphics, streams };
}
