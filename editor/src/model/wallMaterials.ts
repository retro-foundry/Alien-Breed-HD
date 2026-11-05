import { authoredMovingSurfaceOwner } from './authoredMovingRoom';
import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { RoomChange, RoomSource } from './roomSource';
import type { PlayableLevel } from './playableBuild';
import type { NativeSurfacePick } from './engineSurfacePick';
import type { CellGraphics } from './cellGraphics';
import { resolveEngineSurface } from './engineSurfacePick';
import { readGraphicRecord } from './graphics';
import { encodeWallSurface, type WallSurface } from './surfaces';
import catalog from './nativeWalls.json' with { type: 'json' };
import { restoreBoundaryMaterials } from './boundaryMaterials';
import { wallSettingsCurrent } from './wallSettings';

export interface WallMaterial {
  readonly record: number; readonly upper: boolean; readonly texture: number;
  readonly verticalMask: number; readonly verticalShift: number; readonly horizontalMask: number;
}
export interface WallMaterialTarget { readonly room: string; readonly record: number; readonly upper: boolean }
export interface BoundaryMaterialTarget { readonly room: string; readonly index: number; readonly upper: boolean; readonly inherited?: WallSurface }
export type SurfaceMaterialTarget = WallMaterialTarget | BoundaryMaterialTarget;
// renderer.c SWITCHES_WALL_TEX_ID: control flags and animation use this sheet specially.
const switchTexture = 11;
const key = (item: Pick<WallMaterial, 'record' | 'upper'>) => `${item.upper}:${item.record}`;

/** Match renderer.c's signed FROMTILE, zero-field fallback and packed strip
 * addressing. Check the whole mask, including columns revealed during motion. */
export function validateWallMaterialSampling(wall: WallSurface, material: Pick<WallMaterial, 'texture' | 'verticalMask' | 'verticalShift' | 'horizontalMask'>, horizontalOffset = wall.horizontalOffset): void {
  const sheet = catalog.walls[material.texture];
  if (!sheet) throw new Error('The wall material sheet is missing.');
  const fallback = material.verticalMask === 0 || material.verticalShift === 0;
  const shift = fallback ? Math.log2(sheet.rows) : material.verticalShift;
  const mask = fallback ? sheet.rows - 1 : material.verticalMask;
  const phase = (horizontalOffset * 16 << 16) >> 16;
  // Reject negative AND masks: their sign bit can address before the sheet.
  const lastColumn = phase + material.horizontalMask;
  const lastByte = Math.floor(lastColumn / 3) * 2 ** (shift + 1) + mask * 2 + 1;
  if (!Number.isInteger(shift) || shift < 1 || shift > 14 || material.horizontalMask < 0
    || phase < 0 || lastByte >= sheet.size - 2048)
    throw new Error('This material does not fit the wall’s retained texture alignment. Choose a narrower repeat, a smaller horizontal tile offset or a compatible material style; Inherit original repeat clears the repeat override.');
}

function validateSourceSampling(document: LevelDocument, wall: WallSurface, material: WallMaterial, room: string) {
  const settings = wallSettingsCurrent(document, { room, record: material.record, upper: material.upper });
  const column = settings?.horizontalOffset, sampling = settings?.repeat ? { ...material, horizontalMask: settings.repeat.width - 1 } : material;
  validateWallMaterialSampling(wall, sampling, column ?? wall.horizontalOffset);
  if (column !== undefined) return; // The controller base is rebuilt with this explicit tile.
  // LiftRoutine may replace FROMTILE with the wall-list base phase.
  for (const mechanism of document.mechanisms) if (mechanism.kind === 'lift')
    for (const entry of mechanism.walls) if (entry.graphic === material.record)
      validateWallMaterialSampling(wall, sampling, entry.base >> 16);
}

export function restoreWallMaterials(value: unknown): readonly WallMaterial[] {
  if (!Array.isArray(value) || value.length > 8192) throw new Error('Invalid source wall material list.');
  const seen = new Set<string>();
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || typeof item.upper !== 'boolean'
      || Object.keys(item).some(field => !['record', 'upper', 'texture', 'verticalMask', 'verticalShift', 'horizontalMask'].includes(field))) throw new Error('Invalid source wall material.');
    const material = Object.freeze({ record: checkedInteger(item.record, 0, 16777215, 'Wall source record'), upper: item.upper,
      texture: checkedInteger(item.texture, 0, catalog.walls.length - 1, 'Wall material'),
      verticalMask: checkedInteger(item.verticalMask, 0, 255, 'Wall vertical mask'), verticalShift: checkedInteger(item.verticalShift, 0, 255, 'Wall vertical shift'),
      horizontalMask: checkedInteger(item.horizontalMask, -32768, 32767, 'Wall horizontal mask') });
    if (seen.has(key(material))) throw new Error('A source wall has duplicate material overrides.');
    seen.add(key(material)); return material;
  }));
}

/** Generated offsets are resolved once against the exact artifact, then discarded.
 * Source room + native record + stream survive renumbering and copied rooms. */
export function wallMaterialTarget(built: PlayableLevel, zone: number, pick: NativeSurfacePick): WallMaterialTarget | undefined {
  if (authoredMovingSurfaceOwner(built, zone, pick)) return;
  const surface = resolveEngineSurface(built.document, zone, pick);
  if (!surface || ![0, 13].includes(surface.type)) return;
  const room = built.topology ? built.source.roomSource.rooms.find(room => room.id === built.topology!.cells.find(cell => cell.id === zone)?.sourceId)
    : built.source.roomSource.rooms.find(room => room.originZone === zone);
  if (!room) return;
  const record = built.topology ? built.surfaceSources?.find(record => record.room === zone && record.upper === pick.upper && record.offset === pick.offset)?.sourceOffset : pick.offset;
  const template = built.source.zones.find(zone => zone.id === room.recordZone);
  const original = (pick.upper ? template?.upper : template?.lower)?.find(item => item.offset === record && item.wall);
  if (record === undefined || !original || original.wall!.texture === switchTexture
    || built.source.switches.some(control => control.zone >= 0 && control.graphic - 10 === record)) return;
  return Object.freeze({ room: room.id, upper: pick.upper, record });
}

export function surfaceMaterialTarget(built: PlayableLevel, zone: number, pick: NativeSurfacePick): SurfaceMaterialTarget | undefined {
  if (authoredMovingSurfaceOwner(built, zone, pick)) return;
  const native = wallMaterialTarget(built, zone, pick);
  if (native) return native;
  const wall = resolveEngineSurface(built.document, zone, pick);
  if (!wall || ![0, 13].includes(wall.type) || wall.texture === switchTexture) return;
  const identity = built.surfaceSources?.find(item => item.room === zone && item.upper === pick.upper && item.offset === pick.offset);
  const room = built.source.roomSource.rooms.find(room => room.id === built.topology?.cells.find(cell => cell.id === zone)?.sourceId);
  if (!room || identity?.boundaryIndex === undefined || identity.sourceOffset !== undefined || !room.vertices[identity.boundaryIndex]) return;
  const appearance=built.source.roomAppearances?.find(item=>item.room===room.id),style=pick.upper?appearance?.upperWall:appearance?.lowerWall;
  const inherited=style==null?(pick.upper?built.document.zones[zone]!.upper:built.document.zones[zone]!.lower).find(record=>record.offset===pick.offset)?.wall:undefined;
  return Object.freeze({ room: room.id, upper: pick.upper, index: identity.boundaryIndex, ...(inherited?{inherited:Object.freeze({...inherited})}:{}) });
}

export function surfaceMaterialCurrent(document: LevelDocument, target: SurfaceMaterialTarget) {
  const room = document.roomSource.rooms.find(room => room.id === target.room)!;
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  if ('record' in target) return room.wallMaterials?.find(item => item.upper === target.upper && item.record === target.record)
    ?? (target.upper ? template.upper : template.lower).find(item => item.offset === target.record)!.wall!;
  const edge = room.boundaryMaterials?.edges[target.index];
  const painted = room.boundaryMaterials?.overrides.find(item => item.edge === edge && item.upper === target.upper);
  const appearance = document.roomAppearances?.find(item => item.room === room.id);
  const record = target.upper ? appearance?.upperWall : appearance?.lowerWall;
  return painted ?? target.inherited ?? (target.upper ? template.upper : template.lower).find(item => item.offset === record)?.wall;
}

export function hasSurfaceMaterialOverride(document: LevelDocument, target: SurfaceMaterialTarget): boolean {
  const room = document.roomSource.rooms.find(room => room.id === target.room)!;
  return 'record' in target ? !!room.wallMaterials?.some(item => item.upper === target.upper && item.record === target.record)
    : !!room.boundaryMaterials?.overrides.some(item => item.edge === room.boundaryMaterials!.edges[target.index] && item.upper === target.upper);
}

export function setSurfaceMaterial(document: LevelDocument, target: SurfaceMaterialTarget, style: number | null): RoomChange {
  if ('record' in target) return setWallMaterial(document, target, style);
  const source = document.roomSource, room = source.rooms.find(room => room.id === target.room);
  if (!room || !Number.isInteger(target.index) || target.index < 0 || !room.vertices[target.index]) throw new Error('The painted boundary no longer exists.');
  const previous = room.boundaryMaterials ?? restoreBoundaryMaterials({ nextId: room.vertices.length,
    edges: room.vertices.map((_, index) => index), overrides: [] }, room.vertices.length);
  const edge = previous.edges[target.index]!;
  let overrides = previous.overrides.filter(item => item.edge !== edge || item.upper !== target.upper);
  if (style !== null) {
    const appearance = document.roomAppearances?.find(item => item.room === room.id);
    const original = document.zones.find(zone => zone.id === room.recordZone);
    const record = target.upper ? appearance?.upperWall : appearance?.lowerWall;
    const wall = target.inherited ?? (target.upper ? original?.upper : original?.lower)?.find(item => item.offset === record)?.wall;
    const selected = wallMaterialChoices(document).find(item => item.offset === style)?.wall;
    if (!wall || !selected) throw new Error('Choose an available boundary appearance and native material style.');
    if (wall.texture === switchTexture) throw new Error('This boundary uses switch graphics. Keep its native material.');
    const material = { edge, upper: target.upper, texture: selected.texture, verticalMask: selected.verticalMask,
      verticalShift: selected.verticalShift, horizontalMask: selected.horizontalMask };
    const settings = room.wallSettings?.find(item => 'boundary' in item && item.boundary === edge && item.upper === target.upper);
    validateWallMaterialSampling(wall, settings?.repeat ? { ...material, horizontalMask: settings.repeat.width - 1 } : material, settings?.horizontalOffset ?? wall.horizontalOffset);
    overrides = previous.overrides.some(item => item.edge === edge && item.upper === target.upper)
      ? previous.overrides.map(item => item.edge === edge && item.upper === target.upper ? material : item) : [...overrides, material];
  }
  const boundaryMaterials = restoreBoundaryMaterials({ ...previous, overrides }, room.vertices.length);
  if (JSON.stringify(previous.overrides) === JSON.stringify(boundaryMaterials.overrides)) return { label: 'Paint boundary material', before: source, after: source };
  const updated = Object.freeze({ ...room, boundaryMaterials });
  return { label: style === null ? 'Reset boundary material' : 'Paint boundary material', before: source,
    after: Object.freeze({ ...source, rooms: Object.freeze(source.rooms.map(item => item.id === room.id ? updated : item)) }) };
}

export function wallMaterialChoices(document: LevelDocument) {
  const seen = new Set<string>();
  return document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).filter(record => {
    const wall = record.wall;
    if (!wall || wall.texture < 0 || wall.texture >= catalog.walls.length || wall.texture === switchTexture) return false;
    const style = `${wall.texture}:${wall.verticalMask}:${wall.verticalShift}:${wall.horizontalMask}`;
    if (seen.has(style)) return false;
    seen.add(style); return true;
  });
}

export function setWallMaterial(document: LevelDocument, target: WallMaterialTarget, style: number | null): RoomChange {
  const source = document.roomSource, room = source.rooms.find(room => room.id === target.room);
  if (!room) throw new Error('The painted room no longer exists.');
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const record = (target.upper ? template.upper : template.lower).find(record => record.offset === target.record && record.wall);
  if (!record) throw new Error('The wall source record is missing. Restore the source wall or reset its material.');
  let materials = (room.wallMaterials ?? []).filter(material => key(material) !== key(target));
  if (style !== null) {
    if (record.wall!.texture === switchTexture || document.switches.some(item => item.zone >= 0 && item.graphic - 10 === target.record))
      throw new Error('This wall carries switch graphics. Keep its native switch material.');
    const selected = wallMaterialChoices(document).find(record => record.offset === style)?.wall;
    if (!selected) throw new Error('Choose an available native wall material style.');
    const material = { record: target.record, upper: target.upper, texture: selected.texture,
      verticalMask: selected.verticalMask, verticalShift: selected.verticalShift, horizontalMask: selected.horizontalMask };
    validateSourceSampling(document, record.wall!, material, room.id);
    materials = room.wallMaterials?.some(item => key(item) === key(target))
      ? room.wallMaterials.map(item => key(item) === key(target) ? material : item) : [...materials, material];
  }
  const restored = restoreWallMaterials(materials);
  if (JSON.stringify(restored) === JSON.stringify(room.wallMaterials ?? [])) return { label: 'Paint wall material', before: source, after: source };
  const { wallMaterials: _previous, ...unchanged } = room;
  const updated = Object.freeze({ ...unchanged, ...(restored.length ? { wallMaterials: restored } : {}) });
  return { label: style === null ? 'Reset wall material' : 'Paint wall material', before: source,
    after: Object.freeze({ ...source, rooms: Object.freeze(source.rooms.map(item => item.id === room.id ? updated : item)) }) };
}

export function clearRoomWallMaterials(document: LevelDocument, id: string): RoomChange {
  const source = document.roomSource, room = source.rooms.find(room => room.id === id);
  if (!room) throw new Error('The painted room no longer exists.');
  if (!room.wallMaterials?.length && !room.boundaryMaterials?.overrides.length) return { label: 'Reset painted room walls', before: source, after: source };
  const { wallMaterials: _previous, ...unchanged } = room;
  return { label: 'Reset painted room walls', before: source,
    after: Object.freeze({ ...source, rooms: Object.freeze(source.rooms.map(item => item.id === id ? Object.freeze({ ...unchanged,
      ...(room.boundaryMaterials ? { boundaryMaterials: restoreBoundaryMaterials({ ...room.boundaryMaterials, overrides: [] }, room.vertices.length) } : {}) }) : item)) }) };
}

/** Change only native material/sampling fields. Geometry, phases, strip span,
 * lighting, draw order and animation pointers retain the existing contract. */
export function paintCellWalls(document: LevelDocument, source: RoomSource, graphics: CellGraphics): CellGraphics {
  if (!source.rooms.some(room => room.wallMaterials?.length || room.boundaryMaterials?.overrides.length)) return graphics;
  const applied = new Set<string>(), streams = new Map(graphics.streams);
  for (const cell of graphics.topology.cells) {
    const room = source.rooms.find(room => room.id === cell.sourceId)!, stream = streams.get(cell.id)!;
    const layer = (records: typeof stream.lower, upper: boolean) => records.map(item => {
      const boundary = item.sourceOffset === undefined && item.boundaryIndex !== undefined ? room.boundaryMaterials?.overrides.find(material => material.upper === upper
        && material.edge === room.boundaryMaterials!.edges[item.boundaryIndex!]) : undefined;
      const material = boundary ?? room.wallMaterials?.find(material => material.upper === upper && material.record === item.sourceOffset);
      if (!material) return item;
      const wall = readGraphicRecord(new BinaryReader(item.bytes, 'painted wall'), 0, 2048).wall;
      if (!wall || wall.texture === switchTexture || material.texture === switchTexture
        || ('record' in material && document.switches.some(control => control.zone >= 0 && control.graphic - 10 === material.record))) throw new Error(`Room ${room.id} has an incompatible painted switch/control wall. Reset its material before building.`);
      if ('record' in material) { applied.add(`${room.id}:${key(material)}`); validateSourceSampling(document, wall, material, room.id); }
      else { applied.add(`${room.id}:boundary:${upper}:${material.edge}`);
        const settings = room.wallSettings?.find(item => 'boundary' in item && item.boundary === material.edge && item.upper === upper);
        validateWallMaterialSampling(wall, settings?.repeat ? { ...material, horizontalMask: settings.repeat.width - 1 } : material, settings?.horizontalOffset ?? wall.horizontalOffset); }
      return { ...item, bytes: encodeWallSurface({ ...wall, texture: material.texture, verticalMask: material.verticalMask,
        verticalShift: material.verticalShift, horizontalMask: material.horizontalMask }) };
    });
    streams.set(cell.id, { lower: layer(stream.lower, false), ...(stream.upper ? { upper: layer(stream.upper, true) } : {}) });
  }
  for (const room of source.rooms) for (const material of room.wallMaterials ?? []) if (!applied.has(`${room.id}:${key(material)}`))
    throw new Error(`Room ${room.id} no longer draws painted wall ${material.record}. Restore its outline or reset its wall materials before building.`);
  for (const room of source.rooms) for (const material of room.boundaryMaterials?.overrides ?? []) if (!applied.has(`${room.id}:boundary:${material.upper}:${material.edge}`))
    throw new Error(`Room ${room.id} no longer draws painted boundary ${material.edge}. Restore its outline or reset its wall materials before building.`);
  return { ...graphics, streams };
}
