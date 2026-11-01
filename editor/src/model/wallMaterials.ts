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

export interface WallMaterial {
  readonly record: number; readonly upper: boolean; readonly texture: number;
  readonly verticalMask: number; readonly verticalShift: number; readonly horizontalMask: number;
}
export interface WallMaterialTarget { readonly room: string; readonly record: number; readonly upper: boolean }
// renderer.c SWITCHES_WALL_TEX_ID: control flags and animation use this sheet specially.
const switchTexture = 11;
const key = (item: Pick<WallMaterial, 'record' | 'upper'>) => `${item.upper}:${item.record}`;

/** Match renderer.c's signed FROMTILE, zero-field fallback and packed strip
 * addressing. Check the whole mask, including columns revealed during motion. */
export function validateWallMaterialSampling(wall: WallSurface, material: WallMaterial, horizontalOffset = wall.horizontalOffset): void {
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
    throw new Error('This material does not fit the wall’s retained texture alignment. Choose another material style or reset its material.');
}

function validateSourceSampling(document: LevelDocument, wall: WallSurface, material: WallMaterial) {
  validateWallMaterialSampling(wall, material);
  // LiftRoutine may replace FROMTILE with the wall-list base phase.
  for (const mechanism of document.mechanisms) if (mechanism.kind === 'lift')
    for (const entry of mechanism.walls) if (entry.graphic === material.record)
      validateWallMaterialSampling(wall, material, entry.base >> 16);
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
    validateSourceSampling(document, record.wall!, material);
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
  if (!room.wallMaterials?.length) return { label: 'Reset painted room walls', before: source, after: source };
  const { wallMaterials: _previous, ...unchanged } = room;
  return { label: 'Reset painted room walls', before: source,
    after: Object.freeze({ ...source, rooms: Object.freeze(source.rooms.map(item => item.id === id ? Object.freeze(unchanged) : item)) }) };
}

/** Change only native material/sampling fields. Geometry, phases, strip span,
 * lighting, draw order and animation pointers retain the existing contract. */
export function paintCellWalls(document: LevelDocument, source: RoomSource, graphics: CellGraphics): CellGraphics {
  if (!source.rooms.some(room => room.wallMaterials?.length)) return graphics;
  const applied = new Set<string>(), streams = new Map(graphics.streams);
  for (const cell of graphics.topology.cells) {
    const room = source.rooms.find(room => room.id === cell.sourceId)!, stream = streams.get(cell.id)!;
    const layer = (records: typeof stream.lower, upper: boolean) => records.map(item => {
      const material = room.wallMaterials?.find(material => material.upper === upper && material.record === item.sourceOffset);
      if (!material) return item;
      const wall = readGraphicRecord(new BinaryReader(item.bytes, 'painted wall'), 0, 2048).wall;
      if (!wall || wall.texture === switchTexture || material.texture === switchTexture
        || document.switches.some(control => control.zone >= 0 && control.graphic - 10 === material.record)) throw new Error(`Room ${room.id} has an incompatible painted switch/control wall. Reset its material before building.`);
      applied.add(`${room.id}:${key(material)}`);
      validateSourceSampling(document, wall, material);
      return { ...item, bytes: encodeWallSurface({ ...wall, texture: material.texture, verticalMask: material.verticalMask,
        verticalShift: material.verticalShift, horizontalMask: material.horizontalMask }) };
    });
    streams.set(cell.id, { lower: layer(stream.lower, false), ...(stream.upper ? { upper: layer(stream.upper, true) } : {}) });
  }
  for (const room of source.rooms) for (const material of room.wallMaterials ?? []) if (!applied.has(`${room.id}:${key(material)}`))
    throw new Error(`Room ${room.id} no longer draws painted wall ${material.record}. Restore its outline or reset its wall materials before building.`);
  return { ...graphics, streams };
}
