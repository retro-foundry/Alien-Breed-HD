import { BinaryReader, checkedInteger } from './binary';
import { readGraphicRecord } from './graphics';
import { encodeWallSurface } from './surfaces';
import type { CellGraphics } from './cellGraphics';
import type { LevelDocument } from './level';
import type { GraphicsRecordInput } from './graphicsBuild';
import type { SourceRoom } from './roomSource';
import type { SelectGraphicsTarget } from './generatedGraphics';
import type { AnimationTables, MovingRoomAnimation } from './animationBuild';
import { validateWallMaterialSampling } from './wallMaterials';
import type { WallSettings } from './wallSettings';
import { repeatWallSurface } from './wallRepeat';

const identity = (item: WallSettings) => `${item.upper}:${'record' in item ? 'record:' + item.record : 'boundary:' + item.boundary}`;
function settingsFor(room: SourceRoom, item: GraphicsRecordInput, upper: boolean) {
  return room.wallSettings?.find(settings => settings.upper === upper && ('record' in settings ? settings.record === item.sourceOffset
    : item.sourceOffset === undefined && item.boundaryIndex !== undefined && room.boundaryMaterials?.edges[item.boundaryIndex] === settings.boundary));
}

/** Native WallRoutine adds the signed wallbrightoff to endpoint light words.
 * FROMTILE is a 16-column tile offset; totalyoff is a signed row offset.
 * Apply after geometry/material assembly so every generated piece retains
 * its own strip span, heights, points, transparency and provenance. */
export function buildWallSettings(document: LevelDocument, input: CellGraphics): CellGraphics {
  if (!document.roomSource.rooms.some(room => room.wallSettings?.length)) return input;
  const applied = new Set<string>(), streams = new Map(input.streams);
  for (const cell of input.topology.cells) {
    const room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!, stream = streams.get(cell.id)!;
    const layer = (records: typeof stream.lower, upper: boolean) => records.map(item => {
      const settings = settingsFor(room, item, upper);
      if (!settings) return item;
      const wall = readGraphicRecord(new BinaryReader(item.bytes, 'wall alignment and brightness'), 0, input.topology.points.length).wall;
      if (!wall || wall.texture === 11) throw new Error('Wall alignment settings cannot replace switch control graphics or non-wall commands. Reset those settings.');
      const { upper: _upper, ...fields } = settings;
      const repeated = settings.repeat ? repeatWallSurface(document, room, settings, wall, input.topology.points) : wall;
      const updated = { ...repeated,
        ...(fields.brightness !== undefined ? { brightness: fields.brightness } : {}),
        ...(fields.horizontalOffset !== undefined ? { horizontalOffset: fields.horizontalOffset } : {}),
        ...(fields.verticalOffset !== undefined ? { verticalOffset: fields.verticalOffset } : {}) };
      // Brightness/row-only edits must not reject a retained legacy sheet
      // layout. Newly chosen column offsets must fit the actual painted sheet.
      if (fields.horizontalOffset !== undefined || settings.repeat) validateWallMaterialSampling(updated, updated);
      applied.add(`${room.id}:${identity(settings)}`);
      return { ...item, bytes: encodeWallSurface(updated) };
    });
    streams.set(cell.id, { ...stream, lower: layer(stream.lower, false), ...(stream.upper ? { upper: layer(stream.upper, true) } : {}) });
  }
  for (const room of document.roomSource.rooms) for (const item of room.wallSettings ?? []) if (!applied.has(`${room.id}:${identity(item)}`))
    throw new Error(`Room ${room.id} no longer draws a wall with saved alignment or brightness. Restore its outline or reset room wall settings in its inspector.`);
  return { ...input, streams };
}

/** DoorRoutine and LiftRoutine own the moving wall's phase. Re-encode their
 * actual base words, rather than relying on an initial graphics edit that the
 * next native update would overwrite. Brightness never changes these tables. */
export function syncWallSettingsAnimations(document: LevelDocument, graphics: CellGraphics, tables: AnimationTables, selectTarget: SelectGraphicsTarget): AnimationTables {
  if (!document.roomSource.rooms.some(room => room.wallSettings?.some(item => item.horizontalOffset !== undefined || item.verticalOffset !== undefined || item.repeat))) return tables;
  const update = (records: readonly MovingRoomAnimation[] | null, kind: 'door' | 'lift') => records?.map(record => ({ ...record, walls: record.walls.map(wall => {
    const targets = graphics.topology.cells.flatMap(cell => {
      const stream = graphics.streams.get(cell.id)!;
      return ([[false, stream.lower], [true, stream.upper ?? []]] as const).flatMap(([upper, items]) => items.flatMap((item, index) => item.sourceOffset === wall.graphic ? [{ room: cell.id, upper, index }] : []));
    });
    const target = wall.target ?? selectTarget(wall.graphic, { kind: 'moving-wall', zone: record.zone, line: wall.line }, targets);
    const cell = graphics.topology.cells.find(cell => cell.id === target.room)!, stream = graphics.streams.get(target.room)!;
    const item = (target.upper ? stream.upper : stream.lower)![target.index]!, room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!;
    const settings = settingsFor(room, item, target.upper);
    if (!settings || settings.horizontalOffset === undefined && settings.verticalOffset === undefined && !settings.repeat) return wall;
    const current = new BinaryReader(record.header, 'moving wall initial phase').i16(4) / 4;
    const tile = settings.horizontalOffset ?? (kind === 'door' ? wall.base >>> 12 : wall.base >> 16);
    const row = settings.verticalOffset === undefined ? wall.base & (kind === 'door' ? 4095 : 65535)
      : kind === 'door' ? (settings.verticalOffset + Math.floor(current)) & 255 : settings.verticalOffset & 65535;
    // DoorRoutine interprets the legacy base as uint32_t before extracting
    // its tile. Preserve that exact bit pattern when the signed file word is
    // negative and only the row is being edited.
    const base = checkedInteger(tile * (kind === 'door' ? 4096 : 65536) + row, -2147483648, 4294967295, 'Moving wall texture alignment') | 0;
    const surface = readGraphicRecord(new BinaryReader(item.bytes, 'moving wall alignment'), 0, graphics.topology.points.length).wall;
    if (!surface) throw new Error('A moving wall alignment lost its straight wall target.');
    if (settings.horizontalOffset !== undefined || settings.repeat) validateWallMaterialSampling(surface, surface, tile);
    return { ...wall, base };
  }) })) ?? null;
  return { ...tables, doors: update(tables.doors, 'door'), lifts: update(tables.lifts, 'lift') };
}
