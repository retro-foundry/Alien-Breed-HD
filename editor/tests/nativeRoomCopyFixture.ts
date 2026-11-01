import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { applyCommand, editRoomAppearance } from '../src/model/commands';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { buildDataTables, importedDataTables } from '../src/model/dataBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { encodePolygonSurface } from '../src/model/surfaces';

/** Compiler fixture only: the complete room clipboard/controller authoring UI
 * remains separate work. Preserve wall provenance with a fresh source ID. */
export function nativeRoomCopyFixture(zone = 0, base = openLevel('Native material copy', originalFiles(1))) {
  let document = base;
  const original = document.roomSource.rooms.find(room => room.originZone === zone)!;
  const dx = 20000 - original.vertices[0]!.x, dz = 20000 - original.vertices[0]!.z;
  const copy = { ...original, id: `room-${document.roomSource.nextId}`, originZone: null,
    vertices: original.vertices.map(point => ({ x: point.x + dx, z: point.z + dz })) };
  document = { ...document, roomSource: { rooms: [...document.roomSource.rooms, copy], nextId: document.roomSource.nextId + 1 } };
  // A one-sided copied opening needs an explicit closure style. Choose real
  // template records, just as the appearance inspector does.
  const choices = roomAppearanceChoices(document, copy);
  document = applyCommand(document, editRoomAppearance(document, { room: copy.id, lowerWall: choices.lower[0]!.offset,
    upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! }));
  return { document, original, copy, dx, dz };
}

export function nativePartialRoomCopyFixture() {
  let source = openLevel('Partial native material copy', originalFiles(1));
  const floor = source.zones[0]!.lower.find(record => record.type === 1)!;
  const centre = floor.points.reduce((sum, id) => ({ x: sum.x + source.points[id]!.x / floor.points.length,
    z: sum.z + source.points[id]!.z / floor.points.length }), { x: 0, z: 0 });
  const interior = source.points.length, tables = importedDataTables(source);
  source = openLevel(source.name, buildDataTables(source, { ...tables, points: [...tables.points,
    { x: Math.round(centre.x), z: Math.round(centre.z), lowerBrightness: -11, upperBrightness: 19 }] }));
  const points = [floor.points[0]!, floor.points[1]!, interior], polygon = { ...floor.polygon!, points, closingPoint: points[0]! };
  source = openLevel(source.name, { ...source.files, graphics: buildGraphicsStreams(source, undefined, undefined,
    { replacements: new Map([[floor.offset, encodePolygonSurface(polygon)]]) }) });
  return { ...nativeRoomCopyFixture(0, source), source, polygon };
}
