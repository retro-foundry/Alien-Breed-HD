import { BinaryReader } from './binary';
import { normaliseRoomOutline } from './geometry';
import { readGraphicRecord, type Graphic } from './graphics';
import type { GraphicsRecordInput, GraphicsStreamInput } from './graphicsBuild';
import type { LevelDocument } from './level';
import type { RoomSource } from './roomSource';
import type { RoomTopology } from './roomTopology';
import { buildCellPolygons } from './surfaceGeometry';
import { buildCellWallGeometry, type NewRoomWallMaterials, type ClosingWallMaterials } from './wallGeometry';
import { buildDrawTopology } from './drawTopology';

export interface NewRoomGraphics {
  walls: NewRoomWallMaterials;
  /** Explicit insertion indices in the selected template streams. A wall
   * batch must retain its intended position relative to water/object draws. */
  wallInsertion: { lower: number; upper?: number };
}
export interface ClosingRoomGraphics extends NewRoomGraphics { lightingPoint: number }
export interface CellGraphics {
  topology: RoomTopology;
  streams: Map<number, GraphicsStreamInput>;
  pointReferences: Map<number, number[]>;
}

/** Assemble complete ordered streams from the source material/control records.
 * Renderer.c polyloop consumes object/water/clip/backdrop records in sequence;
 * collecting walls and floors into separate batches changes that contract.
 * Returned graphics point references feed the visibility/rotation compiler.
 */
export function buildCellGraphics(document: LevelDocument, source: RoomSource, topology: RoomTopology,
  newRooms: ReadonlyMap<string, NewRoomGraphics> = new Map(), closures: ReadonlyMap<string, ClosingRoomGraphics> = new Map()): CellGraphics {
  for (const id of newRooms.keys()) if (!source.rooms.some(room => room.id === id && room.originZone === null))
    throw new Error(`New-room graphics settings refer to missing or imported room ${id}.`);
  for (const id of closures.keys()) if (!source.rooms.some(room => room.id === id && room.originZone !== null))
    throw new Error(`Closure graphics settings refer to missing or newly drawn room ${id}.`);
  const draw = buildDrawTopology(source, topology);
  const closing = new Map<string, ClosingWallMaterials>([...closures].map(([id, room]) => [id, { walls: room.walls, lightingPoint: room.lightingPoint }]));
  const generated = buildCellWallGeometry(document, source, draw.topology, new Map([...newRooms].map(([id, room]) => [id, room.walls])), closing);
  const compiledTopology = { ...topology, points: generated.topology.points };
  const polygons = buildCellPolygons(document, compiledTopology);
  const wholePolygons = buildCellPolygons(document, { ...compiledTopology, cells: draw.topology.cells.filter(cell => cell.vertices.length <= 10) });
  const points = compiledTopology.points;
  const coordinates = new Map<string, number>();
  points.forEach((point, id) => { const key = `${point.x},${point.z}`; if (!coordinates.has(key)) coordinates.set(key, id); });
  const pointId = (x: number, z: number): number => {
    if (!Number.isSafeInteger(x) || !Number.isSafeInteger(z) || x < -32768 || x > 32767 || z < -32768 || z > 32767)
      throw new Error('A translated graphics point is outside the whole-coordinate level bounds.');
    const key = `${x},${z}`, existing = coordinates.get(key); if (existing !== undefined) return existing;
    if (points.length >= 2048) throw new Error('Graphics geometry exceeds the engine limit of 2,048 points.');
    const id = points.length; points.push({ x, z }); coordinates.set(key, id); return id;
  };
  const reader = new BinaryReader(document.files.graphics, 'source graphics'), table = reader.i32(12);
  const streams = new Map<number, GraphicsStreamInput>(), pointReferences = new Map<number, number[]>();
  for (const cell of topology.cells) {
    const room = source.rooms.find(room => room.id === cell.sourceId)!, template = document.zones.find(zone => zone.id === cell.recordZone)!;
    const originalCorners = normaliseRoomOutline(template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })));
    const dx = room.vertices[0]!.x - originalCorners[0]!.x, dz = room.vertices[0]!.z - originalCorners[0]!.z;
    const translation = room.originZone !== null && room.vertices.length === originalCorners.length
      && room.vertices.every((point, index) => point.x - originalCorners[index]!.x === dx && point.z - originalCorners[index]!.z === dz);
    const owner = draw.owners.get(cell.id)!, members = topology.cells.filter(target => target.sourceId === cell.sourceId);
    const polygonLayers = wholePolygons.get(owner) ?? { lower: members.flatMap(member => polygons.get(member.id)!.lower), upper: members.flatMap(member => polygons.get(member.id)!.upper) };
    const wallLayers = generated.walls.get(owner)!, selection = newRooms.get(room.id) ?? closures.get(room.id);
    const layer = (records: readonly Graphic[], upper: boolean): GraphicsRecordInput[] => {
      const wallRecords = upper ? wallLayers.upper : wallLayers.lower, polygonRecords = upper ? polygonLayers.upper : polygonLayers.lower;
      const insertion = upper ? selection?.wallInsertion.upper : selection?.wallInsertion.lower;
      const additions = wallRecords.filter(item => item.sourceOffset === undefined);
      if ((room.originZone === null || additions.length > 0) && ((insertion === undefined && wallRecords.length > 0)
        || (insertion !== undefined && (!Number.isInteger(insertion) || insertion < 0 || insertion > records.length))))
        throw new Error(`Room ${room.id} needs a valid ${upper ? 'upper' : 'lower'} wall insertion position in its material template.`);
      const result: GraphicsRecordInput[] = [];
      for (let index = 0; index <= records.length; index++) {
        if (index === insertion) result.push(...additions);
        const record = records[index]; if (!record) continue;
        if (record.wall) {
          if (room.originZone !== null) result.push(...wallRecords.filter(item => item.sourceOffset === record.offset));
        } else if (record.polygon) result.push(...polygonRecords.filter(item => item.sourceOffset === record.offset));
        else {
          const bytes = reader.bytes.slice(record.offset, record.offset + record.size);
          if (record.points.length) {
            // Arc center/edge and LightDraw point pairs own geometry, unlike
            // opcode 4's object clip mode. Do not copy stale coordinates.
            if (!translation) throw new Error(`Room ${room.id} graphics record at byte ${record.offset} needs explicit geometry mapping after reshaping.`);
            if (dx !== 0 || dz !== 0 || record.points.some(id => points[id]?.x !== document.points[id]!.x || points[id]?.z !== document.points[id]!.z)) {
              const view = new DataView(bytes.buffer);
              record.points.forEach((id, index) => {
                const point = document.points[id]!; view.setInt16(2 + index * 2, pointId(point.x + dx, point.z + dz));
              });
            }
          }
          result.push({ sourceOffset: record.offset, bytes });
        }
      }
      return result;
    };
    const lower = layer(template.lower, false);
    const hasUpper = reader.i32(table + template.id * 8 + 4) !== 0 || wallLayers.upper.length > 0;
    const upper = hasUpper ? layer(template.upper, true) : undefined;
    streams.set(cell.id, { lower, ...(upper === undefined ? {} : { upper }) });
    pointReferences.set(cell.id, [...new Set([...lower, ...(upper ?? [])].flatMap(item =>
      readGraphicRecord(new BinaryReader(item.bytes, 'compiled cell graphics'), 0, points.length).points))]);
  }
  return { topology: compiledTopology, streams, pointReferences };
}
