import { roomPlaneRecords } from './roomPlanes';
import { BinaryReader } from './binary';
import type { CellGraphics } from './cellGraphics';
import { readGraphicRecord } from './graphics';
import type { GraphicsRecordInput } from './graphicsBuild';
import type { LevelDocument } from './level';
import { roomSettings } from './roomHeights';
import { storeyMaterials } from './roomStoreys';
import { encodePolygonSurface, encodeWallSurface, polygonForCell, polygonPlane } from './surfaces';
import { wallForSpan } from './surfaceGeometry';
import { portalWallSpans } from './portalWalls';
import { sourceBoundaryIndex } from './boundaryMaterials';

/** Current native opcodes, using AMOS LEVELSAVE's upper wall/floor/roof
 * layout. The new storey is source-owned; original lower graphics stay intact. */
export function authorRoomLayers(document: LevelDocument, graphics: CellGraphics): CellGraphics {
  if (!document.roomSource.rooms.some(room => room.upperStorey !== undefined || room.planeSources?.length)) return graphics;
  const streams = new Map(graphics.streams), pointReferences = new Map(graphics.pointReferences);
  for (const cell of graphics.topology.cells) {
    const room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!;
    let stream = streams.get(cell.id)!;
    for (const upper of [false, true]) {
      const added = (room.planeSources ?? []).filter(item => item.upper === upper);
      if (!added.length) continue;
      const settings = roomSettings(document, room);
      const extra = added.map(item => {
        const record = roomPlaneRecords(document, room, upper).find(record => record.offset === item.record)!;
        const height = record.type === 1 ? upper ? settings.upperFloor : settings.floor : upper ? settings.upperCeiling : settings.ceiling;
        return { sourceOffset: record.offset, bytes: encodePolygonSurface(polygonForCell(record.polygon!, cell.vertices, height)), lightingPoint: room.upperStorey ? room.upperStorey.lightingPoint : undefined };
      });
      stream = upper ? { ...stream, upper: [...(stream.upper ?? []), ...extra] } : { ...stream, lower: [...stream.lower, ...extra] };
      streams.set(cell.id, stream);
      pointReferences.set(cell.id, [...new Set([...stream.lower, ...(stream.upper ?? [])].flatMap(item => readGraphicRecord(new BinaryReader(item.bytes, 'added plane points'), 0, graphics.topology.points.length).points))]);
    }
    if (room.upperStorey === undefined) continue;
    if (room.upperStorey === false) { streams.set(cell.id, { lower: stream.lower }); continue; }
    const materials = storeyMaterials(document, room), settings = roomSettings(document, room);
    const planeRecords = roomPlaneRecords(document, room, true);
    const floorRecord = planeRecords.find(record => record.type === 1), ceilingRecord = planeRecords.find(record => record.type === 2);
    const selection = room.upperStorey;
    const reference = materials.walls.find(record => record.offset === selection.wall)?.wall;
    if (!reference || !floorRecord?.polygon || !ceilingRecord?.polygon) throw new Error('The upper storey has lost its wall, floor or ceiling reference. Restore its source materials.');
    const points = graphics.topology.points, a = document.points[reference.points[0]]!, b = document.points[reference.points[1]]!;
    const upper: GraphicsRecordInput[] = cell.vertices.flatMap((id, index) => {
      const next = cell.vertices[(index + 1) % cell.vertices.length]!, destination = graphics.topology.cells.find(target => target.id === cell.neighbours[index]);
      if (destination?.sourceId === cell.sourceId) return [];
      const material = wallForSpan({ ...reference, topFixed: settings.upperCeiling * 256, bottomFixed: settings.upperFloor * 256 },
        Math.hypot(b.x - a.x, b.z - a.z), [id, next], 0, Math.hypot(points[next]!.x - points[id]!.x, points[next]!.z - points[id]!.z));
      const spans = destination ? portalWallSpans(document, material, roomSettings(document, destination)) : [material];
      return spans.map(span => ({ bytes: encodeWallSurface(span), boundaryIndex: sourceBoundaryIndex(room, points[id]!, points[next]!), lightingPoint: selection.lightingPoint }));
    });
    for (const [record, height] of [[floorRecord, settings.upperFloor], [ceilingRecord, settings.upperCeiling]] as const)
      upper.push({ sourceOffset: record.offset, lightingPoint: selection.lightingPoint, bytes: encodePolygonSurface(polygonForCell(record.polygon!, cell.vertices, height)) });
    // LEVELSAVE's dry upper-room tail: ObjDraw (opcode 4), full-room clip 2.
    upper.push({ bytes: new Uint8Array([0, 4, 0, 2]) });
    streams.set(cell.id, { ...stream, upper });
    pointReferences.set(cell.id, [...new Set([...stream.lower, ...upper].flatMap(item => readGraphicRecord(new BinaryReader(item.bytes, 'authored storey'), 0, points.length).points))]);
  }
  return { ...graphics, streams, pointReferences };
}

/** RGPUT NONE (SKY): omit ceiling polygons and emit backdrop opcode 12.
 * The native renderer builds sky spans from the remaining normal floor. */
export function authorSkyCeilings(document: LevelDocument, graphics: CellGraphics): CellGraphics {
  if (!document.roomSource.rooms.some(room => room.sky?.some(Boolean))) return graphics;
  const streams = new Map(graphics.streams), pointReferences = new Map(graphics.pointReferences);
  for (const cell of graphics.topology.cells) {
    const room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!;
    if (!room.sky?.some(Boolean)) continue;
    const stream = streams.get(cell.id)!;
    const layer = (records: readonly GraphicsRecordInput[], upper: boolean) => {
      if (!room.sky?.[upper ? 1 : 0]) return records;
      if (!records.some(item => readGraphicRecord(new BinaryReader(item.bytes, 'sky floor'), 0, graphics.topology.points.length).type === 1)) throw new Error('A sky opening needs a normal floor pattern. Set the floor material for this storey to Normal pattern first.');
      return [...records.filter(item => {
        const record = readGraphicRecord(new BinaryReader(item.bytes, 'sky ceiling'), 0, graphics.topology.points.length);
        return polygonPlane(record.type) !== 'ceiling' && record.type !== 12;
      }), { bytes: new Uint8Array([0, 12]) }];
    };
    const lower = layer(stream.lower, false), upper = stream.upper === undefined ? undefined : layer(stream.upper, true);
    streams.set(cell.id, { lower, ...(upper ? { upper } : {}) });
    pointReferences.set(cell.id, [...new Set([...lower, ...(upper ?? [])].flatMap(item => readGraphicRecord(new BinaryReader(item.bytes, 'sky points'), 0, graphics.topology.points.length).points))]);
  }
  return { ...graphics, streams, pointReferences };
}
