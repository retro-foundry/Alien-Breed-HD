import { BinaryReader } from './binary';
import { importedAnimationTables, type AnimationTables } from './animationBuild';
import type { CellGraphics } from './cellGraphics';
import type { GraphicsRecordInput } from './graphicsBuild';
import type { LevelDocument } from './level';
import { readGraphicRecord } from './graphics';
import { convexParts } from './geometry';
import { encodePolygonSurface, polygonForCell } from './surfaces';
import { assertWaterSurface, waterGroupSettings, waterStartHeight } from './sourceWater';

/** LEVELSAVE's lower-water draw order: objects below, water, objects above.
 * Each collision cell retains a complete copy of its logical room's graphics;
 * every polygon receives an explicit native animation target below. */
export function authorWaterGraphics(document: LevelDocument, input: CellGraphics): CellGraphics {
  if (!document.roomSource.waterGroups?.length && !document.roomSource.rooms.some(room => room.waterSurface !== undefined)) return input;
  const original = importedAnimationTables(document).tables.water, streams = new Map(input.streams), pointReferences = new Map(input.pointReferences);
  for (const cell of input.topology.cells) {
    const room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!, stream = streams.get(cell.id)!;
    let lower = stream.lower;
    if (room.waterSurface !== undefined) {
      if (room.waterSurface) assertWaterSurface(document, room, room.waterSurface);
      const tail: GraphicsRecordInput[] = room.waterSurface === false ? [{ bytes: new Uint8Array([0, 4, 0, 2]) }] : (() => {
        const source = room.waterSurface;
        if (source.material === null) {
          const polygons = stream.lower.filter(item => new BinaryReader(item.bytes, 'existing water').i16(0) === 7).map(item => {
            const bytes = item.bytes.slice(); new DataView(bytes.buffer).setInt16(2, Math.floor(waterStartHeight(document, room) * 4)); return { ...item, bytes };
          });
          if (!polygons.length) throw new Error(`Room ${room.id} has no existing water polygons. Choose a pattern for whole-room water instead.`);
          return [{ bytes: new Uint8Array([0, 4, 0, 0]) }, ...polygons, { bytes: new Uint8Array([0, 4, 0, 1]) }];
        }
        const material = document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === source.material);
        if (!material?.polygon || ![1, 2, 7].includes(material.type)) throw new Error(`Water in ${room.id} has a missing material reference. Choose its starting pattern again.`);
        const polygons = convexParts(room.vertices).map(vertices => {
          const points = vertices.map(point => input.topology.points.findIndex(target => target.x === point.x && target.z === point.z));
          if (points.some(id => id < 0)) throw new Error('Water has lost an authored outline corner. Rebuild the room geometry.');
          return { bytes: encodePolygonSurface(polygonForCell({ ...material.polygon!, type: 7, scale: -1, brightness: 0 }, points, Math.floor(waterStartHeight(document, room) * 4) / 4)) };
        });
        return [{ bytes: new Uint8Array([0, 4, 0, 0]) }, ...polygons, { bytes: new Uint8Array([0, 4, 0, 1]) }];
      })();
      const result: GraphicsRecordInput[] = []; let inserted = false;
      for (const item of lower) {
        const reader = new BinaryReader(item.bytes, 'water draw order'), type = reader.i16(0);
        if (type === 4 && ![0, 1, 2].includes(reader.i16(2))) throw new Error('This room has an unsupported object clipping mode. Restore a standard lower-room draw stream before replacing water.');
        if (type === 7 || type === 4) { if (!inserted) { result.push(...tail); inserted = true; } continue; }
        result.push(item);
      }
      if (!inserted) result.push(...tail);
      lower = result;
    } else if (document.roomSource.waterGroups?.length) {
      lower = lower.map(item => {
        if (item.sourceOffset === undefined || new BinaryReader(item.bytes, 'water group phase').i16(0) !== 7) return item;
        const groups = document.roomSource.waterGroups!.filter(group => original[group.id]?.entries.some(entry => entry.graphic === item.sourceOffset));
        if (!groups.length) return item;
        if (new Set(groups.map(group => group.current)).size > 1) throw new Error('Several water groups write different initial heights to the same polygon. Assign it to one group.');
        const bytes = item.bytes.slice(); new DataView(bytes.buffer).setInt16(2, Math.floor(groups[0]!.current * 4)); return { ...item, bytes };
      });
    }
    streams.set(cell.id, { ...stream, lower });
    pointReferences.set(cell.id, [...new Set([...lower, ...(stream.upper ?? [])].flatMap(item => readGraphicRecord(new BinaryReader(item.bytes, 'water points'), 0, input.topology.points.length).points))]);
  }
  return { ...input, streams, pointReferences };
}

/** WaterList follows the lift 999 terminator even in a level with no lifts.
 * Empty native slots are structural records, not substitute controllers. */
export function appendSourceWater(document: LevelDocument, graphics: CellGraphics, tables: AnimationTables): AnimationTables {
  const authored = document.roomSource.waterGroups, assigned = document.roomSource.rooms.some(room => room.waterSurface && room.waterSurface.group !== null);
  if (!authored?.length && !assigned) return tables;
  const water = (tables.water.length ? tables.water : Array.from({ length: 21 }, () => ({ header: new Uint8Array(14), entries: [] }))).map(group => ({ header: group.header.slice(), entries: [...group.entries] }));
  for (const group of authored ?? []) {
    const header = new Uint8Array(14), view = new DataView(header.buffer);
    view.setInt32(0, group.top * 256); view.setInt32(4, group.bottom * 256); view.setInt32(8, group.current * 256); view.setInt16(12, group.velocity);
    water[group.id] = { header, entries: group.enabled ? water[group.id]!.entries : [] };
  }
  for (const cell of graphics.topology.cells) {
    const room = document.roomSource.rooms.find(room => room.id === cell.sourceId)!;
    if (!room.waterSurface || room.waterSurface.group === null || !waterGroupSettings(document, room.waterSurface.group).enabled) continue;
    const group = room.waterSurface.group;
    graphics.streams.get(cell.id)!.lower.forEach((item, index) => {
      if (new BinaryReader(item.bytes, 'animated water target').i16(0) === 7) water[group]!.entries.push({ zone: cell.id, graphic: 0, target: { room: cell.id, upper: false, index } });
    });
  }
  for (const [index, group] of water.entries()) if (group.entries.length > 256) throw new Error(`Water group ${index + 1} exceeds the native limit of 256 animated polygon attachments. Reduce the room decomposition or divide water between groups.`);
  return { ...tables, lifts: tables.lifts ?? [], water };
}
