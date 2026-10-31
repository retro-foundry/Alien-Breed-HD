import { BinaryReader } from './binary';
import type { CellGraphics } from './cellGraphics';
import type { Position } from './geometry';
import { readGraphicRecord } from './graphics';
import type { LevelDocument } from './level';
import { buildPointLighting, resolvePointLighting, type PointLightingBinding } from './pointLighting';
import type { GraphicsRecordInput } from './graphicsBuild';
import type { RoomSource } from './roomSource';

const on = (point: Position, a: Position, b: Position) => (point.x - a.x) * (b.z - a.z) === (point.z - a.z) * (b.x - a.x)
  && point.x >= Math.min(a.x, b.x) && point.x <= Math.max(a.x, b.x) && point.z >= Math.min(a.z, b.z) && point.z <= Math.max(a.z, b.z);
const same = (a: ReturnType<typeof resolvePointLighting>, b: ReturnType<typeof resolvePointLighting>) => a.lowerBrightness === b.lowerBrightness && a.upperBrightness === b.upperBrightness;

/** Lighting provenance for generated boundaries and material records. Source
 * wall identities define the deformation; graphics identities define partial
 * decorations and arc/beam endpoints. Never use nearest-point sampling.
 * Claims retain record ownership so the assembler can separate coincident
 * graphics points that need different lower/upper light words.
 */
export function planGeometryLighting(document: LevelDocument, source: RoomSource, graphics: CellGraphics,
  newRoomLighting: ReadonlyMap<string, PointLightingBinding> = new Map()) {
  for (const id of newRoomLighting.keys()) if (!source.rooms.some(room => room.id === id && room.originZone === null))
    throw new Error(`New-room lighting refers to missing or imported room ${id}.`);
  const claims: { id: number; binding: PointLightingBinding; recordOffset?: number }[] = [];
  const assigned = new Set<number>(); let activeRecord: number | undefined;
  const claim = (id: number, binding: PointLightingBinding, _owner: string) => {
    resolvePointLighting(document, binding);
    claims.push({ id, binding, recordOffset: activeRecord }); assigned.add(id);
  };
  const points = graphics.topology.points;
  for (const room of source.rooms) {
    const cells = graphics.topology.cells.filter(cell => cell.sourceId === room.id);
    if (!cells.length) continue;
    const references = new Set(cells.flatMap(cell => [...cell.vertices, ...graphics.pointReferences.get(cell.id)!]));
    if (room.originZone === null) {
      const selection = newRoomLighting.get(room.id);
      if (!selection) throw new Error(`Room ${room.id} needs its initial point-lighting selection before building.`);
      references.forEach(id => claim(id, selection, room.id)); continue;
    }
    const template = document.zones.find(zone => zone.id === room.recordZone)!;
    const records = [...template.lower, ...template.upper], sourcePoints = new Set(records.flatMap(record => record.points));
    const originalAt = (position: Position, candidates = sourcePoints): number => {
      let ids = [...candidates].filter(id => document.points[id]!.x === position.x && document.points[id]!.z === position.z);
      if (!ids.length && candidates === sourcePoints) ids = document.points.filter(point => point.x === position.x && point.z === position.z).map(point => point.id);
      if (!ids.length) throw new Error(`Room ${room.id} has no original lighting identity for its source boundary corner.`);
      const first = ids[0]!, value = resolvePointLighting(document, { kind: 'copy', source: first });
      if (ids.some(id => !same(value, resolvePointLighting(document, { kind: 'copy', source: id }))))
        throw new Error(`Room ${room.id} has distinct original lighting identities at one source corner. Preserve separate graphics point identities before building.`);
      return first;
    };
    // Geometric material records retain their original endpoint roles even
    // when a boundary wall is emitted as several ordered segments.
    for (const record of records.filter(record => !record.polygon && record.points.length > 0)) {
      activeRecord = record.offset;
      const emitted = cells.flatMap(cell => {
        const stream = graphics.streams.get(cell.id)!;
        return [...stream.lower, ...(stream.upper ?? [])].filter(item => item.sourceOffset === record.offset)
          .map(item => readGraphicRecord(new BinaryReader(item.bytes, 'generated lighting target'), 0, points.length));
      });
      if (!emitted.length) continue;
      if (!record.wall) {
        emitted.forEach(item => item.points.forEach((id, index) => claim(id, { kind: 'copy', source: record.points[index]! }, room.id))); continue;
      }
      const segments = [...new Map(emitted.map(item => [`${item.points[0]},${item.points[1]}`, item.points])).values()];
      const starts = new Set(segments.map(pair => pair[0]!)), ends = new Set(segments.map(pair => pair[1]!));
      const first = [...starts].filter(id => !ends.has(id)), last = [...ends].filter(id => !starts.has(id));
      if (first.length !== 1 || last.length !== 1) throw new Error(`Room ${room.id} has an ambiguous generated wall-lighting chain.`);
      const from = record.points[0]!, to = record.points[1]!, a = points[first[0]!]!, b = points[last[0]!]!;
      const lower = resolvePointLighting(document, { kind: 'copy', source: from }), upper = resolvePointLighting(document, { kind: 'copy', source: to });
      for (const id of new Set(segments.flat())) {
        if (id === first[0]) claim(id, { kind: 'copy', source: from }, room.id);
        else if (id === last[0]) claim(id, { kind: 'copy', source: to }, room.id);
        else if (same(lower, upper)) claim(id, { kind: 'copy', source: from }, room.id);
        else {
          if (!on(points[id]!, a, b)) throw new Error(`Room ${room.id} needs explicit lighting along its bent material wall.`);
          const dx = b.x - a.x, dz = b.z - a.z;
          claim(id, { kind: 'segment', from, to, numerator: (points[id]!.x - a.x) * dx + (points[id]!.z - a.z) * dz,
            denominator: dx * dx + dz * dz }, room.id);
        }
      }
    }
    const boundaryBinding = (id: number, candidates = sourcePoints): PointLightingBinding => {
      const point = points[id]!;
      if (candidates.has(id) && document.points[id]!.x === point.x && document.points[id]!.z === point.z) return { kind: 'copy', source: id };
      const edge = room.vertices.findIndex((a, index) => room.wallSources[index] !== null && on(point, a, room.vertices[(index + 1) % room.vertices.length]!));
      if (edge < 0) throw new Error(`Room ${room.id} point ${id} needs lighting provenance for its generated geometry.`);
      const lineId = room.wallSources[edge]!, line = document.lines[lineId]!;
      const indices = room.wallSources.flatMap((id, index) => id === lineId ? [index] : []);
      const first = indices.filter(index => room.wallSources[(index + room.wallSources.length - 1) % room.wallSources.length] !== lineId);
      const last = indices.filter(index => room.wallSources[(index + 1) % room.wallSources.length] !== lineId);
      if (first.length !== 1 || last.length !== 1) throw new Error(`Room ${room.id} has an ambiguous source lighting chain.`);
      const a = room.vertices[first[0]!]!, b = room.vertices[(last[0]! + 1) % room.vertices.length]!;
      if (!on(point, a, b)) throw new Error(`Room ${room.id} needs explicit lighting along its bent source wall.`);
      const dx = b.x - a.x, dz = b.z - a.z, denominator = dx * dx + dz * dz;
      const numerator = (point.x - a.x) * dx + (point.z - a.z) * dz;
      return { kind: 'segment', from: originalAt(line, candidates), to: originalAt({ x: line.x + line.dx, z: line.z + line.dz }, candidates), numerator, denominator };
    };
    for (const record of records.filter(record => record.polygon)) {
      activeRecord = record.offset;
      const candidates = new Set(record.points);
      for (const cell of cells) {
        const stream = graphics.streams.get(cell.id)!;
        for (const item of [...stream.lower, ...(stream.upper ?? [])].filter(item => item.sourceOffset === record.offset)) {
          const generated = readGraphicRecord(new BinaryReader(item.bytes, 'polygon lighting'), 0, points.length);
          generated.points.forEach(id => claim(id, boundaryBinding(id, candidates), room.id));
        }
      }
    }
    activeRecord = undefined;
    for (const id of references) {
      if (assigned.has(id)) continue;
      claim(id, boundaryBinding(id), room.id);
    }
  }
  return claims;
}

/** Separate graphics lighting identities automatically where rooms share
 * geometry but require different light words. Collision vertices and portal
 * clip points retain their shared identities; only draw-record references move.
 * renderer.c's clip-bound traversal uses the geometric clip lists, while draw
 * records use their own rotated/brightness point IDs.
 */
export function buildGeometryLighting(document: LevelDocument, source: RoomSource, graphics: CellGraphics,
  newRoomLighting: ReadonlyMap<string, PointLightingBinding> = new Map()) {
  for (const id of newRoomLighting.keys()) if (!source.rooms.some(room => room.id === id && room.originZone === null))
    throw new Error(`New-room lighting refers to missing or imported room ${id}.`);
  const points = graphics.topology.points.map(point => ({ ...point })), bindings = new Map<number, PointLightingBinding>();
  const copies = new Map<string, number>(), streams = new Map(graphics.streams), pointReferences = new Map(graphics.pointReferences);
  for (const room of source.rooms) {
    const selection = newRoomLighting.get(room.id), replacements = new Map<number | undefined, Map<number, number>>();
    const planned = planGeometryLighting(document, { ...source, rooms: [room] }, graphics,
      selection ? new Map([[room.id, selection]]) : new Map());
    for (const { id, binding, recordOffset } of planned) {
      const value = resolvePointLighting(document, binding), previous = bindings.get(id);
      let target = id;
      if (!previous || same(value, resolvePointLighting(document, previous))) {
        if (!previous) bindings.set(id, binding);
      } else {
        const key = `${id}:${value.lowerBrightness}:${value.upperBrightness}`;
        let copy = copies.get(key);
        if (copy === undefined) {
          if (points.length >= 2048) throw new Error(`Room ${room.id} needs separate lighting identities beyond the engine limit of 2,048 points.`);
          copy = points.length; points.push({ ...points[id]! }); bindings.set(copy, binding); copies.set(key, copy);
        }
        target = copy;
      }
      const mapping = replacements.get(recordOffset) ?? new Map<number, number>();
      mapping.set(id, target); replacements.set(recordOffset, mapping);
    }
    if (!replacements.size) continue;
    const rewrite = (item: GraphicsRecordInput): GraphicsRecordInput => {
      const record = readGraphicRecord(new BinaryReader(item.bytes, 'graphics lighting identities'), 0, points.length);
      const mapping = replacements.get(item.sourceOffset) ?? replacements.get(undefined);
      if (!mapping || (!record.points.some(id => mapping.get(id) !== undefined && mapping.get(id) !== id)
        && !(record.polygon && mapping.has(record.polygon.closingPoint) && mapping.get(record.polygon.closingPoint) !== record.polygon.closingPoint))) return item;
      const bytes = item.bytes.slice(), view = new DataView(bytes.buffer);
      record.points.forEach((id, index) => view.setInt16((record.polygon ? 6 : 2) + index * 2, mapping.get(id) ?? id));
      if (record.polygon) view.setInt16(6 + record.points.length * 2, mapping.get(record.polygon.closingPoint) ?? record.polygon.closingPoint);
      return { ...item, bytes };
    };
    for (const cell of graphics.topology.cells.filter(cell => cell.sourceId === room.id)) {
      const stream = streams.get(cell.id)!, lower = stream.lower.map(rewrite), upper = stream.upper?.map(rewrite);
      streams.set(cell.id, { lower, ...(upper === undefined ? {} : { upper }) });
      pointReferences.set(cell.id, [...new Set([...lower, ...(upper ?? [])].flatMap(item =>
        readGraphicRecord(new BinaryReader(item.bytes, 'lit graphics'), 0, points.length).points))]);
    }
  }
  return { bindings, points: buildPointLighting(document, points, bindings),
    graphics: { topology: { ...graphics.topology, points }, streams, pointReferences } satisfies CellGraphics };
}
