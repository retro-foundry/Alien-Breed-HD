import { normaliseRoomOutline, type Position } from './geometry';
import type { Graphic, LevelDocument } from './level';
import type { GraphicsRecordInput } from './graphicsBuild';
import type { RoomSource, SourceRoom } from './roomSource';
import type { RoomTopology } from './roomTopology';
import { encodeWallSurface, type WallSurface } from './surfaces';
import { wallForSpan } from './surfaceGeometry';
import { changedPortalEdges, hasUpperLayer, newPortalTarget, portalWallSpans, unassignedBoundaryEdges } from './portalWalls';

export interface CellWalls { lower: GraphicsRecordInput[]; upper: GraphicsRecordInput[] }
export interface NewRoomWallMaterials { lower: readonly WallSurface[]; upper?: readonly WallSurface[] }
export interface ClosingWallMaterials { walls: NewRoomWallMaterials; lightingPoint: number }
export interface WallGeometry { topology: RoomTopology; walls: Map<number, CellWalls> }
const distance = (a: Position, b: Position) => Math.hypot(b.x - a.x, b.z - a.z);
const on = (p: Position, a: Position, b: Position) => (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
  && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);
const key = (points: readonly Position[]) => JSON.stringify(normaliseRoomOutline(points.map(({ x, z }) => ({ x, z }))));
function wallChain(room: SourceRoom, lineId: number) {
  let total = 0;
  const first = room.wallSources.findIndex((id, index) => id === lineId && room.wallSources[(index + room.wallSources.length - 1) % room.wallSources.length] !== lineId);
  const pieces = room.vertices.flatMap((_, step) => {
    const index = (Math.max(first, 0) + step) % room.vertices.length, start = room.vertices[index]!;
    if (room.wallSources[index] !== lineId) return [];
    const end = room.vertices[(index + 1) % room.vertices.length]!, length = distance(start, end), offset = total;
    total += length; return [{ start, end, length, offset }];
  });
  return { pieces, total };
}

/** Bind source boundary materials to deformed/split cell edges. Preserve
 * strip density, phase, sampling and heights (WallRoutine3.ChipMem.s), and
 * retain source record identities for subsequent mechanism attachment mapping.
 * Partial boundary decorations add their own point identities to the returned
 * topology; callers must use that point table in all following compiler stages.
 * New-room perimeter materials are explicit selections, not guessed textures.
 */
export function buildCellWallGeometry(document: LevelDocument, source: RoomSource, topology: RoomTopology,
  newMaterials: ReadonlyMap<string, NewRoomWallMaterials> = new Map(), closingMaterials: ReadonlyMap<string, ClosingWallMaterials> = new Map()): WallGeometry {
  const points = topology.points.map(point => ({ ...point })), coordinates = new Map<string, number>();
  points.forEach((point, id) => { const name = `${point.x},${point.z}`; if (!coordinates.has(name)) coordinates.set(name, id); });
  const pointId = (point: Position): number => {
    const rounded = { x: Math.round(point.x), z: Math.round(point.z) };
    if (Math.abs(rounded.x - point.x) > 1e-7 || Math.abs(rounded.z - point.z) > 1e-7)
      throw new Error('A wall decoration falls between whole world coordinates after reshaping. Choose a grid-aligned room shape or reassign this decoration before building.');
    const name = `${rounded.x},${rounded.z}`, existing = coordinates.get(name);
    if (existing !== undefined) return existing;
    if (points.length >= 2048) throw new Error('Wall geometry exceeds the engine limit of 2,048 points.');
    const id = points.length; points.push(rounded); coordinates.set(name, id); return id;
  };
  const walls = new Map<number, CellWalls>();
  for (const cell of topology.cells) {
    if (walls.has(cell.id)) throw new Error('Wall geometry contains duplicate runtime room identities.');
    const room = source.rooms.find(room => room.id === cell.sourceId), template = document.zones.find(zone => zone.id === cell.recordZone);
    if (!room || !template) throw new Error(`Room ${cell.sourceId} has a missing wall source or settings template.`);
    const unassigned = unassignedBoundaryEdges(topology, cell);
    const changed = [...changedPortalEdges(document, topology, cell), ...unassigned], closing = closingMaterials.get(room.id);
    if (unassigned.length && !closing?.walls.lower.length) throw new Error(`Room ${room.id.replace(/^room-/, '')} has new boundary walls needing a wall style. Select the room and apply its wall repair appearance before building.`);
    if (changed.length && !closing?.walls.lower.length) throw new Error(`Room ${room.id.replace(/^room-/, '')} has a changed doorway needing a wall style. Select the room and apply its wall repair appearance before building.`);
    if (changed.length && hasUpperLayer(document, template) && !closing?.walls.upper?.length)
      throw new Error(`Room ${room.id.replace(/^room-/, '')} needs an upper wall style for its changed doorway. Apply its upper wall repair appearance before building.`);
    const outline = cell.vertices.map(id => {
      const point = points[id]; if (!point) throw new Error(`Room ${cell.sourceId} has a missing wall corner.`); return point;
    });
    const originalOutline = template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z }));
    const unchanged = key(outline) === key(originalOutline);
    const originalCorners = normaliseRoomOutline(originalOutline), dx = room.vertices[0]!.x - originalCorners[0]!.x, dz = room.vertices[0]!.z - originalCorners[0]!.z;
    const translatedWholeRoom = key(outline) === key(room.vertices) && room.vertices.length === originalCorners.length
      && room.vertices.every((point, index) => point.x - originalCorners[index]!.x === dx && point.z - originalCorners[index]!.z === dz);
    const retained = (record: Graphic) => record.points.every(id => points[id]?.x === document.points[id]!.x && points[id]?.z === document.points[id]!.z);
    const layer = (records: readonly Graphic[], selected: readonly WallSurface[] | undefined): GraphicsRecordInput[] => {
      if (room.originZone === null) {
        if (!selected) throw new Error(`Room ${room.id} needs explicit wall materials before building.`);
        return cell.vertices.flatMap((a, index) => {
          const neighbour = cell.neighbours[index]!;
          if (neighbour >= 0 && topology.cells.find(target => target.id === neighbour)?.sourceId === cell.sourceId) return [];
          const destination = neighbour < 0 ? undefined : topology.cells.find(target => target.id === neighbour);
          const b = cell.vertices[(index + 1) % cell.vertices.length]!;
          return selected.map(material => {
            const start = document.points[material.points[0]], end = document.points[material.points[1]];
            if (!start || !end) throw new Error(`Room ${room.id} has a missing wall material reference endpoint.`);
            const bound = wallForSpan(material, distance(start, end), [a, b], 0, distance(points[a]!, points[b]!));
            const spans = destination ? portalWallSpans(document, bound, document.zones.find(zone => zone.id === destination.recordZone)!) : [bound];
            return spans.map(span => ({ bytes: encodeWallSurface(span) }));
          }).flat();
        });
      }
      const original = records.filter(record => record.wall).flatMap(record => {
        const material = record.wall!, a = document.points[material.points[0]]!, b = document.points[material.points[1]]!;
        const lineId = template.boundaryIds.find(id => {
          const line = document.lines[id]!; return on(a, line, { x: line.x + line.dx, z: line.z + line.dz }) && on(b, line, { x: line.x + line.dx, z: line.z + line.dz });
        });
        const opened = cell.vertices.some((_, edge) => cell.sourceLines[edge] === lineId && newPortalTarget(document, topology, cell, edge));
        const changedLine = changed.some(edge => cell.sourceLines[edge] === lineId);
        if (unchanged && retained(record) && !opened && !changedLine) return [{ sourceOffset: record.offset, bytes: document.files.graphics.slice(record.offset, record.offset + record.size) }];
        if (lineId === undefined) {
          if (translatedWholeRoom) return [{ sourceOffset: record.offset, bytes: encodeWallSurface({ ...material,
            points: [pointId({ x: a.x + dx, z: a.z + dz }), pointId({ x: b.x + dx, z: b.z + dz })] }) }];
          throw new Error(`Room ${room.id} wall at byte ${record.offset} is not attached to a single boundary. Assign its geometry before compiling this room.`);
        }
        const line = document.lines[lineId]!, lengthSquared = line.dx ** 2 + line.dz ** 2;
        const fraction = (p: Position) => ((p.x - line.x) * line.dx + (p.z - line.z) * line.dz) / lengthSquared;
        const { pieces, total } = wallChain(room, lineId);
        const first = fraction(a) * total, last = fraction(b) * total, result: GraphicsRecordInput[] = [], uncut: WallSurface[] = [];
        for (let index = 0; index < cell.vertices.length; index++) {
          if (cell.sourceLines[index] !== lineId) continue;
          const start = points[cell.vertices[index]!]!, end = points[cell.vertices[(index + 1) % cell.vertices.length]!]!;
          const piece = pieces.find(piece => on(start, piece.start, piece.end) && on(end, piece.start, piece.end));
          if (!piece) throw new Error(`Room ${room.id} lost its wall chain while splitting geometry.`);
          const edgeStart = piece.offset + distance(piece.start, start), edgeEnd = piece.offset + distance(piece.start, end);
          const low = Math.max(Math.min(first, last), Math.min(edgeStart, edgeEnd)), high = Math.min(Math.max(first, last), Math.max(edgeStart, edgeEnd));
          if (high <= low) continue;
          const position = (arc: number) => ({ x: start.x + (end.x - start.x) * (arc - edgeStart) / (edgeEnd - edgeStart),
            z: start.z + (end.z - start.z) * (arc - edgeStart) / (edgeEnd - edgeStart) });
          const from = last > first ? low : high, to = last > first ? high : low;
          const bound = wallForSpan(material, distance(a, b), [pointId(position(from)), pointId(position(to))], Math.abs(from - first), Math.abs(to - first));
          uncut.push(bound);
          if (changed.includes(index)) continue;
          const target = newPortalTarget(document, topology, cell, index);
          const spans = target ? portalWallSpans(document, bound, document.zones.find(zone => zone.id === target.recordZone)!) : [bound];
          result.push(...spans.map(span => ({ sourceOffset: record.offset, bytes: encodeWallSurface(span) })));
        }
        if ((!opened && !changedLine) || !uncut.length) return result;
        const starts = new Set(uncut.map(wall => wall.points[0])), ends = new Set(uncut.map(wall => wall.points[1]));
        const start = [...starts].filter(id => !ends.has(id)), end = [...ends].filter(id => !starts.has(id));
        if (start.length !== 1 || end.length !== 1) throw new Error(`Room ${room.id} has an ambiguous doorway material-lighting span.`);
        return result.map(item => ({ ...item, lightingSpan: [points[start[0]!]!, points[end[0]!]!] as const }));
      });
      const fills = changed.flatMap(edge => (selected ?? []).flatMap(material => {
        const start = document.points[material.points[0]], end = document.points[material.points[1]];
        if (!start || !end) throw new Error(`Room ${room.id} has a missing closure material endpoint.`);
        const a = cell.vertices[edge]!, b = cell.vertices[(edge + 1) % cell.vertices.length]!;
        const line = cell.sourceLines[edge];
        const pieces = line == null ? room.vertices.flatMap((start, index) => room.wallSources[index] === null
          ? [{ start, end: room.vertices[(index + 1) % room.vertices.length]!, offset: 0 }] : []) : wallChain(room, line).pieces;
        const piece = pieces.find(piece => on(points[a]!, piece.start, piece.end) && on(points[b]!, piece.start, piece.end));
        if (!piece) throw new Error(`Room ${room.id} lost its doorway closure material chain.`);
        const offset = piece.offset + distance(piece.start, points[a]!);
        const bound = wallForSpan(material, distance(start, end), [a, b], offset, offset + distance(points[a]!, points[b]!));
        const target = newPortalTarget(document, topology, cell, edge);
        const spans = target ? portalWallSpans(document, bound, document.zones.find(zone => zone.id === target.recordZone)!) : [bound];
        return spans.map(span => ({ bytes: encodeWallSurface(span), lightingPoint: closing!.lightingPoint }));
      }));
      return [...original, ...fills];
    };
    const selected = room.originZone === null ? newMaterials.get(room.id) : closing?.walls;
    walls.set(cell.id, { lower: layer(template.lower, selected?.lower), upper: layer(template.upper, selected?.upper ?? (room.originZone === null ? [] : undefined)) });
  }
  return { topology: { ...topology, points }, walls };
}
