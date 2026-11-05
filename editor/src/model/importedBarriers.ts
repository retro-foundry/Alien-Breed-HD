import type { Graphic, LevelDocument } from './level';
import type { Position } from './geometry';
import type { SourceRoom } from './roomSource';
import { sourceBoundaryIndex } from './boundaryMaterials';
import { hasNativeWallMaterials } from './roomMaterials';
import { portalStateAt, portalBarrierAt } from './portalStates';

type NativeBarrierDocument = Pick<LevelDocument, 'zones' | 'lines' | 'points'>;
const cache = new WeakMap<NativeBarrierDocument, ReadonlyMap<number, ReadonlySet<number>>>();
const on = (p: Position, a: Position, b: Position) => (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
  && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);

/** LEVELSAVE WT3/4 writes a full-height opcode-13 boundary and -1 collision.
 * Derive only that proven native contract. A partial decorative strip is not
 * evidence that an otherwise solid boundary is a visibility portal. */
export function importedBarrierLines(document: NativeBarrierDocument): ReadonlyMap<number, ReadonlySet<number>> {
  const previous = cache.get(document); if (previous) return previous;
  const result = new Map<number, ReadonlySet<number>>();
  for (const zone of document.zones) for (const lineId of zone.boundaryIds) {
    const line = document.lines[lineId]!;
    if (line.connect >= 0) continue;
    const end = { x: line.x + line.dx, z: line.z + line.dz }, length = line.dx ** 2 + line.dz ** 2;
    if (!length) continue;
    const references = new Set<number>();
    for (const upper of [false, true]) {
      const roof = upper ? zone.upperCeiling : zone.ceiling, floor = upper ? zone.upperFloor : zone.floor;
      if (floor <= roof) continue;
      const candidates = (upper ? zone.upper : zone.lower).filter(record => record.type === 13 && record.wall!.topFixed <= roof * 256 && record.wall!.bottomFixed >= floor * 256
        && record.points.every(id => on(document.points[id]!, line, end)));
      const intervals = candidates.map(record => {
        const positions = record.points.map(id => { const p = document.points[id]!; return (p.x - line.x) * line.dx + (p.z - line.z) * line.dz; });
        return { from: Math.min(...positions), to: Math.max(...positions), record: record.offset };
      }).sort((a, b) => a.from - b.from);
      let covered = 0;
      for (const interval of intervals) { if (interval.from > covered) break; covered = Math.max(covered, interval.to); }
      if (covered === length) candidates.forEach(record => references.add(record.offset));
    }
    if (references.size) result.set(lineId, references);
  }
  cache.set(document, result); return result;
}

/** Original opposing boundary identities, including partial shared edges. */
export function nativeBoundaryPair(document: NativeBarrierDocument, own: number, other: number): boolean {
  const a = document.lines[own]!, b = document.lines[other]!;
  if (a.dx * b.dx + a.dz * b.dz >= 0 || (b.x - a.x) * a.dz !== (b.z - a.z) * a.dx) return false;
  const length = a.dx ** 2 + a.dz ** 2;
  const from = (b.x - a.x) * a.dx + (b.z - a.z) * a.dz;
  const to = (b.x + b.dx - a.x) * a.dx + (b.z + b.dz - a.z) * a.dz;
  return Math.max(0, Math.min(from, to)) < Math.min(length, Math.max(from, to));
}

export function importedBarrierPair(document: NativeBarrierDocument, own: number, other: number): boolean {
  const lines = importedBarrierLines(document);
  return (lines.has(own) || lines.has(other)) && nativeBoundaryPair(document, own, other);
}

export function importedBarrierHasNeighbour(document: NativeBarrierDocument, line: number): boolean {
  const barriers = importedBarrierLines(document);
  if (!barriers.size) return false;
  if (!barriers.has(line)) return [...barriers.keys()].some(other => nativeBoundaryPair(document, line, other));
  return document.zones.some(zone => zone.boundaryIds.some(other => nativeBoundaryPair(document, line, other)));
}

export function importedBarrierConnection(document: NativeBarrierDocument, room: SourceRoom, other: SourceRoom, start: Position, end: Position): { solid: boolean } | undefined {
  if (!hasNativeWallMaterials(room.originZone, room.wallSources) || !hasNativeWallMaterials(other.originZone, other.wallSources)) return;
  const a = sourceBoundaryIndex(room, start, end), b = sourceBoundaryIndex(other, start, end);
  const own = room.wallSources[a], neighbour = other.wallSources[b];
  if (own == null || neighbour == null || !importedBarrierPair(document, own, neighbour)) return;
  return { solid: document.lines[own]!.connect < 0 };
}

export function barrierBlocksWalking(document: NativeBarrierDocument, room: SourceRoom, other: SourceRoom, start: Position, end: Position): boolean {
  if (portalStateAt(room, start, end, other.id) !== undefined) return portalBarrierAt(room, start, end, other.id);
  return importedBarrierConnection(document, room, other, start, end)?.solid ?? false;
}

/** Explicit Open/Close changes may replace an imported see-through face's
 * opcode. Require intent on that exact source wall, not an unrelated portal. */
export function importedBarrierModeOverride(document: NativeBarrierDocument, room: SourceRoom, record: Graphic): boolean {
  if (record.type !== 13 || !room.boundaryMaterials) return false;
  const lines = importedBarrierLines(document);
  return room.wallSources.some((line, index) => line !== null && lines.get(line)?.has(record.offset) && room.portals?.some(state => !state.barrier && state.boundary === room.boundaryMaterials!.edges[index]));
}

export function importedBarrierModeAt(document: LevelDocument, room: SourceRoom, start: Position, end: Position): boolean {
  const index = sourceBoundaryIndex(room, start, end), line = room.wallSources[index];
  if (line == null || !importedBarrierLines(document).has(line)) return false;
  return !!room.portals?.some(state => !state.barrier && state.boundary === room.boundaryMaterials?.edges[index]
    && document.roomSource.rooms.some(other => other.id === state.neighbour && sourceBoundaryIndex(other, start, end) >= 0));
}
