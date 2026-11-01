import { checkedInteger } from './binary';
import type { Position } from './geometry';
import type { SourceRoom } from './roomSource';
import { restoreWallMaterials, type WallMaterial } from './wallMaterials';

export interface BoundaryMaterial extends Omit<WallMaterial, 'record'> { readonly edge: number }
export interface BoundaryMaterials {
  readonly nextId: number;
  readonly edges: readonly number[];
  readonly overrides: readonly BoundaryMaterial[];
}

export function sourceBoundaryIndex(room: Pick<SourceRoom, 'vertices'>, start: Position, end: Position): number {
  const on = (p: Position, a: Position, b: Position) => (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
    && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);
  return room.vertices.findIndex((a, i) => on(start, a, room.vertices[(i + 1) % room.vertices.length]!)
    && on(end, a, room.vertices[(i + 1) % room.vertices.length]!));
}

export function restoreBoundaryMaterials(value: unknown, corners: number): BoundaryMaterials {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid boundary wall materials.');
  const input = value as Record<string, unknown>;
  if (Object.keys(input).some(key => !['nextId', 'edges', 'overrides'].includes(key))
    || !Array.isArray(input.edges) || input.edges.length !== corners || !Array.isArray(input.overrides))
    throw new Error('Boundary materials need one identity per source edge and an override list.');
  const nextId = checkedInteger(input.nextId as number, 1, 16777216, 'Next boundary identity');
  const edges = input.edges.map(edge => checkedInteger(edge, 0, nextId - 1, 'Boundary identity'));
  const overrides = input.overrides.map(item => {
    if (!item || typeof item !== 'object' || Array.isArray(item) || !('edge' in item)) throw new Error('Invalid boundary material override.');
    const { edge, ...material } = item as Record<string, unknown>;
    if ('record' in material) throw new Error('Boundary materials cannot name native records.');
    return { ...material, record: checkedInteger(edge as number, 0, nextId - 1, 'Painted boundary identity') };
  });
  const normalized = restoreWallMaterials(overrides).map(({ record, ...material }) => Object.freeze({ ...material, edge: record }));
  return Object.freeze({ nextId, edges: Object.freeze(edges), overrides: Object.freeze(normalized) });
}

/** Redrawing an unrelated edge allocates a new ID; it never inherits old paint. */
export function remapBoundaryMaterials(room: Pick<SourceRoom, 'vertices' | 'boundaryMaterials'>, edges: readonly { start: Position; end: Position }[]): BoundaryMaterials | undefined {
  const previous = room.boundaryMaterials;
  if (!previous) return;
  let nextId = previous.nextId;
  const ids = edges.map(edge => {
    const index = sourceBoundaryIndex(room, edge.start, edge.end);
    return index < 0 ? nextId++ : previous.edges[index]!;
  });
  return restoreBoundaryMaterials({ ...previous, nextId, edges: ids }, ids.length);
}
