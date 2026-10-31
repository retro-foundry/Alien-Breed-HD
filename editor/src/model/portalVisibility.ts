/* Portal visibility as feasibility of oriented sightlines through convex cells.
 * A line is a*x+b*z+c=0, oriented along (b,-a). Crossing a negative-area room's
 * directed exit requires L(start)>=0 and L(end)<=0. Each portal therefore adds
 * two linear halfplanes in line space. Four bounded charts cover every line
 * direction. Integer/rational predicates avoid epsilon-based room disappearance.
 * This stage computes membership; legacy clipping/ordering are separate stages. */
import { isConvex, signedArea, type Position } from './geometry';
import type { VisibilityRoom } from './visibilityGeometry';

type Plane = readonly [bigint, bigint, bigint, number?, bigint?];
interface Vertex { t: bigint; c: bigint; w: bigint; edge: Plane }
type Polygon = readonly Vertex[];
interface Cone { polygon: Polygon; axis: number; sign: bigint }
export interface PortalClips { left: number[]; right: number[] }
export interface Sightline { a: number; b: number; c: number }
export interface PortalVisibility {
  visible: Map<number, Set<number>>;
  clips: Map<number, Map<number, PortalClips>>;
  witnesses: Map<number, Map<number, Sightline>>;
  previous: number[]; next: number[];
}
const evaluate = (p: Plane, v: Vertex) => p[0] * v.t + p[1] * v.c + p[2] * v.w;
const coincident = (a: Vertex, b: Vertex) => a.t * b.w === b.t * a.w && a.c * b.w === b.c * a.w;

function intersection(a: Plane, b: Plane, edge: Plane): Vertex {
  let t = a[1] * b[2] - a[2] * b[1], c = a[2] * b[0] - a[0] * b[2], w = a[0] * b[1] - a[1] * b[0];
  if (w === 0n) throw new Error('Parallel visibility boundaries were intersected.');
  if (w < 0n) { t = -t; c = -c; w = -w; }
  return { t, c, w, edge };
}

function hasArea(polygon: Polygon): boolean {
  if (polygon.length < 3) return false;
  const a = polygon[0]!;
  for (let i = 1; i < polygon.length - 1; i++) {
    const b = polygon[i]!, c = polygon[i + 1]!;
    const determinant = a.t * (b.c * c.w - b.w * c.c) - a.c * (b.t * c.w - b.w * c.t) + a.w * (b.t * c.c - b.c * c.t);
    if (determinant !== 0n) return true;
  }
  return false;
}

function clip(polygon: Polygon, boundary: Plane): Polygon {
  const output: Vertex[] = [];
  for (let i = 0; i < polygon.length; i++) {
    const end = polygon[i]!, start = polygon[(i + polygon.length - 1) % polygon.length]!;
    const a = evaluate(boundary, start), b = evaluate(boundary, end);
    if (a >= 0n && b >= 0n) output.push(end);
    else if (a >= 0n) output.push(intersection(end.edge, boundary, end.edge));
    else if (b >= 0n) { output.push(intersection(end.edge, boundary, boundary)); output.push(end); }
  }
  const unique = output.filter((vertex, i) => i === 0 || !coincident(vertex, output[i - 1]!));
  if (unique.length > 1 && coincident(unique[0]!, unique.at(-1)!)) {
    unique[0] = { ...unique[0]!, edge: unique.at(-1)!.edge }; unique.pop();
  }
  return hasArea(unique) ? unique : [];
}

function subset(inner: Polygon, outer: Polygon): boolean {
  return inner.every(vertex => outer.every(boundary => evaluate(boundary.edge, vertex) >= 0n));
}

function initialChart(bound: bigint): Polygon {
  const left: Plane = [1n, 0n, 1n], bottom: Plane = [0n, 1n, bound];
  const right: Plane = [-1n, 0n, 1n], top: Plane = [0n, -1n, bound];
  return [{ t: -1n, c: -bound, w: 1n, edge: left }, { t: 1n, c: -bound, w: 1n, edge: bottom },
    { t: 1n, c: bound, w: 1n, edge: right }, { t: -1n, c: bound, w: 1n, edge: top }];
}

/** Exact planar potential visibility through the supplied open portal graph.
 * Doors use their potentially open graph; current door occlusion is a runtime
 * concern. Concave authoring rooms must be decomposed into convex cells first. */
export function portalVisibility(points: readonly Position[], rooms: readonly VisibilityRoom[], owners?: ReadonlyMap<number, number>): PortalVisibility {
  if (points.length > 2048 || rooms.length > 2048) throw new Error('Visibility compilation exceeds point/cell limits.');
  if (owners && rooms.some(room => !owners.has(room.id))) throw new Error('Every visibility cell must identify its source room.');
  const owner = (id: number) => owners ? owners.get(id)! : id;
  const sources = new Map<number, VisibilityRoom[]>();
  for (const room of rooms) sources.set(owner(room.id), [...(sources.get(owner(room.id)) ?? []), room]);
  if (sources.size > 256) throw new Error('Visibility compilation exceeds 256 runtime rooms.');
  for (const point of points) if (!Number.isSafeInteger(point.x) || !Number.isSafeInteger(point.z) || point.x < -32768 || point.x > 32767 || point.z < -32768 || point.z > 32767)
    throw new Error('Visibility corners must use integer coordinates within the game bounds.');
  const byId = new Map(rooms.map(room => [room.id, room]));
  if (byId.size !== rooms.length) throw new Error('Visibility rooms must have unique identities.');
  for (const room of rooms) {
    if (room.vertices.length < 3 || room.vertices.length > 10 || room.vertices.length !== room.neighbours.length)
      throw new Error(`Room ${room.id} has an invalid portal boundary.`);
    if (room.vertices.some(id => !Number.isInteger(id) || id < 0 || id >= points.length)) throw new Error(`Room ${room.id} has a missing corner.`);
    const outline = room.vertices.map(id => points[id]!);
    if (signedArea(outline) >= 0 || !isConvex(outline)) throw new Error(`Room ${room.id} must be converted to convex runtime cells before visibility compilation.`);
    if (room.neighbours.some(id => id >= 0 && !byId.has(id))) throw new Error(`Room ${room.id} has a portal to a missing room.`);
  }
  const bound = BigInt(Math.max(1, ...points.map(p => Math.abs(p.x) + Math.abs(p.z))));
  const result = new Map<number, Set<number>>();
  const clipping = new Map<number, Map<number, PortalClips>>();
  const witnesses = new Map<number, Map<number, Sightline>>();
  // Same solid-wall connection contract as INITDEFORDER 5702–5724; the
  // runtime uses next/previous to check and near-clip each selected corner.
  const previous = points.map(() => -1), next = points.map(() => -1);
  for (const room of rooms) room.vertices.forEach((id, i) => {
    if (room.neighbours[i]! < 0) { const end = room.vertices[(i + 1) % room.vertices.length]!; next[id] = end; previous[end] = id; }
  });
  for (const [sourceId, sourceCells] of sources) {
    const visible = new Set([sourceId]); result.set(sourceId, visible);
    const cones = new Map<number, Cone[]>();
    for (const source of sourceCells) for (const axis of [0, 1]) for (const sign of [-1n, 1n]) {
      const plane = (id: number, orientation: bigint): Plane => {
        const point = points[id]!;
        return axis === 0 ? [BigInt(point.z) * orientation, orientation, BigInt(point.x) * sign * orientation, id, orientation]
          : [BigInt(point.x) * orientation, orientation, BigInt(point.z) * sign * orientation, id, orientation];
      };
      const reached = new Map<number, Polygon[]>(), pending = [{ room: source, polygon: initialChart(bound) }];
      while (pending.length) {
        const state = pending.pop()!, previous = reached.get(state.room.id) ?? [];
        if (previous.some(polygon => subset(state.polygon, polygon))) continue;
        reached.set(state.room.id, [...previous.filter(polygon => !subset(polygon, state.polygon)), state.polygon]);
        visible.add(owner(state.room.id));
        state.room.neighbours.forEach((neighbour, edge) => {
          if (neighbour < 0) return;
          const a = state.room.vertices[edge]!, b = state.room.vertices[(edge + 1) % state.room.vertices.length]!;
          let polygon = clip(state.polygon, plane(a, 1n));
          if (polygon.length) polygon = clip(polygon, plane(b, -1n));
          if (polygon.length) pending.push({ room: byId.get(neighbour)!, polygon });
        });
      }
      for (const [id, polygons] of reached) { const target = owner(id); cones.set(target, [...(cones.get(target) ?? []), ...polygons.map(polygon => ({ polygon, axis, sign }))]); }
    }
    const sourceClips = new Map<number, PortalClips>(); clipping.set(sourceId, sourceClips);
    const sourceWitnesses = new Map<number, Sightline>(); witnesses.set(sourceId, sourceWitnesses);
    for (const target of visible) {
      if (target === sourceId) { sourceClips.set(target, { left: [], right: [] }); continue; }
      const regions = cones.get(target)!, candidates = new Map<string, { id: number; orientation: bigint }>();
      const witness = regions[0]!, sum = witness.polygon.reduce((total, vertex) => ({ t: total.t + vertex.t, c: total.c + vertex.c, w: total.w + vertex.w }), { t: 0n, c: 0n, w: 0n });
      // A positive sum of every extreme ray lies strictly inside this cone,
      // providing a concrete sightline for inspection and independent checks.
      sourceWitnesses.set(target, { a: Number(witness.axis === 0 ? witness.sign * sum.w : sum.t), b: Number(witness.axis === 0 ? sum.t : witness.sign * sum.w), c: Number(sum.c) });
      for (const region of regions) for (const vertex of region.polygon) {
        const id = vertex.edge[3], orientation = vertex.edge[4];
        if (id !== undefined && orientation !== undefined && (orientation > 0n ? next[id]! >= 0 : previous[id]! >= 0))
          candidates.set(`${id}:${orientation}`, { id, orientation });
      }
      const rays = regions.flatMap(region => region.polygon.map(vertex => region.axis === 0
        ? [region.sign * vertex.w, vertex.t, vertex.c] as const
        : [vertex.t, region.sign * vertex.w, vertex.c] as const));
      const left: number[] = [], right: number[] = [];
      // A legacy clip word is a point halfplane. Emit only halfplanes that
      // contain EVERY feasible portal route, so alternate openings remain
      // visible. Empty lists mean no common angular restriction exists.
      for (const { id, orientation } of candidates.values()) {
        const point = points[id]!, x = BigInt(point.x), z = BigInt(point.z);
        if (rays.every(ray => (ray[0] * x + ray[1] * z + ray[2]) * orientation >= 0n))
          (orientation > 0n ? left : right).push(id);
      }
      sourceClips.set(target, { left: left.sort((a, b) => a - b), right: right.sort((a, b) => a - b) });
    }
  }
  return { visible: result, clips: clipping, witnesses, previous, next };
}
