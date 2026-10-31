/* The 2,496 collision outlines in the 16 shipped maps all have negative signed
 * area in game X/Z coordinates. Authoring accepts either drawing direction;
 * this module owns the engine's winding and shared-edge requirements. */
export interface Position { x: number; z: number }
export interface RoomOutline { id: string; vertices: Position[] }
export interface RoomEdge { room: string; index: number; start: Position; end: Position; neighbour: string | null }

function same(a: Position, b: Position): boolean { return a.x === b.x && a.z === b.z; }
function cross(a: Position, b: Position, c: Position): number { return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x); }
export function signedArea(vertices: readonly Position[]): number {
  let area = 0;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i]!, b = vertices[(i + 1) % vertices.length]!;
    area += a.x * b.z - b.x * a.z;
  }
  return area / 2;
}

function onSegment(point: Position, a: Position, b: Position): boolean {
  return cross(a, b, point) === 0 && point.x >= Math.min(a.x, b.x) && point.x <= Math.max(a.x, b.x)
    && point.z >= Math.min(a.z, b.z) && point.z <= Math.max(a.z, b.z);
}

function intersects(a: Position, b: Position, c: Position, d: Position): boolean {
  const abC = cross(a, b, c), abD = cross(a, b, d), cdA = cross(c, d, a), cdB = cross(c, d, b);
  if (Math.sign(abC) !== Math.sign(abD) && Math.sign(cdA) !== Math.sign(cdB)) return true;
  return (abC === 0 && onSegment(c, a, b)) || (abD === 0 && onSegment(d, a, b)) || (cdA === 0 && onSegment(a, c, d)) || (cdB === 0 && onSegment(b, c, d));
}

export function normaliseRoomOutline(input: readonly Position[]): Position[] {
  const vertices: Position[] = [];
  for (const point of input) {
    if (!Number.isSafeInteger(point.x) || !Number.isSafeInteger(point.z) || point.x < -32768 || point.x > 32767 || point.z < -32768 || point.z > 32767)
      throw new Error('Room corners must sit on whole coordinates within the level bounds.');
    if (!vertices.length || !same(vertices.at(-1)!, point)) vertices.push({ ...point });
  }
  if (vertices.length > 1 && same(vertices[0]!, vertices.at(-1)!)) vertices.pop();
  if (vertices.length < 3) throw new Error('A room needs at least three different corners.');
  if (vertices.length > 2048) throw new Error('A room has too many corners for the game engine.');
  for (let i = 0; i < vertices.length; i++) for (let j = i + 1; j < vertices.length; j++) {
    if (j === i + 1 || (i === 0 && j === vertices.length - 1)) continue;
    if (intersects(vertices[i]!, vertices[(i + 1) % vertices.length]!, vertices[j]!, vertices[(j + 1) % vertices.length]!))
      throw new Error(`The room outline crosses itself near corners ${i + 1} and ${j + 1}. Move those corners apart.`);
  }
  const area = signedArea(vertices);
  if (area === 0) throw new Error('These corners form a line. Move a corner to give the room some space.');
  if (area > 0) vertices.reverse();
  // A stable first corner makes opposite gestures produce the same result too.
  let first = 0;
  for (let i = 1; i < vertices.length; i++)
    if (vertices[i]!.x < vertices[first]!.x || (vertices[i]!.x === vertices[first]!.x && vertices[i]!.z < vertices[first]!.z)) first = i;
  return [...vertices.slice(first), ...vertices.slice(0, first)];
}

export function isConvex(vertices: readonly Position[]): boolean {
  const normalised = normaliseRoomOutline(vertices);
  return normalised.every((point, i) => cross(point, normalised[(i + 1) % normalised.length]!, normalised[(i + 2) % normalised.length]!) <= 0);
}

function insideTriangle(point: Position, a: Position, b: Position, c: Position): boolean {
  return cross(a, b, point) <= 0 && cross(b, c, point) <= 0 && cross(c, a, point) <= 0;
}

/* Split a concave authored room into engine-friendly convex pieces without
 * asking the user to draw or orient individual runtime sectors. */
export function convexParts(input: readonly Position[], maximumCorners = 10): Position[][] {
  if (!Number.isInteger(maximumCorners) || maximumCorners < 3 || maximumCorners > 10)
    throw new Error('Runtime rooms support between three and ten corners.');
  const outline = withoutStraightCorners(normaliseRoomOutline(input));
  if (isConvex(outline) && outline.length <= maximumCorners) return [outline];
  const remaining = outline.slice(), parts: Position[][] = [];
  while (remaining.length > 3) {
    let ear = -1;
    for (let i = 0; i < remaining.length; i++) {
      const a = remaining[(i + remaining.length - 1) % remaining.length]!, b = remaining[i]!, c = remaining[(i + 1) % remaining.length]!;
      if (cross(a, b, c) >= 0) continue;
      if (remaining.some((point, j) => j !== i && j !== (i + remaining.length - 1) % remaining.length && j !== (i + 1) % remaining.length && insideTriangle(point, a, b, c))) continue;
      ear = i; parts.push(normaliseRoomOutline([a, b, c])); break;
    }
    if (ear < 0) throw new Error('The room outline could not be divided into valid game geometry. Check overlapping corners.');
    remaining.splice(ear, 1);
  }
  parts.push(normaliseRoomOutline(remaining));
  // Merge adjacent triangles while their union is convex and fits the actual
  // ordering-mask limit. A complex drawn room should not consume a runtime
  // zone per triangle when a larger convex piece is valid.
  for (;;) {
    let merged = false;
    for (let i = 0; i < parts.length && !merged; i++) for (let j = i + 1; j < parts.length && !merged; j++) {
      const a = parts[i]!, b = parts[j]!;
      for (let ai = 0; ai < a.length && !merged; ai++) for (let bi = 0; bi < b.length && !merged; bi++) {
        if (!same(a[ai]!, b[(bi + 1) % b.length]!) || !same(a[(ai + 1) % a.length]!, b[bi]!)) continue;
        const pathA = a.map((_, k) => a[(ai + 1 + k) % a.length]!);
        const pathB = b.map((_, k) => b[(bi + 1 + k) % b.length]!);
        const joined = withoutStraightCorners([...pathA, ...pathB.slice(1, -1)]);
        if (joined.length > maximumCorners || !isConvex(joined)) continue;
        parts[i] = normaliseRoomOutline(joined); parts.splice(j, 1); merged = true;
      }
    }
    if (!merged) return parts;
  }
}

function withoutStraightCorners(input: readonly Position[]): Position[] {
  const result = input.slice();
  for (let i = 0; i < result.length && result.length > 3;) {
    const a = result[(i + result.length - 1) % result.length]!, b = result[i]!, c = result[(i + 1) % result.length]!;
    if (cross(a, b, c) === 0 && onSegment(b, a, c)) { result.splice(i, 1); i = 0; }
    else i++;
  }
  return result;
}

export function buildRoomEdges(input: readonly RoomOutline[], anchors: readonly Position[] = []): RoomEdge[] {
  const ids = new Set<string>();
  const rooms = input.map(room => {
    if (ids.has(room.id)) throw new Error('Each room must have a unique identity.');
    ids.add(room.id);
    return { ...room, vertices: normaliseRoomOutline(room.vertices) };
  });
  const allCorners = [...rooms.flatMap(room => room.vertices), ...anchors];
  const edges: RoomEdge[] = [];
  for (const room of rooms) {
    let roomEdge = 0;
    for (let i = 0; i < room.vertices.length; i++) {
      const a = room.vertices[i]!, b = room.vertices[(i + 1) % room.vertices.length]!;
      const dx = b.x - a.x, dz = b.z - a.z;
      // Split a wall where an adjacent room meets only part of it. Portal
      // pairing then works regardless of corner count or gesture direction.
      const middle = allCorners.filter(point => !same(point, a) && !same(point, b) && onSegment(point, a, b));
      middle.sort((p, q) => ((p.x - a.x) * dx + (p.z - a.z) * dz) - ((q.x - a.x) * dx + (q.z - a.z) * dz));
      const points = [a, ...middle.filter((point, j) => j === 0 || !same(point, middle[j - 1]!)), b];
      for (let j = 0; j < points.length - 1; j++) edges.push({ room: room.id, index: roomEdge++,
        start: points[j]!, end: points[j + 1]!, neighbour: null });
    }
  }
  const key = (a: Position, b: Position) => `${a.x},${a.z}:${b.x},${b.z}`;
  const directed = new Map<string, RoomEdge>();
  for (const edge of edges) {
    const ownKey = key(edge.start, edge.end), reverseKey = key(edge.end, edge.start);
    if (directed.has(ownKey)) throw new Error('Two rooms overlap along the same wall. Move one room or join their outlines.');
    const other = directed.get(reverseKey);
    if (other) {
      if (other.neighbour !== null) throw new Error('More than two rooms share this wall.');
      other.neighbour = edge.room; edge.neighbour = other.room;
    }
    directed.set(ownKey, edge);
  }
  return edges;
}
