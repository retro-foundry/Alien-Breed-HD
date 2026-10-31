import { importedAnimationTables, type AnimationTargetContext, type MovingRoomAnimation } from './animationBuild';
import { BinaryReader, checkedInteger } from './binary';
import type { CellGraphics } from './cellGraphics';
import type { CollisionGeometry } from './collisionBuild';
import type { GeneratedGraphicsTarget, SelectGraphicsTarget } from './generatedGraphics';
import { readGraphicRecord, type Graphic } from './graphics';
import type { LevelDocument } from './level';
import type { PointLightingBinding } from './pointLighting';
import { resolveAnimationRoom } from './animationRooms';

const key = (pointer: number, context: AnimationTargetContext) => `${pointer}:${context.kind}:${context.zone}:${context.line ?? ''}`;
const on = (p: { x: number; z: number }, a: { x: number; z: number }, b: { x: number; z: number }) =>
  (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
  && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);

/** Reattach the native Anims.s write targets to generated geometry. Keep
 * timing/conditions/state/phase bytes intact. Collision flags and graphics
 * targets have separate identities; neither is an interchangeable wall index.
 * Water entries can fan out within their shared native animation group.
 * Moving-room and switch controller grouping requires a separate runtime
 * contract, so splitting those controlled volumes is rejected explicitly.
 */
export function buildAnimationAttachments(document: LevelDocument, graphics: CellGraphics, collision: CollisionGeometry) {
  const { tables } = importedAnimationTables(document), topology = graphics.topology;
  const points = topology.points.map(point => ({ ...point })), addedLighting = new Map<number, PointLightingBinding>();
  const origins = new Map<number, number[]>();
  for (const cell of topology.cells) if (cell.originZone !== null) origins.set(cell.originZone, [...(origins.get(cell.originZone) ?? []), cell.id]);
  const rooms = (id: number, aliases = false) => {
    const source = resolveAnimationRoom(document, id, aliases), result = source === undefined ? undefined : origins.get(source);
    if (!result?.length) throw new Error(`Animation room ${id} was deleted or has no generated geometry. Remove or reassign its animation before building.`);
    return result;
  };
  const singleRoom = (id: number, kind: string, aliases = false) => {
    const result = rooms(id, aliases);
    if (result.length !== 1) throw new Error(`${kind} room ${id} was split into several cells. Synchronized controller grouping is required before building this moving or switched room.`);
    return result[0]!;
  };
  const originals = new Map(document.zones.flatMap(zone => [...zone.lower, ...zone.upper].map(record => [record.offset, { record, owner: zone.id }] as const)));
  type Target = GeneratedGraphicsTarget & { record: Graphic };
  const targets = new Map<number, Target[]>();
  for (const cell of topology.cells) {
    const stream = graphics.streams.get(cell.id);
    if (!stream) throw new Error(`Animation attachments have no graphics stream for room ${cell.id}.`);
    for (const [upper, records] of [[false, stream.lower], [true, stream.upper]] as const) records?.forEach((item, index) => {
      if (item.sourceOffset === undefined || originals.get(item.sourceOffset)?.owner !== cell.originZone) return;
      const record = readGraphicRecord(new BinaryReader(item.bytes, 'animation target'), 0, points.length);
      targets.set(item.sourceOffset, [...(targets.get(item.sourceOffset) ?? []), { room: cell.id, upper, index, record }]);
    });
  }
  const selected = new Map<string, GeneratedGraphicsTarget>();
  const bind = (pointer: number, context: AnimationTargetContext, target: Target) => {
    const id = key(pointer, context), previous = selected.get(id);
    if (previous && (previous.room !== target.room || previous.upper !== target.upper || previous.index !== target.index))
      throw new Error('Several animation writes need different graphics records with the same attachment identity.');
    selected.set(id, { room: target.room, upper: target.upper, index: target.index });
  };
  const choose = (pointer: number, context: AnimationTargetContext, offset = 0) => {
    const candidates = targets.get(pointer - offset) ?? [];
    // Runtime IDs can differ from source IDs. A target in the controlled room
    // is unambiguous when available; otherwise preserve unique source ownership.
    const local = candidates.filter(target => target.room === context.zone);
    const available = local.length ? local : candidates;
    if (available.length !== 1) throw new Error(`Animation target ${pointer} has ${available.length} matching generated records. Its attachment needs explicit grouping or reassignment.`);
    bind(pointer, context, available[0]!); return available[0]!;
  };
  const lines = new Map<number, { id: number; a: number; b: number }[]>();
  const collisionReader = new BinaryReader(collision.bytes, 'compiled animation collision');
  const lineIds = collision.boundaries.flat();
  if (lineIds.length * 16 !== collision.bytes.length || new Set(lineIds).size !== lineIds.length)
    throw new Error('Animation attachments need complete, unique compiled collision identities.');
  topology.cells.forEach((cell, index) => {
    if (collision.boundaries[index]?.length !== cell.vertices.length) throw new Error('Animation attachments and collision boundaries disagree.');
    cell.sourceLines.forEach((source, edge) => {
      const id = checkedInteger(collision.boundaries[index]![edge]!, 0, lineIds.length - 1, 'Animated collision identity');
      const a = topology.points[cell.vertices[edge]!]!, b = topology.points[cell.vertices[(edge + 1) % cell.vertices.length]!]!;
      const offset = id * 16;
      if (collisionReader.i16(offset) !== a.x || collisionReader.i16(offset + 2) !== a.z
        || collisionReader.i16(offset + 4) !== b.x - a.x || collisionReader.i16(offset + 6) !== b.z - a.z
        || collisionReader.i16(offset + 8) !== cell.neighbours[edge])
        throw new Error('Compiled collision coordinates or neighbours do not match their animation attachment identities.');
      if (source === null) return;
      lines.set(source, [...(lines.get(source) ?? []), { id: collision.boundaries[index]![edge]!, a: cell.vertices[edge]!, b: cell.vertices[(edge + 1) % cell.vertices.length]! }]);
    });
  });
  const moving = (records: readonly MovingRoomAnimation[] | null, kind: 'door' | 'lift') => records?.map(record => {
    const zone = singleRoom(record.zone, kind, kind === 'lift');
    choose(record.graphic, { kind, zone });
    const walls = record.walls.flatMap(wall => {
      const candidates = targets.get(wall.graphic) ?? [], boundaries = lines.get(wall.line) ?? [];
      if (!candidates.length || !boundaries.length) throw new Error(`${kind} room ${record.zone} lost an animated wall or collision attachment.`);
      return candidates.map(target => {
        const [a, b] = target.record.points.map(id => points[id]!);
        const matching = boundaries.filter(line => on(a!, points[line.a]!, points[line.b]!) && on(b!, points[line.a]!, points[line.b]!));
        // A unique authored line/record pair is an explicit relationship even
        // for legacy decorative geometry offset from its collision segment.
        const available = matching.length ? matching : candidates.length === 1 && boundaries.length === 1 ? boundaries : [];
        if (available.length !== 1) throw new Error(`${kind} room ${record.zone} has an ambiguous animated collision-wall attachment.`);
        const line = available[0]!.id; bind(wall.graphic, { kind: 'moving-wall', zone, line }, target);
        return { ...wall, line };
      });
    });
    if (new Set(walls.map(wall => wall.line)).size !== walls.length)
      throw new Error(`${kind} room ${record.zone} needs grouped graphics writes for a shared animated collision line.`);
    checkedInteger(walls.length, 0, 64, 'Generated moving wall count');
    return { ...record, zone, walls };
  }) ?? null;
  const doors = moving(tables.doors, 'door'), lifts = moving(tables.lifts, 'lift');
  const water = tables.water.map(group => ({ ...group, entries: group.entries.flatMap(entry => rooms(entry.zone, true).map(zone => {
    choose(entry.graphic, { kind: 'water', zone }); return { ...entry, zone };
  })) }));
  const switches = tables.switches?.map(record => {
    if (record.zone < 0) return record;
    const zone = singleRoom(record.zone, 'Switch'), target = choose(record.graphic, { kind: 'switch', zone }, 10);
    const original = originals.get(record.graphic - 10)!.record;
    const [a, b] = original.points.map(id => document.points[id]!), [c, d] = target.record.points.map(id => points[id]!);
    const ox = b!.x - a!.x, oz = b!.z - a!.z, nx = d!.x - c!.x, nz = d!.z - c!.z;
    const length = Math.hypot(ox, oz) * Math.hypot(nx, nz);
    if (!length) throw new Error('A switch is attached to a zero-length wall.');
    const cosine = (ox * nx + oz * nz) / length, sine = (ox * nz - oz * nx) / length;
    const pair = [record.point, record.point + 1].map(id => {
      const point = document.points[id]!, x = point.x - (a!.x + b!.x) / 2, z = point.z - (a!.z + b!.z) / 2;
      return { x: checkedInteger(Math.round((c!.x + d!.x) / 2 + cosine * x - sine * z), -32768, 32767, 'Switch activation X'),
        z: checkedInteger(Math.round((c!.z + d!.z) / 2 + sine * x + cosine * z), -32768, 32767, 'Switch activation Z') };
    });
    if (pair.every((point, index) => point.x === document.points[record.point + index]!.x && point.z === document.points[record.point + index]!.z)) return { ...record, zone };
    if (points.length + 2 > 2048) throw new Error('Switch activation pairs exceed the engine limit of 2,048 points.');
    const point = points.length;
    pair.forEach((position, index) => { points.push(position); addedLighting.set(point + index, { kind: 'copy', source: record.point + index }); });
    return { ...record, zone, point };
  }) ?? null;
  const selectTarget: SelectGraphicsTarget = (pointer, context, candidates) => {
    const target = selected.get(key(pointer, context));
    if (!target || !candidates.some(candidate => candidate.room === target.room && candidate.upper === target.upper && candidate.index === target.index))
      throw new Error(`Animation target ${pointer} has no valid compiled attachment.`);
    return target;
  };
  return { tables: { ...tables, doors, lifts, water, switches }, selectTarget, addedLighting,
    graphics: { ...graphics, topology: { ...topology, points } } satisfies CellGraphics };
}
