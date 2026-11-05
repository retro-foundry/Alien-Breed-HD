import { importedAnimationTables, type AnimationTargetContext, type MovingRoomAnimation, type SwitchAnimation } from './animationBuild';
import { BinaryReader, checkedInteger } from './binary';
import { barrierCollisionTopology } from './seeThroughBarriers';
import type { CellGraphics } from './cellGraphics';
import type { CollisionGeometry } from './collisionBuild';
import type { GeneratedGraphicsTarget, SelectGraphicsTarget } from './generatedGraphics';
import { readGraphicRecord, type Graphic } from './graphics';
import type { LevelDocument } from './level';
import type { PointLightingBinding } from './pointLighting';
import { resolveAnimationRoom } from './animationRooms';
import { hasNativeWallMaterials } from './roomMaterials';
import type { MovingControllerCopy } from './movingControllerCopies';
import type { SwitchCopy } from './switchCopies';

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
export function buildAnimationAttachments(document: LevelDocument, graphics: CellGraphics, collision: CollisionGeometry, copies: readonly MovingControllerCopy[] = [], switchCopies: readonly SwitchCopy[] = []) {
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
  const waterTargets = new Map<number, Target[]>();
  const copiedTargets = new Map<number, Target[]>();
  const sourceRooms = new Map(document.roomSource.rooms.map(room => [room.id, room]));
  for (const cell of topology.cells) {
    const stream = graphics.streams.get(cell.id);
    if (!stream) throw new Error(`Animation attachments have no graphics stream for room ${cell.id}.`);
    for (const [upper, records] of [[false, stream.lower], [true, stream.upper]] as const) records?.forEach((item, index) => {
      if (item.sourceOffset === undefined) return;
      const owner = originals.get(item.sourceOffset)?.owner;
      const inheritedMaterials = owner === cell.recordZone && hasNativeWallMaterials(cell.originZone, sourceRooms.get(cell.sourceId)?.wallSources ?? []);
      if (owner !== cell.originZone && !inheritedMaterials) return;
      const record = readGraphicRecord(new BinaryReader(item.bytes, 'animation target'), 0, points.length);
      const target = { room: cell.id, upper, index, record };
      if (owner === cell.originZone) targets.set(item.sourceOffset, [...(targets.get(item.sourceOffset) ?? []), target]);
      else if (inheritedMaterials) copiedTargets.set(item.sourceOffset, [...(copiedTargets.get(item.sourceOffset) ?? []), target]);
      if (record.type === 7 && inheritedMaterials) waterTargets.set(item.sourceOffset, [...(waterTargets.get(item.sourceOffset) ?? []), target]);
    });
  }
  const selected = new Map<string, GeneratedGraphicsTarget>();
  const bind = (pointer: number, context: AnimationTargetContext, target: Target) => {
    const id = key(pointer, context), previous = selected.get(id);
    if (previous && (previous.room !== target.room || previous.upper !== target.upper || previous.index !== target.index))
      throw new Error('Several animation writes need different graphics records with the same attachment identity.');
    selected.set(id, { room: target.room, upper: target.upper, index: target.index });
  };
  const choose = (pointer: number, context: AnimationTargetContext, offset = 0, availableTargets = targets) => {
    const candidates = availableTargets.get(pointer - offset) ?? [];
    // Runtime IDs can differ from source IDs. A target in the controlled room
    // is unambiguous when available; otherwise preserve unique source ownership.
    const local = candidates.filter(target => target.room === context.zone);
    const available = local.length ? local : candidates;
    if (available.length !== 1) throw new Error(`Animation target ${pointer} has ${available.length} matching generated records. Its attachment needs explicit grouping or reassignment.`);
    bind(pointer, context, available[0]!); return available[0]!;
  };
  const lines = new Map<number, { id: number; a: number; b: number }[]>();
  const copiedLines = new Map<number, { id: number; a: number; b: number; sourceId: string }[]>();
  const collisionReader = new BinaryReader(collision.bytes, 'compiled animation collision');
  const collisionTopology = barrierCollisionTopology(document, topology);
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
        || collisionReader.i16(offset + 8) !== collisionTopology.cells[index]!.neighbours[edge])
        throw new Error('Compiled collision coordinates or neighbours do not match their animation attachment identities.');
      if (source === null) return;
      const line = { id: collision.boundaries[index]![edge]!, a: cell.vertices[edge]!, b: cell.vertices[(edge + 1) % cell.vertices.length]! };
      if (cell.originZone !== null) lines.set(source, [...(lines.get(source) ?? []), line]);
      else copiedLines.set(source, [...(copiedLines.get(source) ?? []), { ...line, sourceId: cell.sourceId }]);
    });
  });
  const attachMovingWalls = (record: MovingRoomAnimation, zone: number, subject: string,
    candidatesFor: (graphic: number) => readonly Target[],
    boundariesFor: (line: number) => readonly { id: number; a: number; b: number }[]) => {
    const walls = record.walls.flatMap(wall => {
      const candidates = candidatesFor(wall.graphic), boundaries = boundariesFor(wall.line);
      if (!candidates.length || !boundaries.length) throw new Error(`${subject} lost an animated wall or collision attachment.`);
      return candidates.map(target => {
        const [a, b] = target.record.points.map(id => points[id]!);
        const matching = boundaries.filter(line => on(a!, points[line.a]!, points[line.b]!) && on(b!, points[line.a]!, points[line.b]!));
        // A unique authored line/record pair is an explicit relationship even
        // for legacy decorative geometry offset from its collision segment.
        const available = matching.length ? matching : candidates.length === 1 && boundaries.length === 1 ? boundaries : [];
        if (available.length !== 1) throw new Error(`${subject} has an ambiguous animated collision-wall attachment.`);
        const line = available[0]!.id; bind(wall.graphic, { kind: 'moving-wall', zone, line }, target);
        return { ...wall, line };
      });
    });
    if (new Set(walls.map(wall => wall.line)).size !== walls.length)
      throw new Error(`${subject} needs grouped graphics writes for a shared animated collision line.`);
    checkedInteger(walls.length, 0, 64, 'Generated moving wall count');
    return walls;
  };
  const moving = (records: readonly MovingRoomAnimation[] | null, kind: 'door' | 'lift') => records?.map(record => {
    const zone = singleRoom(record.zone, kind, kind === 'lift');
    choose(record.graphic, { kind, zone });
    const walls = attachMovingWalls(record, zone, `${kind} room ${record.zone}`, graphic => targets.get(graphic) ?? [], line => lines.get(line) ?? []);
    return { ...record, zone, walls };
  }) ?? null;
  const copyMoving = (kind: 'door' | 'lift') => copies.filter(copy => copy.kind === kind).map(copy => {
    const record = (kind === 'door' ? tables.doors : tables.lifts)?.[copy.id];
    if (!record) throw new Error('A copied moving controller lost its native template.');
    const cells = topology.cells.filter(cell => cell.sourceId === copy.room);
    if (cells.length !== 1) throw new Error('A copied moving controller requires one generated room.');
    const zone = cells[0]!.id;
    const mappedTargets = (pointer: number) => {
      const owner = originals.get(pointer)?.owner;
      const room = owner === undefined ? undefined : copy.graphicsRooms.get(owner);
      return (copiedTargets.get(pointer) ?? []).filter(target => topology.cells.find(cell => cell.id === target.room)!.sourceId === room);
    };
    const surface = mappedTargets(record.graphic);
    if (surface.length !== 1) throw new Error('A copied controller needs one explicitly mapped moving surface.');
    bind(record.graphic, { kind, zone }, surface[0]!);
    const walls = attachMovingWalls(record, zone, `Copied ${kind} controller`, mappedTargets,
      source => (copiedLines.get(source) ?? []).filter(line => line.sourceId === copy.collisionRooms.get(source)));
    return { ...record, header: copy.header?.slice() ?? record.header.slice(), zone, walls };
  });
  const append = (original: readonly MovingRoomAnimation[] | null, copies: readonly MovingRoomAnimation[]) => copies.length ? [...(original ?? []), ...copies] : original;
  const doors = append(moving(tables.doors, 'door'), copyMoving('door'));
  const lifts = append(moving(tables.lifts, 'lift'), copyMoving('lift'));
  // Anims.s DoWaterAnims writes one shared phase/height to every listed native
  // polygon and zone header. Material copies need distinct pointer attachments
  // in that same group, without borrowing the original room's runtime identity.
  const water = tables.water.map(group => ({ ...group, entries: group.entries.flatMap(entry => {
    const source = resolveAnimationRoom(document, entry.zone, true);
    const retained = (source === undefined ? [] : origins.get(source) ?? []).filter(zone => sourceRooms.get(topology.cells.find(cell => cell.id === zone)!.sourceId)!.waterSurface === undefined).map(zone => {
      choose(entry.graphic, { kind: 'water', zone }); return { ...entry, zone };
    });
    const copies = (waterTargets.get(entry.graphic) ?? []).filter(target => {
      const cell = topology.cells.find(cell => cell.id === target.room)!;
      return cell.originZone === null && sourceRooms.get(cell.sourceId)!.waterSurface === undefined;
    });
    if (copies.some(target => topology.cells.find(cell => cell.id === target.room)!.recordZone !== source))
      throw new Error('Copied water has a cross-room animation attachment. Map its polygon and controlled room together before building.');
    return [...retained, ...[...new Set(copies.map(target => target.room))].map(zone => {
      choose(entry.graphic, { kind: 'water', zone }, 0, waterTargets); return { ...entry, zone };
    })];
  }) }));
  const attachSwitch = (record: SwitchAnimation, zone: number, target: Target, independent = false) => {
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
    if (!independent && pair.every((point, index) => point.x === document.points[record.point + index]!.x && point.z === document.points[record.point + index]!.z)) return { ...record, zone };
    if (points.length + 2 > 2048) throw new Error('Switch activation pairs exceed the engine limit of 2,048 points.');
    const point = points.length;
    pair.forEach((position, index) => { points.push(position); addedLighting.set(point + index, { kind: 'copy', source: record.point + index }); });
    return { ...record, zone, point };
  };
  const switches = tables.switches?.map((record, slot) => {
    // SwitchRoutine's timer runs before its zone check. Clear every mutable
    // byte and pointer when retiring an imported controller.
    if (document.roomSource.removedNativeSwitches?.includes(slot) || document.roomSource.clearedSwitchSlots?.includes(slot)) return { bytes: new Uint8Array(14), zone: -1, point: 0, graphic: 0 };
    if (record.zone < 0) return record;
    const zone = singleRoom(record.zone, 'Switch'), target = choose(record.graphic, { kind: 'switch', zone }, 10);
    return attachSwitch(record, zone, target);
  }) ?? null;
  for (const copy of switchCopies) {
    const record = tables.switches?.[copy.id];
    if (!record || !switches || record.zone < 0 || switches[copy.slot]!.zone >= 0) throw new Error('A copied switch lost its active template or unused destination slot.');
    const cells = topology.cells.filter(cell => cell.sourceId === copy.room);
    if (cells.length !== 1) throw new Error('A copied switch needs one explicit activation room.');
    const zone = cells[0]!.id;
    const targets = (copiedTargets.get(record.graphic - 10) ?? []).filter(target => topology.cells.find(cell => cell.id === target.room)!.sourceId === copy.graphicsRoom);
    if (targets.length !== 1) throw new Error(`A copied switch needs one independently mapped graphics record; found ${targets.length}. Restore its complete wall before building.`);
    bind(record.graphic, { kind: 'switch', zone }, targets[0]!);
    switches[copy.slot] = attachSwitch({ ...record, bytes: copy.bytes?.slice() ?? record.bytes.slice() }, zone, targets[0]!, true);
  }
  const selectTarget: SelectGraphicsTarget = (pointer, context, candidates) => {
    const target = selected.get(key(pointer, context));
    if (!target || !candidates.some(candidate => candidate.room === target.room && candidate.upper === target.upper && candidate.index === target.index))
      throw new Error(`Animation target ${pointer} has no valid compiled attachment.`);
    return target;
  };
  return { tables: { ...tables, doors, lifts, water, switches }, selectTarget, addedLighting,
    graphics: { ...graphics, topology: { ...topology, points } } satisfies CellGraphics };
}
