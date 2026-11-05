import { activeNativeSwitches } from './nativeSwitches';
import { resolveAnimationRoom } from './animationRooms';
import type { PlayableLevel } from './playableBuild';
import { resolveEngineSurface, type NativeSurfacePick } from './engineSurfacePick';
import { movingRoomConflict } from './movingRoomEligibility';
import type { RoomTopology } from './roomTopology';
import { BinaryReader, checkedInteger } from './binary';
import type { CellGraphics } from './cellGraphics';
import type { CollisionGeometry } from './collisionBuild';
import type { LevelDocument } from './level';
import type { MovingRoomAnimation, AnimationTables } from './animationBuild';
import { readGraphicRecord } from './graphics';
import type { GraphicsRecordInput } from './graphicsBuild';
import { assertMechanismRange } from './mechanismActivation';
import { roomSettings } from './roomHeights';
import { encodeWallSurface, type WallSurface } from './surfaces';
import { wallForSpan } from './surfaceGeometry';

/** Compiler input in stable source room identity, with explicitly selected
 * native settings and material. Project authoring/persistence owns these choices. */
export interface AuthoredMovingRoom {
  readonly room: string;
  readonly kind: 'door' | 'lift';
  readonly header: Uint8Array;
  readonly wall: { readonly surface: WallSurface; readonly length: number };
}

/** Editable air clearance stays valid even when a door initially closes it.
 * Only the generated topology gets the native initial moving-plane height. */
export function initialMovingTopology(document: LevelDocument, topology: RoomTopology, plans: readonly AuthoredMovingRoom[]): RoomTopology {
  if (!plans.length) return topology;
  return { ...topology, cells: topology.cells.map(cell => {
    const plan = plans.find(plan => plan.room === cell.sourceId);
    if (!plan) return cell;
    if (plan.header.length !== 18) throw new Error('An authored controller needs its complete 18-byte native settings header.');
    const settings = roomSettings(document, cell), current = new BinaryReader(plan.header, 'initial moving plane').i16(4) / 4;
    return { ...cell, heights: { floor: settings.floor, ceiling: settings.ceiling, upperFloor: settings.upperFloor, upperCeiling: settings.upperCeiling,
      [plan.kind === 'door' ? 'ceiling' : 'floor']: current } };
  }) };
}

/** Build static authored doorway geometry first. Initial moving heights are
 * then applied only to the controlled plane/header; the mechanism builder owns
 * its lintel/riser bands instead of asking every neighbour for wall repairs. */
export function initialMovingGraphics(document: LevelDocument, graphics: CellGraphics, plans: readonly AuthoredMovingRoom[]): CellGraphics {
  if (!plans.length) return graphics;
  const topology = initialMovingTopology(document, graphics.topology, plans), streams = new Map(graphics.streams);
  for (const cell of topology.cells) {
    const plan = plans.find(plan => plan.room === cell.sourceId);
    if (!plan) continue;
    const current = new BinaryReader(plan.header, 'initial moving plane').i16(4), stream = streams.get(cell.id)!;
    const lower = stream.lower.map(item => {
      const type = new BinaryReader(item.bytes, 'initial moving graphic').i16(0);
      if (!(plan.kind === 'door' ? [2, 9, 11] : [1, 8, 10]).includes(type)) return item;
      const bytes = item.bytes.slice(); new DataView(bytes.buffer).setInt16(2, current);
      return { ...item, bytes };
    });
    streams.set(cell.id, { ...stream, lower });
  }
  return { ...graphics, topology, streams };
}

/** Anims.s DoorRoutine/LiftRoutine and objects.c: roofs/wall bottoms move for
 * doors; floors/wall tops move for lifts. A portal needs both the neighbour's
 * moving band and static interior pit walls over the entire swept range. */
export function attachAuthoredMovingRoom(document: LevelDocument, graphics: CellGraphics, collision: CollisionGeometry, input: AuthoredMovingRoom): { graphics: CellGraphics; animation: MovingRoomAnimation } {
  const { topology } = graphics, cells = topology.cells.filter(cell => cell.sourceId === input.room);
  const source = document.roomSource.rooms.find(room => room.id === input.room);
  if (!source || cells.length !== 1) throw new Error('A door/lift needs one room without inward corners. Split moving volumes need explicit controller grouping.');
  const conflict = movingRoomConflict(document, source);
  if (conflict) throw new Error(conflict);
  if (input.header.length !== 18) throw new Error('An authored controller needs its complete 18-byte native settings header.');
  const cell = cells[0]!, settings = roomSettings(document, cell), baseline = roomSettings(document, source), header = new BinaryReader(input.header, 'authored controller');
  const top = header.i16(2) / 4, bottom = header.i16(0) / 4, current = header.i16(4) / 4;
  assertMechanismRange({ kind: input.kind, direction: header.i16(6), conditions: header.u16(14), current }, top, bottom);
  if (header.u8(16) > (input.kind === 'door' ? 5 : 3) || header.u8(17) > (input.kind === 'door' ? 1 : 3)) throw new Error('The controller has unsupported native trigger rules.');
  if (current !== settings[input.kind === 'door' ? 'ceiling' : 'floor']) throw new Error('The controller initial height must match its room plane. Apply matching room heights before building.');
  if (input.kind === 'lift' && top < settings.ceiling || input.kind === 'door' && (bottom > settings.floor || settings.upper.length > 0 && top < settings.upperFloor))
    throw new Error('The moving range crosses its room floor, ceiling or upper storey. Adjust its destinations before building.');
  const stream = graphics.streams.get(cell.id)!;
  const read = (item: GraphicsRecordInput) => readGraphicRecord(new BinaryReader(item.bytes, 'authored moving graphics'), 0, 2048);
  const planes = stream.lower.flatMap((item, index) => {
    const record = read(item);
    return (input.kind === 'door' ? [2, 9, 11] : [1, 8, 10]).includes(record.type) ? [{ record, index }] : [];
  });
  const pointKey = (id: number) => `${topology.points[id]!.x},${topology.points[id]!.z}`;
  const samePoint = (a: number, b: number) => pointKey(a) === pointKey(b);
  if (planes.length !== 1 || planes[0]!.record.polygon!.heightQuarter !== current * 4
    || new Set(planes[0]!.record.points.map(pointKey)).size !== cell.vertices.length || !cell.vertices.every(point => planes[0]!.record.points.some(other => samePoint(point, other))))
    throw new Error('A new moving room needs one complete lower floor/ceiling polygon. Partial surfaces need synchronized animation targets.');
  if (cell.neighbours.every(id => id < 0)) throw new Error('Connect the new door/lift room to another room before creating its moving wall attachments.');
  const streams = new Map(graphics.streams), references = new Map(graphics.pointReferences), links: { line: number; base: number; owner: number; item: GraphicsRecordInput }[] = [];
  const nativeTargets = new Set([...document.mechanisms.flatMap(item => item.walls.map(wall => wall.graphic)),
    ...activeNativeSwitches(document).map(item => item.graphic - 10)]);
  const update = (id: number, items: readonly GraphicsRecordInput[]) => {
    streams.set(id, { ...streams.get(id)!, lower: items });
    references.set(id, [...new Set([...(references.get(id) ?? []), ...items.flatMap(item => read(item).points)])]);
  };
  // Solid sides must still cover the air exposed by the moving plane.
  update(cell.id, stream.lower.map(item => {
    const record = read(item), wall = record.wall;
    if (!wall || !cell.vertices.some((point, edge) => cell.neighbours[edge]! < 0 && wall.points.some(other => samePoint(point, other)) && wall.points.some(other => samePoint(cell.vertices[(edge + 1) % cell.vertices.length]!, other)))) return item;
    return { ...item, bytes: encodeWallSurface({ ...wall, ...(input.kind === 'door' ? { topFixed: Math.min(wall.topFixed, top * 256) } : { bottomFixed: Math.max(wall.bottomFixed, bottom * 256) }) }) };
  }));
  const length = input.wall.length;
  const makeWall = (points: [number, number], topHeight: number, bottomHeight: number) => {
    const [a, b] = points.map(point => topology.points[point]!);
    return { ...wallForSpan(input.wall.surface, length, points, 0, Math.hypot(b!.x - a!.x, b!.z - a!.z)), topFixed: topHeight * 256, bottomFixed: bottomHeight * 256 };
  };
  const putBand = (owner: number, points: [number, number], oldTop: number, oldBottom: number, wall: WallSurface) => {
    const items = [...streams.get(owner)!.lower];
    const matches = items.flatMap((item, index) => {
      const candidate = read(item).wall;
      return candidate && samePoint(candidate.points[0], points[0]) && samePoint(candidate.points[1], points[1])
        && candidate.topFixed === oldTop * 256 && candidate.bottomFixed === oldBottom * 256 ? [index] : [];
    });
    if (matches.length > 1) throw new Error('Several decorative walls overlap a moving band. Assign their animation ownership before building.');
    const existing = matches.length ? items[matches[0]!] : undefined;
    const index = matches[0] ?? items.reduce((position, item, index) => [0, 5, 13].includes(read(item).type) ? index + 1 : position, 0);
    if (existing?.sourceOffset !== undefined && nativeTargets.has(existing.sourceOffset)) throw new Error('A new moving band overlaps an existing animated wall. Reassign that mechanism before building.');
    const { sourceOffset, ...metadata } = existing ?? {};
    const item = { ...metadata, ...(sourceOffset !== undefined && read(existing!).type === wall.type ? { sourceOffset } : {}), bytes: encodeWallSurface(wall) };
    if (existing) items[index] = item; else items.splice(index, 0, item);
    update(owner, items); return { owner, item };
  };
  cell.neighbours.forEach((id, edge) => {
    if (id < 0) return;
    const neighbourIndex = topology.cells.findIndex(target => target.id === id), neighbour = topology.cells[neighbourIndex]!;
    const adjacent = roomSettings(document, neighbour);
    const a = cell.vertices[edge]!, b = cell.vertices[(edge + 1) % cell.vertices.length]!;
    const opposite = neighbour.vertices.findIndex((point, index) => point === b && neighbour.vertices[(index + 1) % neighbour.vertices.length] === a);
    if (opposite < 0 || neighbour.neighbours[opposite] !== cell.id) throw new Error('A moving room portal needs matching reverse collision boundaries.');
    if (top < adjacent.ceiling || input.kind === 'door' && bottom > adjacent.floor) throw new Error('The moving range crosses the adjoining room ceiling or door floor. Adjust its destinations before building.');
    const band = input.kind === 'door' ? makeWall([b, a], adjacent.ceiling, current) : makeWall([b, a], current, adjacent.floor);
    const target = putBand(neighbour.id, [b, a], input.kind === 'door' ? adjacent.ceiling : baseline.floor, input.kind === 'door' ? baseline.ceiling : adjacent.floor, band);
    // door_wall_texture_words decodes a legacy 4096-byte tile base and applies
    // -(position >> 8) & 255. Invert that scroll modulo 256 so the selected
    // authored row phase stays aligned at the initial position (mask <= 255).
    const tile = checkedInteger(band.horizontalOffset, 0, 32767, 'Moving wall tile');
    // renderer_lift_wall_texture_info_for_gfx_off instead consumes packed
    // fromtile/yoff words directly, without the door's 4096-byte decoding.
    const base = input.kind === 'door' ? tile * 4096 + ((band.verticalOffset + Math.floor(current)) & 255)
      : tile * 65536 + (band.verticalOffset & 65535);
    const ownIndex = topology.cells.indexOf(cell);
    for (const line of [collision.boundaries[ownIndex]![edge]!, collision.boundaries[neighbourIndex]![opposite]!])
      links.push({ line, base, ...target });
    if (input.kind === 'lift' && bottom > adjacent.floor) {
      const pit = makeWall([a, b], adjacent.floor, bottom);
      putBand(cell.id, [a, b], adjacent.floor, baseline.floor, pit);
    }
  });
  cell.neighbours.forEach((neighbour, edge) => {
    if (neighbour >= 0) return;
    const points: [number, number] = [cell.vertices[edge]!, cell.vertices[(edge + 1) % cell.vertices.length]!];
    if (streams.get(cell.id)!.lower.some(item => { const wall = read(item).wall; return wall && samePoint(wall.points[0], points[0]) && samePoint(wall.points[1], points[1]); })) return;
    const wall = makeWall(points, input.kind === 'door' ? top : settings.ceiling, input.kind === 'lift' ? bottom : settings.floor);
    putBand(cell.id, points, current, current, wall);
  });
  checkedInteger(links.length, 1, 64, 'Authored moving wall count');
  const walls = links.map(({ line, base, owner, item }) => ({ line, base, graphic: 0, target: { room: owner, upper: false, index: streams.get(owner)!.lower.indexOf(item) } }));
  return { graphics: { ...graphics, streams, pointReferences: references }, animation: { header: input.header.slice(), zone: cell.id, graphic: 0,
    target: { room: cell.id, upper: false, index: streams.get(cell.id)!.lower.indexOf(stream.lower[planes[0]!.index]!) }, walls } };
}


/** Insert all new bands before resolving final animation record identities. */
export function buildAuthoredMovingRooms(document: LevelDocument, graphics: CellGraphics, collision: CollisionGeometry, plans: readonly AuthoredMovingRoom[]) {
  const topology = graphics.topology;
  let movingGraphics = graphics;
  const authored: { kind: 'door' | 'lift'; animation: MovingRoomAnimation }[] = [];
  const occupied = new Set(plans.map(plan => plan.room));
  for (const control of document.mechanisms) {
    const origin = resolveAnimationRoom(document, control.zone, control.kind === 'lift');
    for (const cell of topology.cells) if (cell.originZone === origin) occupied.add(cell.sourceId);
  }
  for (const copy of document.roomSource.controllerCopies ?? []) occupied.add(copy.room);
  if (new Set(plans.map(plan => plan.room)).size !== plans.length) throw new Error('Each authored moving room needs one controller.');
  for (const plan of plans) {
    const cell = topology.cells.find(cell => cell.sourceId === plan.room);
    if (cell?.neighbours.some(id => topology.cells.some(target => target.id === id && target.sourceId !== plan.room && occupied.has(target.sourceId))))
      throw new Error('Adjacent independently moving rooms need explicit synchronized controller grouping.');
    const before = movingGraphics, built = attachAuthoredMovingRoom(document, before, collision, plan);
    // A new band enters the existing wall batch. Rebind earlier exact targets
    // by record identity after insertion, preserving stream draw order.
    const rebind = (target: NonNullable<MovingRoomAnimation['target']>) => {
      const old = before.streams.get(target.room)!.lower[target.index]!;
      const index = built.graphics.streams.get(target.room)!.lower.indexOf(old);
      if (index < 0) throw new Error('Authored moving controllers share incompatible graphics ownership.');
      return { ...target, index };
    };
    authored.forEach((item, index) => { authored[index] = { ...item, animation: { ...item.animation,
      target: rebind(item.animation.target!), walls: item.animation.walls.map(wall => ({ ...wall, target: rebind(wall.target!) })) } }; });
    movingGraphics = built.graphics; authored.push({ kind: plan.kind, animation: built.animation });
  }
  return { graphics: movingGraphics, authored };
}

export function appendAuthoredMovingAnimations(tables: AnimationTables, authored: readonly { kind: 'door' | 'lift'; animation: MovingRoomAnimation }[]): AnimationTables {
  if (!authored.length) return tables;
  const doors = authored.filter(item => item.kind === 'door').map(item => item.animation);
  const lifts = authored.filter(item => item.kind === 'lift').map(item => item.animation);
  return { ...tables, doors: doors.length ? [...(tables.doors ?? []), ...doors] : tables.doors,
    lifts: lifts.length ? [...(tables.lifts ?? []), ...lifts] : tables.lifts,
    // DoWaterAnims scans 21 groups after a lift terminator. Empty structural
    // groups have no plane/zone entries, so they cause no level writes.
    water: lifts.length && !tables.water.length ? Array.from({ length: 21 }, () => ({ header: new Uint8Array(14), entries: [] })) : tables.water };
}


/** Native animation tables identify the actual generated moving face. Resolve
 * that ownership back to its source controller instead of painting a neighbour
 * whose rebuilt material would immediately be replaced by the controller. */
export function authoredMovingSurfaceOwner(built: PlayableLevel, zone: number, pick: NativeSurfacePick): { room: string; kind: 'door' | 'lift' } | undefined {
  if (!built.topology || !resolveEngineSurface(built.document, zone, pick)) return;
  const reader = new BinaryReader(built.document.files.graphics, 'moving surface ownership');
  for (const controller of built.document.mechanisms) {
    if (!controller.walls.some(wall => wall.graphic === pick.offset) && reader.i32(controller.offset + 8) !== pick.offset) continue;
    const id = built.topology.cells.find(cell => cell.id === controller.zone)?.sourceId;
    const room = built.source.roomSource.rooms.find(room => room.id === id && room.moving?.kind === controller.kind);
    if (room) return Object.freeze({ room: room.id, kind: controller.kind });
  }
}
