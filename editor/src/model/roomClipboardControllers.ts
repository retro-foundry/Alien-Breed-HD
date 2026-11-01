import { sourceMovingSettings, setSourceMovingRoom } from './sourceMovingRoom';
import type { LevelDocument } from './level';
import { importedAnimationTables } from './animationBuild';
import { mechanismDependencies, mechanismsForRoom } from './mechanismDependencies';
import { compileControllerCopies, copiedMechanismSettings, setRoomControllerCopies, type MovingControllerCopy } from './movingControllerCopies';
import { compileSwitchCopies, setRoomSwitchCopies, remapSwitchConditions, type SwitchCopy } from './switchCopies';
import { applyCommand, editRoomSource } from './commands';

/** Rebind explicit source relationships before allocating runtime tables. Keys
 * in native attachment maps retain provenance; their source room values change. */
export function copyRoomControllers(destination: LevelDocument, source: LevelDocument, rooms: ReadonlyMap<string, string>): LevelDocument {
  const tables = importedAnimationTables(source).tables;
  const capturedMotions = compileControllerCopies(source.roomSource.controllerCopies ?? []);
  const capturedSwitches = compileSwitchCopies(source.roomSource.switchCopies ?? []);
  const mapRoom = (id: string | undefined, subject: string) => {
    const mapped = id === undefined ? undefined : rooms.get(id);
    if (!mapped) throw new Error(`${subject} needs adjoining room ${id ?? '(missing original room)'}. Include that room in the copied group. No rooms were pasted.`);
    return mapped;
  };
  const originalRoom = (zone: number) => source.roomSource.rooms.find(room => room.originZone === zone)?.id;
  const motions: MovingControllerCopy[] = [], switches: { copy: SwitchCopy; previousSlot: number }[] = [];
  for (const [id, mapped] of rooms) {
    const room = source.roomSource.rooms.find(room => room.id === id)!;
    if (room.originZone !== null) {
      for (const mechanism of mechanismsForRoom(source, room.originZone)) {
        const dependencies = mechanismDependencies(source, mechanism.kind, mechanism.id);
        const record = tables[mechanism.kind === 'door' ? 'doors' : 'lifts']![mechanism.id]!;
        motions.push({ kind: mechanism.kind, id: mechanism.id, room: mapped, header: record.header.slice(),
          graphicsRooms: new Map(dependencies.graphicsRooms.map(zone => [zone, mapRoom(originalRoom(zone), `${mechanism.kind} graphics`)])),
          collisionRooms: new Map(dependencies.walls.map(wall => {
            const candidates = wall.collisionRooms.map(originalRoom).filter((room): room is string => !!room && rooms.has(room));
            if (candidates.length !== 1) throw new Error(`Moving collision line ${wall.line} needs one copied activation room. Include its original attachment room before pasting.`);
            return [wall.line, mapRoom(candidates[0], `${mechanism.kind} collision`)];
          })) });
      }
      for (const item of source.switches.filter(item => item.zone === room.originZone)) {
        const owner = source.zones.find(zone => [...zone.lower, ...zone.upper].some(record => record.offset === item.graphic - 10))!;
        switches.push({ previousSlot: item.id, copy: { id: item.id, slot: -1, room: mapped,
          graphicsRoom: mapRoom(originalRoom(owner.id), 'Switch graphics'), bytes: tables.switches![item.id]!.bytes.slice() } });
      }
    }
    for (const copy of capturedMotions.filter(copy => copy.room === id)) motions.push({ ...copy, room: mapped,
      graphicsRooms: new Map([...copy.graphicsRooms].map(([zone, id]) => [zone, mapRoom(id, 'Copied moving graphics')])),
      collisionRooms: new Map([...copy.collisionRooms].map(([line, id]) => [line, mapRoom(id, 'Copied moving collision')])) });
    for (const copy of capturedSwitches.filter(copy => copy.room === id)) switches.push({ previousSlot: copy.slot,
      copy: { ...copy, room: mapped, graphicsRoom: mapRoom(copy.graphicsRoom, 'Copied switch graphics') } });
  }
  const occupied = new Set(destination.roomSource.switchCopies?.map(copy => copy.slot));
  const internalMask = switches.reduce((mask, item) => mask | (1 << (item.previousSlot + 4)), 0);
  const mappedRooms = new Set(rooms.values());
  const claimed = [...destination.mechanisms.map(item => item.conditions),
    ...destination.roomSource.rooms.flatMap(room => room.moving ? [sourceMovingSettings(room.moving).conditions & ~(mappedRooms.has(room.id) ? internalMask : 0)] : []),
    ...(destination.roomSource.controllerCopies ?? []).map(copy => copiedMechanismSettings(copy).conditions),
    ...motions.map(copy => new DataView(copy.header!.buffer, copy.header!.byteOffset, copy.header!.byteLength).getUint16(14) & ~internalMask)];
  const free = destination.switches.filter(item => item.zone < 0 && !occupied.has(item.id) && !claimed.some(mask => mask & (1 << (item.id + 4)))).map(item => item.id);
  if (switches.length > free.length) throw new Error(`The copied group needs ${switches.length} switch slots, but only ${free.length} are unclaimed. Active switches and existing requirements reserve condition numbers. The native limit is eight switches. No rooms were pasted.`);
  const slots = new Map<number, number>();
  switches.forEach((item, index) => { item.copy.slot = free[index]!; slots.set(item.previousSlot, item.copy.slot); });
  for (const motion of motions) {
    const header = motion.header!, view = new DataView(header.buffer, header.byteOffset, header.byteLength);
    view.setUint16(14, remapSwitchConditions(view.getUint16(14), slots));
  }
  let working = destination;
  if (motions.length) working = applyCommand(working, editRoomSource(setRoomControllerCopies(working,
    [...compileControllerCopies(working.roomSource.controllerCopies ?? []), ...motions])));
  if (switches.length) working = applyCommand(working, editRoomSource(setRoomSwitchCopies(working,
    [...compileSwitchCopies(working.roomSource.switchCopies ?? []), ...switches.map(item => item.copy)])));
  for (const mapped of rooms.values()) {
    const room = working.roomSource.rooms.find(room => room.id === mapped)!;
    if (!room.moving) continue;
    const header = new Uint8Array(room.moving.header), view = new DataView(header.buffer);
    view.setUint16(14, remapSwitchConditions(view.getUint16(14), slots));
    working = applyCommand(working, editRoomSource(setSourceMovingRoom(working, mapped, { ...room.moving, header: [...header] })));
  }
  return working;
}
