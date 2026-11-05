import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { RoomChange, SourceRoom } from './roomSource';
import type { MechanismOwner } from './mechanismSettingsTransfer';
import { mechanismDependencies } from './mechanismDependencies';
import { activeNativeMechanisms, validateRetiredMechanisms } from './nativeMechanisms';
import { buildPlayableLevel } from './playableBuild';
import { resolveAnimationRoom } from './animationRooms';
import { roomHeightFields, roomSettings } from './roomHeights';
import { roomTemplate, type StaticMechanismPose } from './staticMechanismPose';
import { restoreWallSettings, type WallSettings } from './wallSettings';
import { restoreSourceMovingRoom, setSourceMovingRoom } from './sourceMovingRoom';

export type ImportedMechanismOwner = Exclude<MechanismOwner, { type: 'authored' }>;
function controller(document: LevelDocument, owner: ImportedMechanismOwner) {
  const copy = owner.type === 'copy' ? document.roomSource.controllerCopies?.find(item => item.room === owner.room && item.kind === owner.kind) : undefined;
  const id = owner.type === 'native' ? owner.id : copy?.id;
  const native = document.mechanisms.find(item => item.kind === owner.kind && item.id === id);
  if (!native || owner.type === 'copy' && !copy || owner.type === 'native' && !activeNativeMechanisms(document).includes(native)) throw new Error('That original or copied controller no longer exists.');
  const dependencies = mechanismDependencies(document, native.kind, native.id);
  const originalRoom = (zone: number) => document.roomSource.rooms.find(room => room.originZone === zone)?.id;
  const room = copy?.room ?? originalRoom(dependencies.room);
  if (!room) throw new Error('The controller has lost its source room.');
  const graphics = (zone: number) => copy ? copy.graphicsRooms.find(pair => pair[0] === zone)?.[1] : originalRoom(zone);
  const collision = (line: number, zone: number) => copy ? copy.collisionRooms.find(pair => pair[0] === line)?.[1] : originalRoom(zone);
  const header = new Uint8Array(copy?.header ?? document.files.graphics.slice(native.offset, native.offset + 18));
  return { native, dependencies, room, graphics, collision, header };
}
export function retiringMechanismSettings(document: LevelDocument, owner: ImportedMechanismOwner) {
  const item = controller(document, owner), reader = new BinaryReader(item.header, 'controller retirement');
  return { room: item.room, top: reader.i16(2) / 4, bottom: reader.i16(0) / 4, current: reader.i16(4) / 4 };
}
/** Remove animation ownership, preserving a separately editable static pose.
 * All graphic and collision owners change in the same source transaction.
 * Native templates and independently copied controllers remain intact. */
export function retireMechanism(document: LevelDocument, owner: ImportedMechanismOwner, height: number): RoomChange {
  const item = controller(document, owner), { native, dependencies } = item;
  const reader = new BinaryReader(item.header, 'controller retirement');
  checkedInteger(height * 4, -32768, 32767, 'Static controller height in quarters');
  if (height < reader.i16(2) / 4 || height > reader.i16(0) / 4) throw new Error('The static height must be between this controller’s upper and lower destinations.');
  const owners = new Set<string>([item.room]);
  for (const zone of dependencies.graphicsRooms) {
    const id = item.graphics(zone); if (!id) throw new Error('The controller has a missing graphics owner.'); owners.add(id);
  }
  for (const wall of dependencies.walls) for (const zone of wall.collisionRooms) {
    const id = item.collision(wall.line, zone); if (!id) throw new Error('The controller has a missing collision owner.'); owners.add(id);
  }
  // One wall cannot become static while another live controller still writes it.
  const others: ImportedMechanismOwner[] = [
    ...activeNativeMechanisms(document).map(control => ({ type: 'native' as const, kind: control.kind, id: control.id })),
    ...(document.roomSource.controllerCopies ?? []).map(copy => ({ type: 'copy' as const, kind: copy.kind, room: copy.room })),
  ];
  for (const otherOwner of others) {
    if (otherOwner.type === owner.type && otherOwner.kind === owner.kind && (otherOwner.type === 'native' && owner.type === 'native' ? otherOwner.id === owner.id : otherOwner.type === 'copy' && owner.type === 'copy' && otherOwner.room === owner.room)) continue;
    const other = controller(document, otherOwner);
    if (other.room === item.room && otherOwner.kind === owner.kind || other.dependencies.walls.some(wall => dependencies.walls.some(own => own.graphic === wall.graphic && item.graphics(own.room) === other.graphics(wall.room)))) throw new Error('These controllers share a moving plane or wall. Reassign their synchronized group together.');
  }
  const built = buildPlayableLevel(document, -1), controlledZone = built.topology?.cells.find(cell => cell.sourceId === item.room)?.id ?? dependencies.room;
  const matches = built.document.mechanisms.filter(control => control.kind === native.kind && resolveAnimationRoom(built.document, control.zone, native.kind === 'lift') === controlledZone);
  if (matches.length !== 1) throw new Error('Controller retirement needs one unambiguous compiled moving room.');
  const effective = matches[0]!;
  const phase = (graphic: number, sourceRoom: string) => {
    const entries = effective.walls.filter(wall => {
      if (!built.topology) return wall.graphic === graphic;
      const identity = built.surfaceSources?.find(record => record.offset === wall.graphic);
      return identity?.sourceOffset === graphic && built.topology.cells.find(cell => cell.id === identity.room)?.sourceId === sourceRoom;
    });
    if (!entries.length || entries.some(entry => entry.base !== entries[0]!.base)) throw new Error('A moving wall has missing or inconsistent compiled texture alignment.');
    const base = entries[0]!.base >>> 0, signed = (word: number) => word << 16 >> 16;
    return { horizontalOffset: signed(native.kind === 'door' ? base >>> 12 : base >>> 16),
      verticalOffset: signed(native.kind === 'door' ? (base & 4095) + ((-Math.floor(height)) & 255) : base & 65535) };
  };
  const rooms = document.roomSource.rooms.map((room): SourceRoom => {
    if (!owners.has(room.id)) return room;
    const walls = dependencies.walls.filter(wall => item.graphics(wall.room) === room.id).map(wall => ({ record: wall.graphic, ...phase(wall.graphic, room.id) }));
    const pose: StaticMechanismPose = { kind: native.kind, id: native.id, height, plane: room.id === item.room,
      surface: item.graphics(dependencies.surface.room) === room.id ? dependencies.surface.graphic : null, walls };
    const staticMechanisms = Object.freeze([...(room.staticMechanisms ?? []), Object.freeze(pose)]);
    const projected = roomTemplate(document, { ...room, staticMechanisms });
    const previous = roomSettings(document, room), heights = Object.fromEntries(roomHeightFields.map(field => [field, pose.plane && field === (native.kind === 'door' ? 'ceiling' : 'floor') ? height : previous[field]])) as unknown as NonNullable<SourceRoom['heights']>;
    if (heights.floor < heights.ceiling || projected.upper.length && heights.upperFloor > heights.upperCeiling && heights.upperFloor > heights.ceiling) throw new Error('This static height conflicts with the room’s lower or upper air space.');
    const { heights: _oldHeights, wallSettings: _oldSettings, ...retained } = room;
    // The captured phase belongs to the static baseline. Retain independent
    // brightness/repeat edits, but do not create alignment overrides for walls
    // which may disappear when the now-static room's heights change.
    const wallSettings = restoreWallSettings((room.wallSettings ?? []).flatMap<WallSettings>(settings => {
      if (!('record' in settings) || !walls.some(wall => wall.record === settings.record)) return [settings];
      const { horizontalOffset: _horizontal, verticalOffset: _vertical, ...retained } = settings;
      return retained.brightness !== undefined || retained.repeat ? [retained] : [];
    }));
    return Object.freeze({ ...retained, staticMechanisms, ...(roomHeightFields.some(field => heights[field] !== projected[field]) ? { heights: Object.freeze(heights) } : {}), ...(wallSettings.length ? { wallSettings } : {}) });
  });
  const { controllerCopies: _oldCopies, ...retainedSource } = document.roomSource;
  const copies = (document.roomSource.controllerCopies ?? []).filter(copy => !(owner.type === 'copy' && copy.kind === owner.kind && copy.room === owner.room));
  const after = Object.freeze({ ...retainedSource, ...(copies.length ? { controllerCopies: Object.freeze(copies) } : {}), rooms: Object.freeze(rooms),
    ...(owner.type === 'native' ? { removedNativeMechanisms: Object.freeze([...(document.roomSource.removedNativeMechanisms ?? []), { kind: native.kind, id: native.id }]) }
      : {}) });
  validateRetiredMechanisms({ ...document, roomSource: after });
  buildPlayableLevel({ ...document, roomSource: after }, -1);
  return { label: `Remove ${owner.type === 'native' ? 'original' : 'copied'} ${native.kind}`, before: document.roomSource, after };
}

export function reassignImportedMechanism(document: LevelDocument, owner: ImportedMechanismOwner, target: string, height: number, wall: number): RoomChange {
  const item = controller(document, owner);
  if (target === item.room) throw new Error('Choose another existing room for the mechanism.');
  const destination = document.roomSource.rooms.find(room => room.id === target);
  if (!destination) throw new Error('Choose an existing destination room.');
  if (destination.moving) throw new Error('The destination already has an authored controller. Choose an empty static room.');
  const removed = retireMechanism(document, owner, height);
  const moving = restoreSourceMovingRoom({ kind: owner.kind, header: [...item.header], wall });
  const after = setSourceMovingRoom({ ...document, roomSource: removed.after }, target, moving).after;
  buildPlayableLevel({ ...document, roomSource: after }, -1);
  return { label: `Move ${owner.type === 'native' ? 'original' : 'copied'} ${owner.kind}`, before: document.roomSource, after };
}
