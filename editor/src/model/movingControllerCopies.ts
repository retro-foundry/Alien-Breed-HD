import type { LevelDocument } from './level';
import { resolveAnimationRoom } from './animationRooms';
import { nativeRoomTranslation } from './roomTranslation';
import { mechanismDependencies } from './mechanismDependencies';
import { BinaryReader, checkedInteger } from './binary';
import { assertMechanismActivation } from './mechanismActivation';
import { importedAnimationTables } from './animationBuild';
import type { RoomChange } from './roomSource';

function validateHeader(kind: 'door' | 'lift', header: Uint8Array): number {
  if (header.length !== 18) throw new Error('Copied controller settings require a complete 18-byte native header.');
  const reader = new BinaryReader(header, 'copied controller settings');
  const bottom = reader.i16(0) / 4, top = reader.i16(2) / 4, current = reader.i16(4) / 4;
  if (reader.i16(0) === 999 || top > bottom || current < top || current > bottom) throw new Error('Copied controller settings have an invalid native travel range.');
  if (reader.u8(16) > (kind === 'door' ? 5 : 3) || reader.u8(17) > (kind === 'door' ? 1 : 3)) throw new Error('Copied controller settings contain unsupported activation rules.');
  assertMechanismActivation({ kind, direction: reader.i16(6), conditions: reader.u16(14), current, top, bottom });
  return current;
}

export interface SourceControllerCopy {
  readonly kind: 'door' | 'lift';
  readonly id: number;
  readonly room: string;
  readonly graphicsRooms: readonly (readonly [number, string])[];
  readonly collisionRooms: readonly (readonly [number, string])[];
  readonly header: readonly number[];
}

/** Source snapshots contain frozen arrays rather than mutable Maps/typed arrays.
 * Geometry references may need repair after later edits, but damaged archive
 * data must never be silently defaulted or discarded. */
export function restoreControllerCopies(value: unknown): readonly SourceControllerCopy[] {
  if (!Array.isArray(value) || value.length > 512) throw new Error('Copied controller source must contain at most 512 records.');
  const identities = new Set<string>();
  const roomId = (value: unknown) => {
    if (typeof value !== 'string' || !value.length || value.length > 128) throw new Error('Copied controller room identity is invalid.');
    return value;
  };
  return Object.freeze(value.map(item => {
    if (!item || typeof item !== 'object' || Object.keys(item).length !== 6
      || Object.keys(item).some(key => !['kind', 'id', 'room', 'graphicsRooms', 'collisionRooms', 'header'].includes(key))) throw new Error('Copied controller source has missing or unsupported fields.');
    if (item.kind !== 'door' && item.kind !== 'lift') throw new Error('Copied controller kind must be door or lift.');
    const kind: 'door' | 'lift' = item.kind, id = checkedInteger(item.id, 0, 255, 'Native controller identity'), room = roomId(item.room);
    const identity = `${kind}:${room}`;
    if (identities.has(identity)) throw new Error('Copied controller source repeats a moving-plane controller.');
    identities.add(identity);
    const mapping = (value: unknown, maximum: number) => {
      if (!Array.isArray(value) || value.length > 256) throw new Error('Copied controller room mapping is invalid.');
      const keys = new Set<number>();
      return Object.freeze(value.map(pair => {
        if (!Array.isArray(pair) || pair.length !== 2) throw new Error('Copied controller mapping needs native identity and source room pairs.');
        const key = checkedInteger(pair[0], 0, maximum, 'Native attachment identity');
        if (keys.has(key)) throw new Error('Copied controller mapping repeats a native attachment identity.');
        keys.add(key); return Object.freeze([key, roomId(pair[1])] as const);
      }).sort((a, b) => a[0] - b[0]));
    };
    if (!Array.isArray(item.header) || item.header.length !== 18) throw new Error('Copied controller source needs every native settings byte.');
    const header = Object.freeze(item.header.map((byte: number) => checkedInteger(byte, 0, 255, 'Copied controller settings byte')));
    validateHeader(kind, new Uint8Array(header));
    return Object.freeze({ kind, id, room, graphicsRooms: mapping(item.graphicsRooms, 255), collisionRooms: mapping(item.collisionRooms, 32767), header });
  }));
}

export function compileControllerCopies(value: readonly SourceControllerCopy[]): readonly MovingControllerCopy[] {
  return value.map(copy => ({ ...copy, graphicsRooms: new Map(copy.graphicsRooms), collisionRooms: new Map(copy.collisionRooms), header: new Uint8Array(copy.header) }));
}

export function setRoomControllerCopies(document: LevelDocument, copies: readonly MovingControllerCopy[]): RoomChange {
  const tables = importedAnimationTables(document).tables;
  const captured = restoreControllerCopies(copies.map(copy => {
    const template = tables[copy.kind === 'door' ? 'doors' : 'lifts']?.[copy.id];
    if (!template) throw new Error('A copied controller has no native template.');
    return { ...copy, graphicsRooms: [...copy.graphicsRooms], collisionRooms: [...copy.collisionRooms], header: [...(copy.header ?? template.header)] };
  }));
  validateMovingControllerCopies(document, compileControllerCopies(captured));
  const before = document.roomSource;
  if (JSON.stringify(before.controllerCopies ?? []) === JSON.stringify(captured)) return { label: 'Change copied controllers', before, after: before };
  const { controllerCopies: _previous, ...unchanged } = before;
  return { label: 'Change copied controllers', before, after: Object.freeze({ ...unchanged, ...(captured.length ? { controllerCopies: captured } : {}) }) };
}

/** Explicit source identities, never generated zone/line offsets. Graphics
 * ownership and collision incidence are independent native relationships. */
export interface MovingControllerCopy {
  kind: 'door' | 'lift';
  id: number;
  room: string;
  graphicsRooms: ReadonlyMap<number, string>;
  collisionRooms: ReadonlyMap<number, string>;
  /** Optional complete native settings header. Pointer/room fields are rebuilt. */
  header?: Uint8Array;
}

export function validateMovingControllerCopies(document: LevelDocument, copies: readonly MovingControllerCopy[]): void {
  const assigned = new Set<string>();
  for (const copy of copies) {
    const mechanism = document.mechanisms.find(item => item.kind === copy.kind && item.id === copy.id);
    if (!mechanism) throw new Error('A copied controller has no native template.');
    const dependencies = mechanismDependencies(document, copy.kind, copy.id);
    const controlled = document.roomSource.rooms.find(room => room.id === copy.room);
    if (!controlled || controlled.originZone !== null || controlled.recordZone !== resolveAnimationRoom(document, mechanism.zone, copy.kind === 'lift'))
      throw new Error('A copied controller needs a fresh room identity matching its controlled native template.');
    const identity = `${copy.kind}:${copy.room}`;
    if (assigned.has(identity)) throw new Error('A copied room has several controllers for the same moving plane.');
    assigned.add(identity);
    const translation = nativeRoomTranslation(document, controlled);
    if (!translation) throw new Error('Copied moving controllers require an exact whole-room translation.');
    const checkRoom = (id: string | undefined, template: number | readonly number[], subject: string) => {
      const room = document.roomSource.rooms.find(room => room.id === id);
      const allowed = typeof template === 'number' ? [template] : template;
      if (!room || room.originZone !== null || !allowed.includes(room.recordZone)) throw new Error(`${subject} needs an explicit copied room with the matching native attachment.`);
      const offset = nativeRoomTranslation(document, room);
      if (!offset || offset.x !== translation.x || offset.z !== translation.z) throw new Error(`${subject} must move with the copied controller's room group.`);
    };
    for (const room of dependencies.graphicsRooms) checkRoom(copy.graphicsRooms.get(room), room, `Moving graphics from Zone ${room}`);
    if (copy.graphicsRooms.get(dependencies.room) !== copy.room) throw new Error('The controller room and graphics-room mapping disagree.');
    for (const wall of dependencies.walls) checkRoom(copy.collisionRooms.get(wall.line), wall.collisionRooms, `Moving collision line ${wall.line}`);
    if (copy.header) {
      const current = validateHeader(copy.kind, copy.header);
      if (current !== mechanism.current) throw new Error('Changing a copied controller initial position requires synchronized source geometry.');
    }
  }
}
