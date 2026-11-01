import stepUp from '../../../src/player_step_up.json' with { type: 'json' };
import { checkedInteger } from './binary';
import type { EditCommand } from './commands';
import type { Position } from './geometry';
import type { LevelDocument } from './level';
import { validateRoomAppearances, type RoomAppearance } from './roomAppearance';
import { addSourceRooms, type SourceRoom } from './roomSource';
import { compileRoomTopology } from './roomTopology';

if (!Number.isSafeInteger(stepUp) || stepUp <= 0) throw new Error('The native player step allowance is invalid.');
export const playerStepUpWorld = stepUp;
export interface StairFlight {
  start: Position; end: Position; width: number; steps: number;
  startFloor: number; endFloor: number; clearance: number;
  recordZone: number; appearance: Omit<RoomAppearance, 'room'>;
}
export interface StairProposal { command: EditCommand; rooms: readonly SourceRoom[]; rises: readonly number[] }

/** Generate flat native rooms, not replacement ramp physics. Shared sections
 * are snapped once for both neighbours. Validate immutable temporary source
 * before returning one stale-checked command; cancellation never edits source. */
export function createStairFlight(document: LevelDocument, flight: StairFlight): StairProposal {
  checkedInteger(flight.steps, 1, 256, 'Number of steps');
  for (const point of [flight.start, flight.end]) for (const coordinate of [point.x, point.z]) checkedInteger(coordinate, -32768, 32767, 'Stair endpoint');
  if (!Number.isFinite(flight.width) || flight.width <= 0) throw new Error('Stair width must be positive.');
  for (const value of [flight.startFloor, flight.endFloor, flight.clearance]) checkedInteger(value * 4, -32768, 32767, 'Stair height in quarter units');
  if (flight.clearance <= 0) throw new Error('Stair clearance must be positive.');
  if (flight.startFloor === flight.endFloor) throw new Error('The final step floor must differ from the floor before the flight.');
  const template = document.zones.find(zone => zone.id === flight.recordZone);
  if (!template) throw new Error('Choose an existing stair material template.');
  const occupied = document.roomSource.rooms.length ? compileRoomTopology(document, document.roomSource).cells.length : 0;
  if (occupied + flight.steps > 256) throw new Error(`The flight needs ${flight.steps} rooms; only ${256 - occupied} runtime room slots remain.`);
  const dx = flight.end.x - flight.start.x, dz = flight.end.z - flight.start.z, length = Math.hypot(dx, dz);
  if (length === 0) throw new Error('The stair endpoints must be different.');
  const side = { x: -dz / length * flight.width / 2, z: dx / length * flight.width / 2 };
  const sections = Array.from({ length: flight.steps + 1 }, (_, index) => {
    const x = flight.start.x + dx * index / flight.steps, z = flight.start.z + dz * index / flight.steps;
    return [{ x: Math.round(x + side.x), z: Math.round(z + side.z) }, { x: Math.round(x - side.x), z: Math.round(z - side.z) }] as const;
  });
  let previous = flight.startFloor;
  const drafts: Parameters<typeof addSourceRooms>[2][number][] = [], rises: number[] = [];
  for (let index = 0; index < flight.steps; index++) {
    const floor = Math.round((flight.startFloor + (flight.endFloor - flight.startFloor) * (index + 1) / flight.steps) * 4) / 4;
    const rise = previous - floor;
    // movement.c transition_height_ok_single uses a strict step-up bound.
    // Check both directions; clearance remains an explicit author choice.
    if (Math.abs(rise) >= stepUp) throw new Error(`Step ${index + 1} changes height by ${Math.abs(rise)}; native climbing needs risers below ${stepUp}. Increase the number of steps.`);
    if (rise === 0) throw new Error(`Step ${index + 1} has no rise after quarter-unit snapping. Reduce the number of steps.`);
    const a = sections[index]!, b = sections[index + 1]!;
    drafts.push({ recordZone: template.id, vertices: [a[0], a[1], b[1], b[0]], heights: { floor, ceiling: floor - flight.clearance,
      upperFloor: template.upperFloor, upperCeiling: template.upperCeiling } });
    rises.push(rise); previous = floor;
  }
  const source = addSourceRooms(document, document.roomSource, drafts).after;
  const rooms = source.rooms.slice(-flight.steps);
  const appearances = validateRoomAppearances({ ...document, roomSource: source }, [
    ...(document.roomAppearances ?? []), ...rooms.map(room => ({ ...flight.appearance, room: room.id })),
  ]);
  compileRoomTopology(document, source);
  return { rooms, rises, command: { label: `Create ${flight.steps} stair steps`, patches: [],
    roomSource: { before: document.roomSource, after: source }, roomAppearances: { before: document.roomAppearances, after: appearances } } };
}
