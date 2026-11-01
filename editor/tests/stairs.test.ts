import { expect, it } from 'vitest';
import { execute, redo, undo } from '../src/model/commands';
import { createStairFlight, playerStepUpWorld } from '../src/model/stairs';
import { decodeProject, encodeProject } from '../src/model/project';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { compileRoomTopology } from '../src/model/roomTopology';
import { signedArea, validateRoomInteriors } from '../src/model/geometry';
import { stairFixture } from './stairFixture';

it('builds adjoining native steps with independent height planes, reciprocal portals and one undoable source command', () => {
  const { document, approach, flight } = stairFixture(), files = structuredClone(document.files);
  const proposal = createStairFlight(document, flight);
  expect(proposal.rises).toEqual([16, 16, 16, 16]);
  expect(proposal.rooms.map(room => room.heights?.floor)).toEqual([-16, -32, -48, -64]);
  expect(proposal.rooms.every(room => signedArea(room.vertices) < 0)).toBe(true);
  let history = execute({ document, past: [], future: [] }, proposal.command);
  expect(history.past).toHaveLength(1);
  const saved = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(saved, -1);
  const sourceIds = [approach.id, ...proposal.rooms.map(room => room.id)];
  const zones = sourceIds.map(id => built.document.zones[built.topology!.cells.find(cell => cell.sourceId === id)!.id]!);
  expect(zones.map(zone => zone.floor)).toEqual([0, -16, -32, -48, -64]);
  expect(zones.map(zone => zone.ceiling)).toEqual([-128, -144, -160, -176, -192]);
  for (let index = 1; index < zones.length; index++) {
    const a = zones[index - 1]!, b = zones[index]!;
    expect(a.boundaryIds.some(id => built.document.lines[id]!.connect === b.id)).toBe(true);
    expect(b.boundaryIds.some(id => built.document.lines[id]!.connect === a.id)).toBe(true);
  }
  expect(saved.files).toEqual(files);
  expect(saved.roomSource.rooms.slice(-4)).toEqual(proposal.rooms);
  history = undo(history);
  expect(history.document.roomSource).toBe(document.roomSource);
  expect(history.document.roomAppearances).toBe(document.roomAppearances);
  history = redo(history);
  expect(buildPlayableLevel(history.document, -1).document.files).toEqual(built.document.files);
});

it('snaps oblique and descending flights once per shared boundary without asking for winding', () => {
  const { document, flight } = stairFixture();
  for (const end of [{ x: 22300, z: 22400 }, { x: 21700, z: 22400 }]) {
    const proposal = createStairFlight(document, { ...flight, start: { x: 22000, z: 22000 }, end, width: 100, steps: 7, endFloor: 17.25 });
    const edges = validateRoomInteriors(proposal.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
    expect(edges.filter(edge => edge.neighbour !== null)).toHaveLength(12);
    expect(proposal.rooms.at(-1)!.heights?.floor).toBe(17.25);
    expect(proposal.rises.every(rise => rise < 0 && Math.abs(rise) < playerStepUpWorld)).toBe(true);
    expect(proposal.rooms.flatMap(room => room.vertices).every(p => Number.isInteger(p.x) && Number.isInteger(p.z))).toBe(true);
    expect(proposal.rooms.every(room => Number.isInteger(room.heights!.floor * 4))).toBe(true);
  }
});

it('rejects oversized risers, snapped flat steps, collapsed geometry and missing materials without partial edits', () => {
  const { document, flight } = stairFixture(), before = structuredClone(document);
  for (const change of [
    { endFloor: -playerStepUpWorld * flight.steps }, { endFloor: playerStepUpWorld * flight.steps },
    { endFloor: -0.25 }, { startFloor: 0.1 }, { clearance: 0 }, { clearance: 8192 }, { width: 0.01 },
    { end: flight.start }, { end: { x: 20064.5, z: 20200 } }, { width: 65536 },
    { steps: 1.5 }, { recordZone: -1 }, { appearance: { ...flight.appearance, lowerWall: 16777215 } },
    { start: { x: 20064, z: 20000 }, end: { x: 20064, z: 20128 } },
  ]) expect(() => createStairFlight(document, { ...flight, ...change })).toThrow();
  expect(document).toEqual(before);
});

it('checks the existing convex runtime cell budget before constructing a large flight', () => {
  const { document, flight } = stairFixture();
  const occupied = compileRoomTopology(document, document.roomSource).cells.length;
  expect(() => createStairFlight(document, { ...flight, steps: 257 - occupied })).toThrow(/runtime room slots remain/);
});
