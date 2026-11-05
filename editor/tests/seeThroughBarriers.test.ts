import { expect, it } from 'vitest';
import { closedPortalFixture } from './portalFixture';
import { roomPortalTargets, setRoomPortal } from '../src/model/roomPortals';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { applyCommand, execute, undo, redo, editRoomSource } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { encodeProject, decodeProject } from '../src/model/project';
import { roomSourceSnapshot, restoreRoomSource, insertSourceCorner } from '../src/model/roomSource';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { restorePortalStates } from '../src/model/portalStates';
import { setWallSettings } from '../src/model/wallSettings';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { translateSourceRooms } from '../src/model/roomSource';
import { createRoomOpening } from '../src/model/roomOpening';

function barrierFixture() {
  const fixture = closedPortalFixture(), document = fixture.baseline, a = fixture.owner.id, b = fixture.removed.id;
  const target = roomPortalTargets(document, a).find(target => target.neighbour === b)!;
  const appearances = [a, b].map(id => {
    const room = document.roomSource.rooms.find(room => room.id === id)!, choices = roomAppearanceChoices(document, room);
    return { room: id, lowerWall: choices.lower[0]!.offset, upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! };
  });
  const command = setRoomPortal(document, a, target.edges[0]!, b, 'barrier', appearances);
  return { ...fixture, document, a, b, target, command, barrier: applyCommand(document, command) };
}

it('authors reciprocal opcode-13 barriers with visible neighbours and solid native collision, saves and Undo', () => {
  const fixture = barrierFixture(), { a, b, document, command, barrier } = fixture;
  const history = execute({ document, past: [], future: [] }, command);
  expect(undo(history).document).toEqual(document); expect(redo(undo(history)).document.roomSource).toEqual(barrier.roomSource);
  expect(roomPortalTargets(barrier, a).find(target => target.neighbour === b)).toMatchObject({ open: true, barrier: true });
  const built = buildPlayableLevel(decodeProject(encodeProject(barrier)), -1);
  for (const [own, other] of [[a, b], [b, a]]) {
    const cell = built.topology!.cells.find(cell => cell.sourceId === own)!, destination = built.topology!.cells.find(cell => cell.sourceId === other)!;
    expect(cell.neighbours).toContain(destination.id);
    const zone = built.document.zones[cell.id]!;
    const edges = cell.neighbours.flatMap((id, edge) => id === destination.id ? [edge] : []);
    for (const edge of edges) expect(built.document.lines[zone.boundaryIds[edge]!]!.connect).toBe(-1);
    expect(zone.visibleZones).toContain(destination.id);
    expect(zone.lower.some(face => face.type === 13)).toBe(true);
  }
  expect(roomSourceSnapshot(barrier.roomSource).version).toBe(21);
  expect(() => restoreRoomSource(barrier, { ...roomSourceSnapshot(barrier.roomSource), version: 20 })).toThrow(/version 21/);
  const opened = applyCommand(barrier, setRoomPortal(barrier, a, fixture.target.edges[0]!, b, true));
  expect(roomPortalTargets(opened, a).find(target => target.neighbour === b)).toMatchObject({ open: true, barrier: false });
  const native = buildPlayableLevel(opened, -1), cell = native.topology!.cells.find(cell => cell.sourceId === a)!, destination = native.topology!.cells.find(cell => cell.sourceId === b)!;
  expect(native.document.zones[cell.id]!.boundaryIds.some(id => native.document.lines[id]!.connect === destination.id)).toBe(true);
});

it('retains barrier identity and wall settings through source-edge splitting and captured paste after Undo', () => {
  const fixture = barrierFixture(), { a, b, barrier } = fixture;
  const painted = applyCommand(barrier, editRoomSource(setWallSettings(barrier, { room: a, index: fixture.target.edges[0]!, upper: false }, { brightness: -3, repeat: { width: 64, fit: 'up' } })));
  const room = painted.roomSource.rooms.find(room => room.id === a)!, index = fixture.target.edges[0]!, start = room.vertices[index]!, end = room.vertices[(index + 1) % room.vertices.length]!;
  const midpoint = { x: (start.x + end.x) / 2, z: (start.z + end.z) / 2 };
  const natural = { ...barrier, roomSource: insertSourceCorner(barrier.roomSource, a, index, midpoint).after };
  const naturalBuild = buildPlayableLevel(natural, -1), naturalCell = naturalBuild.topology!.cells.find(cell => cell.sourceId === a)!;
  const naturalFaces = naturalBuild.document.zones[naturalCell.id]!.lower.filter(face => face.type === 13);
  expect(naturalFaces).toHaveLength(2); expect(naturalFaces[0]!.wall!.stripEnd).toBe(naturalFaces[1]!.wall!.stripStart);
  const split = { ...painted, roomSource: insertSourceCorner(painted.roomSource, a, index, { x: (start.x + end.x) / 2, z: (start.z + end.z) / 2 }).after };
  const built = buildPlayableLevel(split, -1), cell = built.topology!.cells.find(cell => cell.sourceId === a)!;
  const faces = built.document.zones[cell.id]!.lower.filter(face => face.type === 13 && face.wall?.brightness === -3);
  expect(faces.length).toBeGreaterThan(1);
  expect(faces.every(face => face.wall!.horizontalMask === 63)).toBe(true);
  expect(faces[0]!.wall!.stripEnd).toBe(faces[1]!.wall!.stripStart);
  const clipboard = copyRooms(painted, [a, b]), pasted = pasteRooms(fixture.document, clipboard, 20000, 20000), copied = applyCommand(fixture.document, pasted.command);
  expect(roomPortalTargets(copied, pasted.ids[0]!).find(target => target.neighbour === pasted.ids[1])).toMatchObject({ barrier: true });
  expect(roomSourceSnapshot(copied.roomSource).version).toBe(21);
  expect(() => buildPlayableLevel(decodeProject(encodeProject(copied)), -1)).not.toThrow();
});

it('rejects malformed or one-sided barrier declarations', () => {
  expect(() => restorePortalStates([{ boundary: 0, neighbour: 'room-1', open: false, barrier: true }], 4)).toThrow();
  expect(() => restorePortalStates([{ boundary: 0, neighbour: 'room-1', open: true, barrier: false }], 4)).toThrow();
  const fixture = barrierFixture(), room = fixture.barrier.roomSource.rooms.find(room => room.id === fixture.a)!;
  const broken = { ...fixture.barrier, roomSource: { ...fixture.barrier.roomSource, rooms: fixture.barrier.roomSource.rooms.map(item => item.id === room.id ? { ...item, portals: item.portals!.map(({ barrier: _barrier, ...state }) => state) } : item) } };
  expect(() => buildPlayableLevel(broken, -1)).toThrow(/disagree.*barrier/);
});

it('draws both storeys and retains reciprocal barriers after a skylight partitions their room', () => {
  const fixture = roomHeightFixture(twoStoreyHeightTemplate());
  const document = { ...fixture.document, roomSource: translateSourceRooms(fixture.document.roomSource, [fixture.to.id], -744, 0).after };
  const target = roomPortalTargets(document, fixture.from.id).find(target => target.neighbour === fixture.to.id)!;
  const barrier = applyCommand(document, setRoomPortal(document, fixture.from.id, target.edges[0]!, fixture.to.id, 'barrier'));
  const built = buildPlayableLevel(barrier, -1), cell = built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!;
  expect(built.document.zones[cell.id]!.lower.some(face => face.type === 13)).toBe(true);
  expect(built.document.zones[cell.id]!.upper.some(face => face.type === 13)).toBe(true);
  const proposal = createRoomOpening(barrier, fixture.from.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 }, barrier.roomAppearances![0]!, 'sky');
  const opened = applyCommand(barrier, proposal.command);
  expect(opened.roomSource.rooms.some(room => room.portals?.some(state => state.barrier && state.neighbour === fixture.to.id))).toBe(true);
  expect(() => buildPlayableLevel(decodeProject(encodeProject(opened)), -1)).not.toThrow();
});
