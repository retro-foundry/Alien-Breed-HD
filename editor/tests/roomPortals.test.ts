import { expect, it } from 'vitest';
import { closedPortalFixture } from './portalFixture';
import { roomPortalTargets, setRoomPortal } from '../src/model/roomPortals';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { applyCommand } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { roomHeightFixture } from './roomHeightFixture';
import { translateSourceRooms, insertSourceCorner } from '../src/model/roomSource';
import { createRoomOpening } from '../src/model/roomOpening';

it('closes and reopens an existing native connection reciprocally, with history, saves and independent copies', () => {
  const fixture = closedPortalFixture(), document = fixture.baseline, a = fixture.owner.id, b = fixture.removed.id;
  const target = roomPortalTargets(document, a).find(target => target.neighbour === b)!;
  expect(target.open).toBe(true);
  const appearances = [a, b].map(id => {
    const room = document.roomSource.rooms.find(room => room.id === id)!, choices = roomAppearanceChoices(document, room);
    return { room: id, lowerWall: choices.lower[0]!.offset, upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! };
  });
  const command = setRoomPortal(document, a, target.edges[0]!, b, false, appearances), closed = applyCommand(document, command);
  expect(roomPortalTargets(closed, a).find(target => target.neighbour === b)!.open).toBe(false);
  expect(roomPortalTargets(closed, b).find(target => target.neighbour === a)!.open).toBe(false);
  const blocked = buildPlayableLevel(closed, -1);
  const own = blocked.topology!.cells.find(cell => cell.sourceId === a)!, other = blocked.topology!.cells.find(cell => cell.sourceId === b)!;
  expect(own.neighbours).not.toContain(other.id); expect(other.neighbours).not.toContain(own.id);
  expect(decodeProject(encodeProject(closed)).roomSource).toEqual(closed.roomSource);
  expect(applyCommand(closed, command, true)).toEqual(document);
  const reopened = applyCommand(closed, setRoomPortal(closed, a, target.edges[0]!, b, true));
  expect(roomPortalTargets(reopened, a).find(target => target.neighbour === b)!.open).toBe(true);
  const copied = pasteRooms(closed, copyRooms(closed, [a, b]), 20000, 20000), pasted = applyCommand(closed, copied.command);
  expect(roomPortalTargets(pasted, copied.ids[0]!).find(target => target.neighbour === copied.ids[1])!.open).toBe(false);
  expect(decodeProject(encodeProject(pasted)).roomSource).toEqual(pasted.roomSource);
  const legacyClosed = blocked.document, legacyA = legacyClosed.roomSource.rooms.find(room => room.originZone === own.id)!, legacyB = legacyClosed.roomSource.rooms.find(room => room.originZone === other.id)!;
  const solid = roomPortalTargets(legacyClosed, legacyA.id).find(target => target.neighbour === legacyB.id)!;
  const opened = applyCommand(legacyClosed, setRoomPortal(legacyClosed, legacyA.id, solid.edges[0]!, legacyB.id, true));
  expect(roomPortalTargets(opened, legacyA.id).find(target => target.neighbour === legacyB.id)!.open).toBe(true);
  expect(buildPlayableLevel(opened, -1).topology!.cells[own.id]!.neighbours).toContain(other.id);
}, 30000);

it('keeps a closed adjoining boundary closed after corner insertion and a skylight partition', () => {
  const fixture = roomHeightFixture(), document = { ...fixture.document, roomSource: translateSourceRooms(fixture.document.roomSource, [fixture.to.id], -744, 0).after };
  const target = roomPortalTargets(document, fixture.from.id).find(target => target.neighbour === fixture.to.id)!;
  const closed = applyCommand(document, setRoomPortal(document, fixture.from.id, target.edges[0]!, fixture.to.id, false));
  const split = { ...closed, roomSource: insertSourceCorner(closed.roomSource, fixture.from.id, target.edges[0]!, { x: 20256, z: 20128 }).after };
  expect(roomPortalTargets(split, fixture.from.id).find(target => target.neighbour === fixture.to.id)!.open).toBe(false);
  expect(buildPlayableLevel(split, -1)).toBeDefined();
  const style = document.roomAppearances!.find(item => item.room === fixture.from.id)!;
  const opened = applyCommand(closed, createRoomOpening(closed, fixture.from.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 }, style, 'sky').command);
  expect(roomPortalTargets(opened, fixture.to.id).filter(target => target.authored).every(target => target.open === false)).toBe(true);
  expect(roomPortalTargets(opened, fixture.to.id).some(target => target.authored)).toBe(true);
  expect(decodeProject(encodeProject(opened)).roomSource).toEqual(opened.roomSource);
});
