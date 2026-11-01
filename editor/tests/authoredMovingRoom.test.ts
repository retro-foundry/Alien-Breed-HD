import { expect, it } from 'vitest';
import { authoredMovingFixture } from './authoredMovingFixture';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { importedAnimationTables, validateAnimationReferences } from '../src/model/animationBuild';
import { applyCommand, editRoomSource, editRoomAppearance } from '../src/model/commands';
import { beginRoomDraft, setDraftCorners, commitRoomDraft } from '../src/model/roomSource';
import { roomAppearanceChoices, roomAppearanceBuildOptions } from '../src/model/roomAppearance';
import { appendAuthoredMovingAnimations } from '../src/model/authoredMovingRoom';

it.each(['door', 'lift'] as const)('compiles an independent authored %s with reciprocal collision links, complete travel walls and exact selected material', kind => {
  const { document, room, neighbour, input, options, built, mechanism } = authoredMovingFixture(kind);
  const source = document.files.graphics.slice();
  expect(built.topology.cells.find(cell => cell.id === mechanism.zone)!.sourceId).toBe(room.id);
  expect(mechanism.walls).toHaveLength(2);
  expect(new Set(mechanism.walls.map(wall => wall.line)).size).toBe(2);
  for (const wall of mechanism.walls) {
    const record = built.document.zones.flatMap(zone => zone.lower).find(record => record.offset === wall.graphic)!;
    expect(record.wall).toBeDefined(); expect(record.wall!.texture).toBe(input.wall.surface.texture);
    const line = built.document.lines[wall.line]!;
    expect([mechanism.zone, built.topology.cells.find(cell => cell.sourceId === neighbour.id)!.id]).toContain(line.connect);
    expect(line.length).toBeGreaterThan(0);
    const base = kind === 'door' ? input.wall.surface.horizontalOffset * 4096 + ((input.wall.surface.verticalOffset + Math.floor(mechanism.current)) & 255)
      : input.wall.surface.horizontalOffset * 65536 + (input.wall.surface.verticalOffset & 65535);
    expect(wall.base).toBe(base);
    expect(record.wall).toMatchObject(kind === 'door' ? { topFixed: mechanism.current * 256, bottomFixed: mechanism.current * 256 }
      : { topFixed: mechanism.current * 256, bottomFixed: (mechanism.current - 32) * 256 });
  }
  if (kind === 'lift') expect(built.document.zones[mechanism.zone]!.lower.some(record => record.wall?.topFixed === (mechanism.current - 32) * 256 && record.wall?.bottomFixed === mechanism.bottom * 256)).toBe(true);
  validateAnimationReferences(built.document);
  expect(importedAnimationTables(built.document).tables[kind === 'door' ? 'doors' : 'lifts']!.at(-1)!.walls).toHaveLength(2);
  expect(document.files.graphics).toEqual(source);
  for (const change of [{ header: input.header.slice(0, 17) }, { room: 'missing-room' }])
    expect(() => buildSourceLevel(document, { ...options, authoredControllers: [{ ...input, ...change }] })).toThrow();
  const wrong = input.header.slice(); new DataView(wrong.buffer).setInt16(4, (mechanism.current + 0.25) * 4);
  expect(() => buildSourceLevel(document, { ...options, authoredControllers: [{ ...input, header: wrong }] })).toThrow();
  expect(() => buildSourceLevel(document, { ...options, authoredControllers: [input, input] })).toThrow(/one controller/);
  const crossing = input.header.slice(), bounds = new DataView(crossing.buffer);
  if (kind === 'door') bounds.setInt16(0, (document.zones[room.recordZone]!.floor + 0.25) * 4);
  else bounds.setInt16(2, (document.zones[room.recordZone]!.ceiling - 0.25) * 4);
  expect(() => buildSourceLevel(document, { ...options, authoredControllers: [{ ...input, header: crossing }] })).toThrow(/crosses/);
});


it('adds the required empty native water structure only when authoring the first lift', () => {
  const tables = { doors: null, lifts: null, water: [], switches: null, trailing: new Uint8Array() };
  expect(appendAuthoredMovingAnimations(tables, [])).toBe(tables);
  const input = authoredMovingFixture('lift'), animation = importedAnimationTables(input.built.document).tables.lifts!.at(-1)!;
  const added = appendAuthoredMovingAnimations(tables, [{ kind: 'lift', animation }]);
  expect(added.lifts).toHaveLength(1); expect(added.doors).toBeNull(); expect(added.water).toHaveLength(21);
  for (const group of added.water) { expect(group.header).toEqual(new Uint8Array(14)); expect(group.entries).toEqual([]); }
  expect(tables.water).toEqual([]);
});

it('inverts native door scrolling without changing the sampled authored row phase', () => {
  const { input, mechanism } = authoredMovingFixture('door');
  const base = input.wall.surface.horizontalOffset * 4096 + ((input.wall.surface.verticalOffset + Math.floor(mechanism.current)) & 255);
  const row = (base & 4095) + ((-Math.floor(mechanism.current)) & 255);
  for (const mask of [15, 31, 63, 127, 255]) for (let y = -512; y < 512; y++)
    expect((row + y) & mask).toBe((input.wall.surface.verticalOffset + y) & mask);
});


it('keeps exact independent targets when two new doors insert wall bands into the same neighbouring room', () => {
  const fixture = authoredMovingFixture('door');
  let document = fixture.document;
  document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, fixture.room.recordZone),
    [{ x: 20512, z: 20000 }, { x: 20768, z: 20000 }, { x: 20768, z: 20256 }, { x: 20512, z: 20256 }]))));
  const room = document.roomSource.rooms.at(-1)!, choices = roomAppearanceChoices(document, room);
  document = applyCommand(document, editRoomAppearance(document, { room: room.id, lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]! }));
  const header = fixture.input.header.slice(); new DataView(header.buffer).setInt16(0, (fixture.mechanism.bottom - 16) * 4);
  const built = buildSourceLevel(document, { resolvedExit: -1, ...roomAppearanceBuildOptions(document), authoredControllers: [fixture.input, { ...fixture.input, room: room.id, header }] });
  const doors = built.document.mechanisms.filter(item => item.kind === 'door').slice(-2);
  expect(doors).toHaveLength(2); expect(doors[0]!.bottom).not.toBe(doors[1]!.bottom);
  const neighbour = built.topology.cells.find(cell => cell.sourceId === fixture.neighbour.id)!;
  const walls = doors.map(door => {
    expect(door.walls).toHaveLength(2);
    const record = built.document.zones[neighbour.id]!.lower.find(record => record.offset === door.walls[0]!.graphic)!;
    expect(record.wall).toBeDefined();
    expect(door.walls[1]!.graphic).toBe(record.offset);
    return record;
  });
  expect(walls[0]!.offset).not.toBe(walls[1]!.offset);
  expect(walls[0]!.points.map(id => built.document.points[id]!.x)).toEqual([20256, 20256]);
  expect(walls[1]!.points.map(id => built.document.points[id]!.x)).toEqual([20512, 20512]);
  validateAnimationReferences(built.document);
});
