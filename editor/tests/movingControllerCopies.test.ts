import { expect, it } from 'vitest';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { roomAppearanceBuildOptions } from '../src/model/roomAppearance';
import { importedAnimationTables } from '../src/model/animationBuild';
import { decodeProject, encodeProject } from '../src/model/project';

it.each(['door', 'lift'] as const)('allocates an independent %s controller with copied graphics and collision identities', kind => {
  const { document, copy, mechanism } = movingControllerCopyFixture(kind);
  const result = buildSourceLevel(document, { resolvedExit: -1, ...roomAppearanceBuildOptions(document), controllerCopies: [copy] });
  const records = importedAnimationTables(result.document).tables[kind === 'door' ? 'doors' : 'lifts']!;
  expect(records).toHaveLength(2);
  const [original, copied] = records;
  expect(copied!.zone).toBe(result.topology.cells.find(cell => cell.sourceId === copy.room)!.id);
  expect(copied!.zone).not.toBe(original!.zone); expect(copied!.graphic).not.toBe(original!.graphic);
  expect(copied!.header.slice(0, 8)).toEqual(copy.header!.slice(0, 8));
  expect(result.document.mechanisms.filter(item => item.kind === kind).map(item => item.openRule)).toEqual([kind === 'door' ? 5 : 3, kind === 'door' ? 4 : 2]);
  expect(copied!.walls).toHaveLength(mechanism.walls.length);
  for (const wall of copied!.walls) {
    expect(original!.walls.some(item => item.graphic === wall.graphic || item.line === wall.line)).toBe(false);
    const line = result.document.lines[wall.line]!;
    expect(line.x).toBeGreaterThan(10000); expect(line.z).toBeGreaterThan(10000);
  }
  expect(decodeProject(encodeProject(result.document)).files).toEqual(result.document.files);
});

it('rejects incomplete mappings, mismatched groups and unsynchronized initial positions atomically', () => {
  const { document, copy } = movingControllerCopyFixture('door'), files = document.files;
  const build = (candidate: typeof copy, input = document) => buildSourceLevel(input, {
    resolvedExit: -1, ...roomAppearanceBuildOptions(input), controllerCopies: [candidate],
  });
  expect(() => build({ ...copy, graphicsRooms: new Map() })).toThrow(/explicit copied room/);
  expect(() => build({ ...copy, collisionRooms: new Map() })).toThrow(/collision line.*explicit copied room/);
  expect(() => build({ ...copy, room: document.roomSource.rooms[0]!.id })).toThrow(/fresh room identity/);
  const header = copy.header!.slice(); new DataView(header.buffer).setInt16(4, new DataView(header.buffer).getInt16(4) - 1);
  expect(() => build({ ...copy, header })).toThrow(/initial position/);
  const other = [...copy.graphicsRooms.values()].find(id => id !== copy.room)!;
  const moved = { ...document, roomSource: { ...document.roomSource, rooms: document.roomSource.rooms.map(room => room.id !== other ? room
    : { ...room, vertices: room.vertices.map(point => ({ x: point.x + 128, z: point.z })) }) } };
  expect(() => build(copy, moved)).toThrow(/must move with/);
  expect(() => buildSourceLevel(document, { resolvedExit: -1, ...roomAppearanceBuildOptions(document), controllerCopies: [copy, copy] })).toThrow(/several controllers/);
  expect(document.files).toBe(files);
  expect(document.files).toEqual(document.original);
});
