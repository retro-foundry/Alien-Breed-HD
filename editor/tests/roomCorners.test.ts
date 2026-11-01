import { objectTemplates, createObject } from '../src/model/objectAuthoring';
import { setSourceTeleport } from '../src/model/commands';
import { setSurfaceMaterial, wallMaterialChoices } from '../src/model/wallMaterials';
import { expect, it } from 'vitest';
import { openLevel, sameBytes } from '../src/model/level';
import { originalFiles } from './helpers';
import { addRoomCorner, deleteRoomCorner, edgeCornerPoint } from '../src/model/roomCorners';
import { applyCommand, execute, undo, redo, editRoomSource, setSourceStart } from '../src/model/commands';
import { moveSourceCorner } from '../src/model/roomSource';
import { roomHeightFixture } from './roomHeightFixture';
import { moveRoomGroup } from '../src/model/roomGroups';
import { decodeProject, encodeProject } from '../src/model/project';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { validateRoomInteriors } from '../src/model/geometry';

it('snaps slanted edges along the native integer lattice and rejects unsplittable edges', () => {
  expect(edgeCornerPoint({ x: 0, z: 0 }, { x: 12, z: 8 }, { x: 4, z: 3 })).toEqual({ x: 3, z: 2 });
  expect(edgeCornerPoint({ x: 0, z: 0 }, { x: 12, z: 8 }, { x: 0, z: 0 })).toEqual({ x: 3, z: 2 });
  expect(() => edgeCornerPoint({ x: 0, z: 0 }, { x: 3, z: 2 })).toThrow('whole-coordinate');
});
it.each(Array.from({ length: 16 }, (_, i) => i + 1))('level %s splits native edge identities, compiles, saves and removes the split exactly', level => {
  const document = openLevel(`Level ${level}`, originalFiles(level));
  let room = document.roomSource.rooms[0]!, edge = 0;
  outer: for (const candidate of document.roomSource.rooms) for (let i = 0; i < candidate.vertices.length; i++) {
    try { edgeCornerPoint(candidate.vertices[i]!, candidate.vertices[(i + 1) % candidate.vertices.length]!); room = candidate; edge = i; break outer; } catch { /* Native unit edges have no interior lattice point. */ }
  }
  const point = edgeCornerPoint(room.vertices[edge]!, room.vertices[(edge + 1) % room.vertices.length]!);
  const changed = applyCommand(document, addRoomCorner(document, room.id, edge));
  const result = changed.roomSource.rooms.find(r => r.id === room.id)!;
  expect(result.vertices).toHaveLength(room.vertices.length + 1);
  expect(result.wallSources.filter(id => id === room.wallSources[edge]).length).toBe(room.wallSources.filter(id => id === room.wallSources[edge]).length + 1);
  const saved = decodeProject(encodeProject(changed));
  expect(saved.roomSource.rooms.find(r => r.id === room.id)!.vertices).toContainEqual(point);
  const built = buildPlayableLevel(saved, -1); expect(built.document.zones.length).toBeGreaterThan(0);
  const restored = applyCommand(changed, deleteRoomCorner(changed, room.id, result.vertices.findIndex(p => p.x === point.x && p.z === point.z)));
  expect(restored.roomSource.rooms).toEqual(document.roomSource.rooms);
  expect(sameBytes(restored.files.data, document.files.data)).toBe(true);
});
it('shared edges split on both sides, move without cracks and undo as one operation', () => {
  const fixture = roomHeightFixture();
  const document = applyCommand(fixture.document, moveRoomGroup(fixture.document, [fixture.to.id], -744, 0));
  const room = document.roomSource.rooms.find(r => r.id === fixture.from.id)!, edge = room.vertices.findIndex((p, i) => p.x === 20256 && room.vertices[(i + 1) % room.vertices.length]!.x === 20256);
  let history = execute({ document, past: [], future: [] }, addRoomCorner(document, room.id, edge));
  const point = { x: 20256, z: 20128 };
  for (const id of [fixture.from.id, fixture.to.id]) expect(history.document.roomSource.rooms.find(r => r.id === id)!.vertices).toContainEqual(point);
  expect(history.past).toHaveLength(1); history = undo(history); expect(history.document.roomSource).toBe(document.roomSource); history = redo(history);
  const selected = history.document.roomSource.rooms.find(r => r.id === room.id)!;
  const moved = applyCommand(history.document, editRoomSource(moveSourceCorner(history.document.roomSource, room.id, selected.vertices.findIndex(p => p.x === point.x && p.z === point.z), { x: 20240, z: 20128 })));
  const edges = validateRoomInteriors(moved.roomSource.rooms.map(r => ({ id: r.id, vertices: [...r.vertices] })));
  expect(edges.filter(e => e.room === fixture.from.id && e.neighbour === fixture.to.id)).toHaveLength(2);
  expect(buildPlayableLevel(moved, -1).document.zones.length).toBeGreaterThan(document.zones.length);
});
it('deletion rejects a stranded player before changing any room', () => {
  const fixture = roomHeightFixture();
  const document = applyCommand(fixture.document, setSourceStart(fixture.document, 1, fixture.from.id, 20008, 20008));
  const room = document.roomSource.rooms.find(r => r.id === fixture.from.id)!;
  expect(() => deleteRoomCorner(document, room.id, room.vertices.findIndex(p => p.x === 20000 && p.z === 20000))).toThrow('Player 1 start');
});

it('split and merged painted edges retain their material identity and sampling', () => {
  const fixture = roomHeightFixture(), style = wallMaterialChoices(fixture.document)[0]!;
  const document = applyCommand(fixture.document, editRoomSource(setSurfaceMaterial(fixture.document, { room: fixture.from.id, index: 0, upper: false }, style.offset)));
  const room = document.roomSource.rooms.find(r => r.id === fixture.from.id)!, point = edgeCornerPoint(room.vertices[0]!, room.vertices[1]!);
  const split = applyCommand(document, addRoomCorner(document, room.id, 0)), result = split.roomSource.rooms.find(r => r.id === room.id)!;
  expect(result.boundaryMaterials!.overrides).toEqual(room.boundaryMaterials!.overrides);
  expect(result.boundaryMaterials!.edges.filter(id => id === room.boundaryMaterials!.edges[0])).toHaveLength(2);
  const restored = applyCommand(split, deleteRoomCorner(split, room.id, result.vertices.findIndex(p => p.x === point.x && p.z === point.z)));
  expect(restored.roomSource.rooms.find(r => r.id === room.id)!.boundaryMaterials).toEqual(room.boundaryMaterials);
  expect(() => buildPlayableLevel(split, -1)).not.toThrow();
});
it('deletion protects object and teleport destinations, and never reduces a room below three corners', () => {
  const fixture = roomHeightFixture(), room = fixture.from, index = room.vertices.findIndex(p => p.x === 20000 && p.z === 20000);
  const template = objectTemplates(fixture.document).find(o => o.type === 10)!;
  const added = createObject(fixture.document, { template: template.id, room: room.id, x: 20008, z: 20008, y: template.y });
  const objectDocument = applyCommand(fixture.document, added.command);
  expect(() => deleteRoomCorner(objectDocument, room.id, index)).toThrow(`Object ${added.id}`);
  const teleportDocument = applyCommand(fixture.document, setSourceTeleport(fixture.document, fixture.to.id, room.id, 20008, 20008));
  expect(() => deleteRoomCorner(teleportDocument, room.id, index)).toThrow('Teleport destination');
  const triangle = applyCommand(fixture.document, deleteRoomCorner(fixture.document, room.id, index));
  expect(() => deleteRoomCorner(triangle, room.id, 0)).toThrow('at least three');
});
