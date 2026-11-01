import { expect, it } from 'vitest';
import { roomHeightFixture } from './roomHeightFixture';
import { moveRoomGroup } from '../src/model/roomGroups';
import { applyCommand, execute, undo, redo, setSourceTeleport } from '../src/model/commands';
import { createObject, objectTemplates } from '../src/model/objectAuthoring';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners, translateSourceRooms } from '../src/model/roomSource';
import { validateRoomInteriors } from '../src/model/geometry';

it('moves room geometry, starts, fractional object positions and inbound teleports atomically through native compilation and undo', () => {
  const fixture = roomHeightFixture(), template = objectTemplates(fixture.document).find(object => object.type === 0)!;
  const object = createObject(fixture.document, { template: template.id, room: fixture.from.id, x: 20160.25, z: 20128.5, y: -7.5 });
  let document = applyCommand(fixture.document, object.command);
  const incoming = document.roomSource.rooms.find(room => room.originZone === 0)!;
  document = applyCommand(document, setSourceTeleport(document, incoming.id, fixture.from.id, 20128, 20128));
  const command = moveRoomGroup(document, [fixture.from.id, fixture.to.id], 64, -32);
  let history = execute({ document, past: [], future: [] }, command);
  const saved = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(saved, -1).document;
  expect(built.starts[0]).toMatchObject({ x: 20192, z: 20096 });
  expect(built.objects[object.id]).toMatchObject({ x: 20224.25, z: 20096.5, y: -7.5, type: 0 });
  expect(saved.sourceTeleports!.find(link => link.room === incoming.id)).toMatchObject({ destination: fixture.from.id, x: 20192, z: 20096 });
  expect(saved.files.data).toEqual(document.files.data);
  expect(history.past).toHaveLength(1);
  history = undo(history); expect(history.document.roomSource).toBe(document.roomSource);
  expect(history.document.sourceStarts).toBe(document.sourceStarts); expect(history.document.sourceObjects).toBe(document.sourceObjects); expect(history.document.sourceTeleports).toBe(document.sourceTeleports);
  history = redo(history); expect(history.document.sourceStarts![0]).toMatchObject({ x: 20192, z: 20096 });
});

it('moves groups simultaneously rather than rejecting an intermediate overlap with another selected room', () => {
  const { document, from, to } = roomHeightFixture();
  expect(() => moveRoomGroup(document, [from.id], 1000, 0)).toThrow(/overlap/);
  const moved = applyCommand(document, moveRoomGroup(document, [from.id, to.id], 1000, 0));
  expect(moved.roomSource.rooms.find(room => room.id === from.id)!.vertices.some(point => point.x === 21000)).toBe(true);
  expect(moved.roomSource.rooms.find(room => room.id === to.id)!.vertices.some(point => point.x === 22000)).toBe(true);
  expect(() => moveRoomGroup(document, [from.id, to.id], 0.25, 0)).toThrow(/whole number/);
  expect(() => moveRoomGroup(document, [from.id, to.id], 20000, 0)).toThrow();
  expect(moveRoomGroup(document, [from.id, to.id], 0, 0).patches).toHaveLength(0);
});

it('moves a partial shared edge through its T-junction fan while retaining unrelated neighbour corners', () => {
  const { original: document } = roomHeightFixture();
  let source = removeSourceRooms(document.roomSource, document.roomSource.rooms.map(room => room.id)).after;
  for (const vertices of [
    [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 128 }, { x: 0, z: 128 }],
    [{ x: 128, z: 32 }, { x: 256, z: 32 }, { x: 256, z: 96 }, { x: 128, z: 96 }],
  ]) source = commitRoomDraft(source, setDraftCorners(beginRoomDraft(document, source, 0), vertices)).after;
  const moved = translateSourceRooms(source, [source.rooms[1]!.id], 16, 0).after;
  expect(moved.rooms[0]!.vertices).toEqual(expect.arrayContaining([{ x: 0, z: 0 }, { x: 0, z: 128 }, { x: 144, z: 32 }, { x: 144, z: 96 }]));
  const edges = validateRoomInteriors(moved.rooms.map(room => ({ id: room.id, vertices: [...room.vertices] })));
  expect(edges.filter(edge => edge.neighbour !== null)).toHaveLength(2);
  expect(source.rooms[0]!.vertices).toHaveLength(4);
});
