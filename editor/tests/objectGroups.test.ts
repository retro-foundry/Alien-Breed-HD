import { expect, it } from 'vitest';
import { roomHeightFixture } from './roomHeightFixture';
import { createObject, objectTemplates } from '../src/model/objectAuthoring';
import { applyCommand, execute, redo, undo, numberPatch, moveObject, setSourceObject } from '../src/model/commands';
import { editObjectGroup, objectSelectionIds, objectsInRectangle } from '../src/model/objectGroups';
import { editableObjects } from '../src/model/sourceObjects';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { BinaryReader } from '../src/model/binary';
import { objectStoreyFixture } from './objectStoreyFixture';
import { objectDestinationRoom } from '../src/model/sourceObjects';

function fixture() {
  const fixture = roomHeightFixture(); let document = fixture.document;
  const template = objectTemplates(document).find(object => object.type === 0)!.id, ids: number[] = [];
  for (const [x, z, y] of [[20128, 20128, -7.5], [20240, 20192, -10]]) {
    const proposal = createObject(document, { template, room: fixture.from.id, x: x!, z: z!, y: y! });
    document = applyCommand(document, proposal.command); ids.push(proposal.id);
  }
  return { document, ids };
}

it('moves and aligns an object group atomically through undo, saved source and native builds', () => {
  const { document, ids } = fixture();
  let history = execute({ document, past: [], future: [] }, editObjectGroup(document, ids, { kind: 'move', x: -8.5, z: 16.25 }));
  expect(editableObjects(history.document).filter(object => ids.includes(object.id)).map(object => [object.x, object.z, object.y])).toEqual([[20119.5, 20144.25, -7.5], [20231.5, 20208.25, -10]]);
  history = undo(history); expect(history.document.sourceObjects).toBe(document.sourceObjects);
  history = redo(history);
  history = execute(history, editObjectGroup(history.document, ids, { kind: 'align', axis: 'x' }));
  const restored = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(restored, -1).document;
  expect(ids.map(id => built.objects[id]!.x)).toEqual([20119.5, 20119.5]);
  expect(ids.map(id => built.objects[id]!.y)).toEqual([-7.5, -10]);
  expect(restored.files).toEqual(document.files);
});

it('rejects a partly valid movement without changing any object and deletes/restores the whole group', () => {
  const { document, ids } = fixture(), before = document.sourceObjects;
  expect(() => editObjectGroup(document, ids, { kind: 'move', x: 32, z: 0 })).toThrow(/outside/);
  expect(document.sourceObjects).toBe(before);
  const history = execute({ document, past: [], future: [] }, editObjectGroup(document, ids, { kind: 'delete' }));
  expect(ids.every(id => history.document.sourceObjects!.find(object => object.id === id)!.room === null)).toBe(true);
  expect(ids.every(id => buildPlayableLevel(history.document, -1).document.objects[id]!.zone === -1)).toBe(true);
  expect(undo(history).document.sourceObjects).toBe(before);
});

it('rejects duplicate IDs, missing objects, invalid offsets and player avatars', () => {
  const { document, ids } = fixture();
  for (const selection of [[ids[0]!, ids[0]!], [ids[0]!], [ids[0]!, 999]])
    expect(() => editObjectGroup(document, selection, { kind: 'delete' })).toThrow();
  expect(() => editObjectGroup(document, ids, { kind: 'move', x: NaN, z: 0 })).toThrow();
  const avatarOffset = new BinaryReader(document.files.data, 'avatar').i32(46);
  const avatar = document.objects.find(object => object.offset === avatarOffset)!;
  expect(() => editObjectGroup(document, [ids[0]!, avatar.id], { kind: 'delete' })).toThrow(/avatars/);
});

it('chooses a compatible upper room on a shared edge instead of the first lower-only candidate', () => {
  const fixture = objectStoreyFixture(), original = fixture.document, objects = editableObjects(original);
  expect(objectDestinationRoom(original, objects.find(object => object.id === fixture.ids[0])!, { x: 21000, z: 20128 })?.id).toBe(fixture.to.id);
  const changed = applyCommand(original, editObjectGroup(original, fixture.ids, { kind: 'move', x: 872, z: 0 }));
  expect(fixture.ids.map(id => changed.sourceObjects!.find(object => object.id === id)!.room)).toEqual([fixture.to.id, fixture.to.id]);
  const built = buildPlayableLevel(changed, -1);
  for (const id of fixture.ids) expect(built.document.objects[id]).toMatchObject({ zone: built.topology!.sourceCells.get(fixture.to.id)![0], upper: true, x: 21000 });
  const current = editableObjects(changed).find(object => object.id === fixture.ids[0])!;
  expect(objectDestinationRoom(changed, current, { x: 21000, z: 20128 })?.id).toBe(fixture.to.id);
});

it('rejects moving upper objects wholly inside a single-storey room without partial changes', () => {
  const fixture = objectStoreyFixture(), before = fixture.document.sourceObjects;
  expect(() => editObjectGroup(fixture.document, fixture.ids, { kind: 'move', x: 800, z: 0 })).toThrow(/supporting its floor/);
  expect(fixture.document.sourceObjects).toBe(before);
});

it('also enforces upper-floor compatibility for explicit native object moves', () => {
  const fixture = roomHeightFixture(), original = fixture.original;
  const object = original.objects.find(object => object.type === 1)!, zone = original.zones[0]!;
  const document = applyCommand(original, { label: 'Upper fixture', patches: [numberPatch(original, 'data', object.offset + 63, 1, 1)] });
  const lines = zone.boundaryIds.map(id => document.lines[id]!);
  const x = lines.reduce((sum, line) => sum + line.x, 0) / lines.length, z = lines.reduce((sum, line) => sum + line.z, 0) / lines.length;
  expect(() => moveObject(document, object.id, x, z, zone.id)).toThrow(/upper floor/);
});

it('selects active search matches with floor filtering while excluding both player avatars', () => {
  const fixture = objectStoreyFixture(), document = fixture.document;
  const upperHealth = objectSelectionIds(document, 'HEALTH', true);
  expect(fixture.ids.every(id => upperHealth.includes(id))).toBe(true);
  expect(fixture.ids.some(id => objectSelectionIds(document, 'health', false).includes(id))).toBe(false);
  expect(objectSelectionIds(document, 'no-such-object')).toEqual([]);
  const deleted = applyCommand(document, editObjectGroup(document, fixture.ids, { kind: 'delete' }));
  expect(fixture.ids.some(id => objectSelectionIds(deleted).includes(id))).toBe(false);
  const reader = new BinaryReader(document.files.data, 'avatars');
  for (const field of [46, 50]) expect(objectSelectionIds(document)).not.toContain(document.objects.find(object => object.offset === reader.i32(field))!.id);
});

it('box selection uses authored positions, includes edge points and accepts either drag direction', () => {
  const fixture = objectStoreyFixture(), document = fixture.document;
  const a = { x: 20128, z: 20128 }, b = { x: 20129, z: 20192 };
  expect(objectsInRectangle(document, a, b, true)).toEqual(fixture.ids);
  expect(objectsInRectangle(document, b, a, true)).toEqual(fixture.ids);
  expect(objectsInRectangle(document, { x: a.x, z: b.z }, { x: b.x, z: a.z }, true)).toEqual(fixture.ids);
  expect(objectsInRectangle(document, a, b, false).some(id => fixture.ids.includes(id))).toBe(false);
  expect(() => objectsInRectangle(document, a, { x: Infinity, z: 0 }, true)).toThrow(/finite/);
  const moved = applyCommand(document, setSourceObject(document, fixture.ids[0]!, fixture.from.id, 20145.25, 20155.75));
  expect(objectsInRectangle(moved, { x: 20145, z: 20155 }, { x: 20146, z: 20156 }, true)).toEqual([fixture.ids[0]!]);
});
