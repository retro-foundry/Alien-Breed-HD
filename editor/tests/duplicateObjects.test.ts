import { expect, it } from 'vitest';
import { roomHeightFixture } from './roomHeightFixture';
import { objectStoreyFixture } from './objectStoreyFixture';
import { applyCommand, execute, undo, redo, numberPatch } from '../src/model/commands';
import { createObject, objectTemplates, objectSlotLimit } from '../src/model/objectAuthoring';
import { duplicateObjects } from '../src/model/duplicateObjects';
import { encodeProject, decodeProject } from '../src/model/project';
import { buildPlayableLevel } from '../src/model/playableBuild';

function fixture() {
  const fixture = roomHeightFixture(); let document = fixture.document;
  const template = objectTemplates(document).find(object => object.type === 0)!.id, ids: number[] = [];
  for (const [x, z, y] of [[20128.5, 20128, -7.5], [20160.25, 20192, -10]]) {
    const proposal = createObject(document, { template, room: fixture.from.id, x: x!, z: z!, y: y! });
    document = applyCommand(document, proposal.command); ids.push(proposal.id);
  }
  document = applyCommand(document, { label: 'Distinct health', patches: [numberPatch(document, 'data', document.objects[ids[1]!]!.offset + 18, 37, 1)] });
  return { ...fixture, document, ids };
}

it('duplicates complete records as one relocation with independent CIDs, exact spacing, portable saves and undo', () => {
  const { document, ids } = fixture(), proposal = duplicateObjects(document, ids, 16.5, -8.25);
  let history = execute({ document, past: [], future: [] }, proposal.command);
  expect(history.past).toHaveLength(1);
  const saved = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(saved, -1).document;
  proposal.ids.forEach((id, index) => {
    const original = document.objects[ids[index]!]!, copy = built.objects[id]!;
    expect(copy).toMatchObject({ x: original.x + 16.5, z: original.z - 8.25, y: original.y, health: original.health, type: original.type, upper: original.upper });
    expect(copy.cid).not.toBe(original.cid);
    const sourceCopy = history.document.objects[id]!;
    expect(history.document.files.data.slice(sourceCopy.offset + 2, sourceCopy.offset + 64)).toEqual(document.files.data.slice(original.offset + 2, original.offset + 64));
  });
  expect(new Set(built.objects.filter(object => [...ids, ...proposal.ids].includes(object.id)).map(object => object.cid)).size).toBe(4);
  history = undo(history); expect(history.document.files).toEqual(document.files); expect(history.document.sourceObjects).toBe(document.sourceObjects);
  history = redo(history); expect(history.document.objects.length).toBe(document.objects.length + ids.length);
  expect(history.document.original).toBe(document.original);
});

it('rejects invalid destinations and unsupported templates without appending a partial copy', () => {
  const { document, ids } = fixture(), before = document.files;
  expect(() => duplicateObjects(document, ids, 100, 0)).toThrow(/outside rooms/);
  expect(() => duplicateObjects(document, ids, NaN, 0)).toThrow();
  expect(() => duplicateObjects(document, [ids[0]!, ids[0]!], 0, 0)).toThrow();
  const unsupported = applyCommand(document, { label: 'Unsupported template', patches: [numberPatch(document, 'data', document.objects[ids[1]!]!.offset + 16, -1, 1)] });
  expect(() => duplicateObjects(unsupported, ids, 0, 0)).toThrow(/cannot be duplicated/);
  expect(document.files).toBe(before);
});

it('resolves upper-floor copies to a compatible destination without changing original objects', () => {
  const fixture = objectStoreyFixture(), proposal = duplicateObjects(fixture.document, fixture.ids, 872, 0);
  const changed = applyCommand(fixture.document, proposal.command);
  for (const id of proposal.ids) {
    expect(changed.sourceObjects!.find(object => object.id === id)).toMatchObject({ room: fixture.to.id, x: 21000 });
    expect(changed.objects[id]!.upper).toBe(true);
  }
  for (const id of fixture.ids) expect(changed.sourceObjects!.find(object => object.id === id)!.room).toBe(fixture.from.id);
});

it('checks capacity for the entire batch before appending any record', () => {
  const fixture = fixtureForCapacity();
  expect(() => duplicateObjects(fixture.document, fixture.ids, 0, 0)).toThrow(/capacity/);
});

function fixtureForCapacity() {
  const start = fixture(); let document = start.document;
  while (document.objects.length < objectSlotLimit - 1) {
    const proposal = createObject(document, { template: start.ids[0]!, room: start.from.id, x: 20128, z: 20128, y: -8 });
    document = applyCommand(document, proposal.command);
  }
  return { document, ids: start.ids };
}
