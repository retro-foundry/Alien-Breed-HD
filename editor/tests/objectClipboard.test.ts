import { expect, it } from 'vitest';
import { roomHeightFixture } from './roomHeightFixture';
import { applyCommand, execute, undo, redo, numberPatch, removeObject } from '../src/model/commands';
import { createObject, objectSlotLimit, objectTemplates } from '../src/model/objectAuthoring';
import { copyObjects, pasteObjects } from '../src/model/objectClipboard';
import { encodeProject, decodeProject } from '../src/model/project';

function fixture() {
  const fixture = roomHeightFixture();
  const template = objectTemplates(fixture.document).find(object => object.type === 0)!.id;
  const proposal = createObject(fixture.document, { template, room: fixture.from.id, x: 20128, z: 20128, y: -7.5 });
  return { ...fixture, added: applyCommand(fixture.document, proposal.command), id: proposal.id };
}

it('pastes captured records after their originals change or are undone, with independent identities and one undo', () => {
  const { document, added, id } = fixture();
  const clipboard = copyObjects(added, [id]);
  // Undoing the creation must not invalidate the captured template.
  const proposal = pasteObjects(document, clipboard);
  let history = execute({ document, past: [], future: [] }, proposal.command);
  expect(history.past).toHaveLength(1);
  let copy = history.document.objects[proposal.ids[0]!]!;
  expect(copy).toMatchObject({ x: 20128, z: 20128, y: -7.5, type: 0 });
  expect(decodeProject(encodeProject(history.document)).objects[copy.id]).toMatchObject({ y: -7.5, type: 0 });
  history = undo(history); expect(history.document.files).toEqual(document.files);
  history = redo(history); expect(history.document.objects.length).toBe(document.objects.length + 1);

  const changed = applyCommand(added, { label: 'Change original health', patches: [numberPatch(added, 'data', added.objects[id]!.offset + 18, 37, 1)] });
  const deleted = applyCommand(changed, removeObject(changed, id));
  const pasted = pasteObjects(deleted, clipboard, 8, 0), result = applyCommand(deleted, pasted.command);
  copy = result.objects[pasted.ids[0]!]!;
  expect(copy).toMatchObject({ health: added.objects[id]!.health, y: -7.5, x: 20136, z: 20128 });
  expect(copy.cid).not.toBe(result.objects[id]!.cid);
  expect(result.files.data.slice(copy.offset + 2, copy.offset + 64)).toEqual(added.files.data.slice(added.objects[id]!.offset + 2, added.objects[id]!.offset + 64));
  expect(result.objects[id]!.health).toBe(37);
  expect(result.sourceObjects!.find(object => object.id === id)!.room).toBeNull();
});

it('copies a group without aliasing caller selection and rejects the whole invalid paste', () => {
  const { added, id, from } = fixture();
  const proposal = createObject(added, { template: id, room: from.id, x: 20160, z: 20128, y: -10 });
  const document = applyCommand(added, proposal.command), selection = [id, proposal.id];
  const clipboard = copyObjects(document, selection); selection.pop();
  expect(clipboard.ids).toHaveLength(2);
  const paste = pasteObjects(document, clipboard, 8.25, 0), result = applyCommand(document, paste.command);
  expect(paste.ids.map(id => result.objects[id]!.x)).toEqual([20136.25, 20168.25]);
  expect(paste.ids.map(id => result.objects[id]!.y)).toEqual([-7.5, -10]);
  expect(() => pasteObjects(document, clipboard, 100, 0)).toThrow(/No objects were pasted/);
  expect(() => pasteObjects(document, clipboard, NaN, 0)).toThrow();
  expect(document.objects.length).toBe(added.objects.length + 1);
});

it('rejects empty, repeated, inactive, player and cross-project copies', () => {
  const { added, id } = fixture();
  expect(() => copyObjects(added, [])).toThrow();
  expect(() => copyObjects(added, [id, id])).toThrow();
  expect(() => copyObjects(applyCommand(added, removeObject(added, id)), [id])).toThrow(/cannot be copied/);
  const playerOffset = new DataView(added.files.data.buffer, added.files.data.byteOffset).getInt32(46);
  expect(() => copyObjects(added, [added.objects.find(object => object.offset === playerOffset)!.id])).toThrow(/cannot be copied/);
  expect(() => pasteObjects(roomHeightFixture().document, copyObjects(added, [id]))).toThrow(/another project/);
});

it('checks capacity for the whole clipboard before allocating any source records', () => {
  const { added, id, from } = fixture();
  const second = createObject(added, { template: id, room: from.id, x: 20160, z: 20128, y: -10 });
  let document = applyCommand(added, second.command);
  const clipboard = copyObjects(document, [id, second.id]);
  while (document.objects.length < objectSlotLimit - 1) {
    const proposal = createObject(document, { template: id, room: from.id, x: 20128, z: 20128, y: -7.5 });
    document = applyCommand(document, proposal.command);
  }
  const before = document.files;
  expect(() => pasteObjects(document, clipboard)).toThrow(/capacity/);
  expect(document.files).toBe(before);
  expect(document.objects.length).toBe(objectSlotLimit - 1);
});
