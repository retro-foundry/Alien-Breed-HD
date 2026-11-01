import { expect, it } from 'vitest';
import { objectTemplates, replaceObjectSettings, createObject } from '../src/model/objectAuthoring';
import { applyCommand, execute, undo, redo } from '../src/model/commands';
import { openLevel } from '../src/model/level';
import { encodeProject, decodeProject } from '../src/model/project';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { originalFiles } from './helpers';
import { roomHeightFixture } from './roomHeightFixture';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('changes complete object settings without moving identities or other records in Level %i', level => {
  const document = openLevel(`Level ${level}`, originalFiles(level));
  const templates = objectTemplates(document), object = templates.find(item => item.zone >= 0)!;
  const template = templates.find(item => item.type !== object.type)!;
  expect(template).toBeDefined();
  const command = replaceObjectSettings(document, object.id, template.id), changed = applyCommand(document, command);
  const actual = changed.objects[object.id]!;
  expect(actual).toMatchObject({ id: object.id, cid: object.cid, pointOffset: object.pointOffset, x: object.x, z: object.z, y: object.y, zone: object.zone, upper: object.upper, type: template.type, health: template.health });
  const preserved = new Set([0, 1, 4, 5, 12, 13, 26, 27, 63]);
  for (let byte = 0; byte < 64; byte++) expect(changed.files.data[object.offset + byte]).toBe(document.files.data[(preserved.has(byte) ? object.offset : template.offset) + byte]);
  expect(Buffer.from(changed.files.data.subarray(0, object.offset)).equals(Buffer.from(document.files.data.subarray(0, object.offset)))).toBe(true);
  expect(Buffer.from(changed.files.data.subarray(object.offset + 64)).equals(Buffer.from(document.files.data.subarray(object.offset + 64)))).toBe(true);
  expect(changed.roomSource).toBe(document.roomSource); expect(changed.sourceObjects).toBe(document.sourceObjects);
  expect(changed.files.graphics).toBe(document.files.graphics); expect(changed.files.clips).toBe(document.files.clips);
  let history = execute({ document, past: [], future: [] }, command);
  history = undo(history); expect(history.document.files.data).toEqual(document.files.data);
  history = redo(history); expect(history.document.files.data).toEqual(changed.files.data);
});

it('retains authored placement through type change, portable save and native compilation', () => {
  const { document, from } = roomHeightFixture();
  const templates = objectTemplates(document), template = templates.find(item => item.type === 1)!;
  const created = createObject(document, { template: template.id, room: from.id, x: 20128, z: 20192, y: -7.5 });
  const placed = applyCommand(document, created.command), key = objectTemplates(placed).find(item => item.type === 4)!;
  const changed = applyCommand(placed, replaceObjectSettings(placed, created.id, key.id));
  const saved = decodeProject(encodeProject(changed)), built = buildPlayableLevel(saved, -1);
  expect(saved.objects[created.id]).toMatchObject({ type: 4, y: -7.5 });
  expect(saved.sourceObjects!.find(item => item.id === created.id)).toMatchObject({ room: from.id, x: 20128, z: 20192 });
  expect(built.document.objects[created.id]).toMatchObject({ type: 4, x: 20128, z: 20192, y: -7.5, zone: built.document.starts[0]!.zone });
  for (const object of document.objects.filter(item => !templates.some(template => template.id === item.id)))
    expect(() => replaceObjectSettings(document, object.id, template.id)).toThrow(/editable/);
  expect(() => replaceObjectSettings(document, template.id, -1)).toThrow(/editable/);
});
