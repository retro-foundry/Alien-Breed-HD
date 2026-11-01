import { expect, it } from 'vitest';
import { originalObjectTypes, changeObjectType } from '../src/model/objectTypes';
import { originalFiles } from './helpers';
import { openLevel } from '../src/model/level';
import { objectTemplates } from '../src/model/objectAuthoring';
import { applyCommand, execute, undo, redo } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';

const catalog = originalObjectTypes(Array.from({ length: 16 }, (_, index) => originalFiles(index + 1).data));
it('offers unique named original types with verifiable pristine records and no reserved slots', () => {
  expect(catalog.length).toBeGreaterThan(14);
  expect(new Set(catalog.map(item => item.type)).size).toBe(catalog.length);
  for (const choice of catalog) {
    const source = openLevel('Original', originalFiles(choice.level)), object = objectTemplates(source).find(item => item.id === choice.object)!;
    expect(object.type).toBe(choice.type);
    expect([...source.files.data.slice(object.offset, object.offset + 64)]).toEqual(choice.record);
    expect([2, 5, 11]).not.toContain(choice.type);
  }
});
it.each(catalog.map(choice => [choice.name, choice] as const))('changes directly to %s using native class settings while retaining identity, placement, history and saved bytes', (_name, choice) => {
  const document = openLevel('Original', originalFiles(1)), target = objectTemplates(document).find(item => item.type === 10 && item.zone >= 0)!;
  const command = changeObjectType(document, target.id, choice), changed = applyCommand(document, command), result = changed.objects[target.id]!;
  expect(result).toMatchObject({ type: choice.type, cid: target.cid, zone: target.zone, x: target.x, z: target.z, y: target.y, upper: target.upper });
  const saved = decodeProject(encodeProject(changed)); expect(saved.files.data).toEqual(changed.files.data);
  for (const object of document.objects.filter(item => item.id !== target.id)) expect(changed.objects[object.id]).toEqual(object);
  let history = execute({ document, past: [], future: [] }, command);
  history = undo(history); expect(history.document.files.data).toEqual(document.files.data);
  history = redo(history); expect(history.document.files.data).toEqual(changed.files.data);
});
