import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { execute, undo, redo, applyCommand, numberPatch } from '../src/model/commands';
import { BinaryReader } from '../src/model/binary';
import { keyColour, setKeyColour, ammunitionChoices, ammunitionType, setAmmunitionType } from '../src/model/objectSettings';
import ammoFrames from '../src/model/ammoFrames.json' with { type: 'json' };
import { decodeProject, encodeProject } from '../src/model/project';

it('changes key artwork and door condition atomically across original levels', () => {
  let checked = 0;
  for (let level = 1; level <= 16; level++) {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    for (const object of document.objects.filter(object => object.type === 4)) {
      checked++;
      expect(keyColour(document, object.id)).not.toBeUndefined();
      for (let colour = 0; colour < 4; colour++) {
        let history = execute({ document, past: [], future: [] }, setKeyColour(document, object.id, colour));
        const data = new BinaryReader(history.document.files.data, 'key');
        expect(data.i32(object.offset + 8)).toBe(5 * 65536 + colour);
        expect(data.u8(object.offset + 17)).toBe(2 ** colour);
        if (history.past.length) {
          history = undo(history);
          for (const file of ['data', 'graphics', 'clips'] as const)
            expect(Buffer.from(history.document.files[file]).equals(Buffer.from(document.files[file]))).toBe(true);
          history = redo(history);
        }
        expect(keyColour(history.document, object.id)).toBe(colour);
      }
    }
  }
  expect(checked).toBeGreaterThan(0);
});

it('retains chosen artwork and conditions in a saved project', () => {
  const document = openLevel('Level 1', originalFiles(1)), key = document.objects.find(object => object.type === 4)!;
  const changed = execute({ document, past: [], future: [] }, setKeyColour(document, key.id, 3)).document;
  expect(keyColour(decodeProject(encodeProject(changed)), key.id)).toBe(3);
});

it('rejects unsupported objects and invalid colours', () => {
  const document = openLevel('Level 1', originalFiles(1)), key = document.objects.find(object => object.type === 4)!;
  for (const colour of [-1, 4, 0.5, NaN]) expect(() => setKeyColour(document, key.id, colour)).toThrow();
  expect(() => setKeyColour(document, document.objects.find(object => object.type !== 4)!.id, 0)).toThrow(/key pickup/);
});

it('preserves custom key artwork and masks until an explicit colour choice and restores them on undo', () => {
  const original = openLevel('Custom key', originalFiles(1)), key = original.objects.find(object => object.type === 4)!;
  const document = applyCommand(original, { label: 'Custom key fixture', patches: [
    numberPatch(original, 'data', key.offset + 8, 5 * 65536 + 2, 4),
    numberPatch(original, 'data', key.offset + 17, 3, 1),
  ] });
  expect(keyColour(document, key.id)).toBeUndefined();
  const history = execute({ document, past: [], future: [] }, setKeyColour(document, key.id, 0));
  expect(keyColour(history.document, key.id)).toBe(0);
  const restored = undo(history).document;
  expect(Buffer.from(restored.files.data).equals(Buffer.from(document.files.data))).toBe(true);
});

it('updates ammunition type and native artwork together, preserves other fields and supports project round trips', () => {
  const document = openLevel('Ammo', originalFiles(1)), object = document.objects.find(object => object.type === 9)!;
  for (const choice of ammunitionChoices) {
    const command = setAmmunitionType(document, object.id, choice.id), history = execute({ document, past: [], future: [] }, command);
    const data = new BinaryReader(history.document.files.data, 'ammo');
    expect(data.i16(object.offset + 18)).toBe(choice.id);
    expect(data.i32(object.offset + 8)).toBe(65536 + ammoFrames[choice.id]!);
    expect(ammunitionType(decodeProject(encodeProject(history.document)), object.id)).toBe(choice.id);
    for (let index = 0; index < document.files.data.length; index++) {
      if (index >= object.offset + 8 && index < object.offset + 12 || index >= object.offset + 18 && index < object.offset + 20) continue;
      if (history.document.files.data[index] !== document.files.data[index]) throw new Error(`Unexpected mutation at ${index}`);
    }
    if (history.past.length) expect(Buffer.from(undo(history).document.files.data).equals(Buffer.from(document.files.data))).toBe(true);
  }
  for (const type of [-1, 3, 5, 6, 8, NaN]) expect(() => setAmmunitionType(document, object.id, type)).toThrow();
  expect(() => setAmmunitionType(document, document.objects.find(object => object.type !== 9)!.id, 0)).toThrow(/ammunition pickup/);
});

it('preserves custom ammunition artwork until a weapon is explicitly selected', () => {
  const original = openLevel('Custom ammo', originalFiles(1)), object = original.objects.find(object => object.type === 9)!;
  const document = applyCommand(original, { label: 'Custom ammo fixture', patches: [numberPatch(original, 'data', object.offset + 8, 65536 + 99, 4)] });
  expect(ammunitionType(document, object.id)).toBeUndefined();
  const history = execute({ document, past: [], future: [] }, setAmmunitionType(document, object.id, 7));
  expect(ammunitionType(history.document, object.id)).toBe(7);
  expect(Buffer.from(undo(history).document.files.data).equals(Buffer.from(document.files.data))).toBe(true);
});
