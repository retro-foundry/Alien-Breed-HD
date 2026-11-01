import { expect, it } from 'vitest';
import { originalFiles } from './helpers';
import { openLevel } from '../src/model/level';
import { BinaryReader } from '../src/model/binary';
import { applyCommand, execute, numberPatch, redo, undo } from '../src/model/commands';
import { weaponPickupChoices, weaponPickupType, setWeaponPickupType } from '../src/model/objectSettings';
import { decodeProject, encodeProject } from '../src/model/project';

it('has original weapon pickups to exercise across the reference maps', () => {
  expect(Array.from({ length: 16 }, (_, index) => openLevel(`Level ${index + 1}`, originalFiles(index + 1)))
    .flatMap(document => document.objects).filter(object => object.type === 3).length).toBeGreaterThan(0);
});

it.each(Array.from({ length: 16 }, (_, index) => index + 1))('updates complete weapon records in original level %i with the original floor-anchor formula', level => {
  const document = openLevel(`Level ${level}`, originalFiles(level)), original = new BinaryReader(document.files.data, 'original');
  for (const object of document.objects.filter(object => object.type === 3)) {
    for (const choice of weaponPickupChoices) {
      const history = execute({ document, past: [], future: [] }, setWeaponPickupType(document, object.id, choice.id));
      const changed = history.document, data = new BinaryReader(changed.files.data, 'changed');
      expect(weaponPickupType(changed, object.id)).toBe(choice.id);
      expect(changed.objects[object.id]!.y + choice.height * 2).toBe(object.y + original.u8(object.offset + 7));
      expect(data.u8(object.offset + 17)).toBe(choice.id);
      expect(data.i32(object.offset + 8)).toBe(65536 + choice.frame);
      if (history.past.length) {
        const restored = undo(history);
        expect(Buffer.from(restored.document.files.data).equals(Buffer.from(document.files.data))).toBe(true);
        expect(weaponPickupType(redo(restored).document, object.id)).toBe(choice.id);
      }
    }
  }
});

it('retains deliberate height offsets and custom records through saves and undo', () => {
  const original = openLevel('Weapon', originalFiles(1)), object = original.objects.find(object => object.type === 3)!;
  const document = applyCommand(original, { label: 'Custom weapon', patches: [
    numberPatch(original, 'data', object.offset + 4, object.y * 2 + 15, 2),
    numberPatch(original, 'data', object.offset + 8, 65536 + 99, 4),
  ] });
  expect(weaponPickupType(document, object.id)).toBeUndefined();
  const history = execute({ document, past: [], future: [] }, setWeaponPickupType(document, object.id, 1));
  const saved = decodeProject(encodeProject(history.document));
  expect(weaponPickupType(saved, object.id)).toBe(1);
  expect(saved.objects[object.id]!.y).toBe(document.objects[object.id]!.y + original.files.data[object.offset + 7]! - 64);
  expect(Buffer.from(undo(history).document.files.data).equals(Buffer.from(document.files.data))).toBe(true);
});

it('rejects unsupported weapon types, polygon templates and height overflow before changing source', () => {
  const document = openLevel('Weapon', originalFiles(1)), object = document.objects.find(object => object.type === 3)!;
  for (const type of [-1, 2, 4, 5, 7, NaN]) expect(() => setWeaponPickupType(document, object.id, type)).toThrow();
  const polygon = applyCommand(document, { label: 'Polygon fixture', patches: [numberPatch(document, 'data', object.offset + 6, -1, 1)] });
  expect(() => setWeaponPickupType(polygon, object.id, 0)).toThrow(/polygon/);
  const low = applyCommand(document, { label: 'Overflow fixture', patches: [numberPatch(document, 'data', object.offset + 4, -32768, 2), numberPatch(document, 'data', object.offset + 7, 32, 1)] });
  expect(() => setWeaponPickupType(low, object.id, 1)).toThrow(/height/);
});
