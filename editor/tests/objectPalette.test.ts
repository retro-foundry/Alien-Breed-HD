import { roomSettings } from '../src/model/roomHeights';
import { editableObjects } from '../src/model/sourceObjects';
import { expect, it } from 'vitest';
import { originalObjectPalette, placePaletteObject, duplicatePaletteChoice } from '../src/model/objectPalette';
import { originalFiles } from './helpers';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { applyCommand, execute, undo, redo } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';
const files = Array.from({ length: 16 }, (_, i) => originalFiles(i + 1));
const palette = originalObjectPalette(files.map(f => f.data), files.map(f => f.graphics));
it('names distinct native variants and deduplicates level instances', () => {
  expect(new Set(palette.map(c => c.id)).size).toBe(palette.length);
  expect(palette.filter(c => c.name === 'Barrel')).toHaveLength(1);
  expect(palette.map(c => c.name)).toEqual(expect.arrayContaining(['Green key', 'Red key', 'Yellow key', 'Blue key', 'Shotgun ammo']));
  expect(palette.every(c => !c.name.includes('Object ') && Number.isFinite(c.floorOffset))).toBe(true);
});
it('places independent records with exact duplicate height, history and portable saves', () => {
  const { document, from } = roomHeightFixture(), choice = palette.find(c => c.name === 'Blue key')!;
  const first = placePaletteObject(document, choice, from.id, 20128, 20192, false);
  const changed = applyCommand(document, first.command), duplicate = duplicatePaletteChoice(changed, first.id);
  const second = placePaletteObject(changed, duplicate, from.id, 20144, 20192, false);
  const result = applyCommand(changed, second.command);
  expect(result.objects[first.id]!.y).toBe(result.objects[second.id]!.y);
  expect(result.objects[first.id]!.cid).not.toBe(result.objects[second.id]!.cid);
  expect(decodeProject(encodeProject(result)).files.data).toEqual(result.files.data);
  let history = execute({ document, past: [], future: [] }, first.command);
  history = undo(history); expect(history.document.files.data).toEqual(document.files.data);
  history = redo(history); expect(history.document.files.data).toEqual(changed.files.data);
  expect(() => placePaletteObject(document, choice, from.id, -30000, -30000, false)).toThrow();
  expect(() => placePaletteObject(document, choice, 'missing-room', 0, 0, false)).toThrow('Click inside');
});

it('each library variant follows both authored floors using its native anchor', () => {
  const { document, from } = roomHeightFixture(twoStoreyHeightTemplate());
  const settings = roomSettings(document, from);
  for (const upper of [false, true]) for (const choice of palette) {
    const proposal = placePaletteObject(document, choice, from.id, 20128, 20192, upper);
    const changed = applyCommand(document, proposal.command), actor = editableObjects(changed).find(o => o.id === proposal.id)!;
    expect(actor.upper).toBe(upper);
    expect(actor.y).toBe(Math.round((settings[upper ? 'upperFloor' : 'floor'] + choice.floorOffset) * 2) / 2);
    expect(actor.room).toBe(from.id);
  }
  const lower = roomHeightFixture();
  expect(() => placePaletteObject(lower.document, palette[0]!, lower.from.id, 20128, 20192, true)).toThrow('no upper floor');
});
