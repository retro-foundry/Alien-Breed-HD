import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { setMechanismRange } from '../src/model/mechanismRanges';
import { execute, undo, redo, applyCommand, numberPatch, setMechanismRules } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';

it('has original controller ranges to exercise across the reference maps', () => {
  expect(Array.from({ length: 16 }, (_, index) => openLevel(`Level ${index + 1}`, originalFiles(index + 1)))
    .flatMap(document => document.mechanisms).filter(mechanism => {
      const top = mechanism.direction === 0 && mechanism.current !== mechanism.bottom ? mechanism.current : Math.min(mechanism.top, mechanism.current) - 0.25;
      const bottom = mechanism.direction === 0 && mechanism.current === mechanism.bottom ? mechanism.current : Math.max(mechanism.bottom, mechanism.current) + 0.5;
      return top >= -8192 && bottom <= 8191.75 && bottom !== 249.75;
    }).length).toBeGreaterThan(0);
});

it.each(Array.from({ length: 16 }, (_, index) => index + 1))('changes native quarter-unit movement destinations atomically in original level %i', level => {
  const document = openLevel(`Level ${level}`, originalFiles(level));
  for (const mechanism of document.mechanisms) {
    const top = mechanism.direction === 0 && mechanism.current !== mechanism.bottom ? mechanism.current : Math.min(mechanism.top, mechanism.current) - 0.25;
    const bottom = mechanism.direction === 0 && mechanism.current === mechanism.bottom ? mechanism.current : Math.max(mechanism.bottom, mechanism.current) + 0.5;
    if (top < -8192 || bottom > 8191.75 || bottom === 249.75) continue;
    let history = execute({ document, past: [], future: [] }, setMechanismRange(document, mechanism.kind, mechanism.id, top, bottom));
    expect(history.document.mechanisms.find(item => item.kind === mechanism.kind && item.id === mechanism.id)).toEqual({ ...mechanism, top, bottom });
    expect(history.document.files.data).toBe(document.files.data);
    expect(history.document.files.clips).toBe(document.files.clips);
    const before = document.files.graphics, after = history.document.files.graphics;
    for (let offset = 0; offset < before.length; offset++) if (offset < mechanism.offset || offset >= mechanism.offset + 4)
      if (before[offset] !== after[offset]) throw new Error(`Unrelated graphics change at ${offset}`);
    history = undo(history); expect(Buffer.from(history.document.files.graphics).equals(Buffer.from(before))).toBe(true);
    history = redo(history); expect(history.document.mechanisms.find(item => item.kind === mechanism.kind && item.id === mechanism.id)!.top).toBe(top);
  }
});

it('rejects stranding a gated idle controller while retaining ungated door and in-transit behaviour', () => {
  const original = openLevel('Range', originalFiles(1)), mechanism = original.mechanisms.find(item => item.kind === 'door' && item.direction === 0)!;
  const document = applyCommand(original, { label: 'Gated door fixture', patches: [numberPatch(original, 'graphics', mechanism.offset + 14, 1, 2)] });
  const top = Math.min(mechanism.top, mechanism.current) - 1, bottom = Math.max(mechanism.bottom, mechanism.current) + 1;
  expect(() => setMechanismRange(document, 'door', mechanism.id, top, bottom)).toThrow(/Keep one destination/);
  const moving = applyCommand(document, { label: 'In-transit fixture', patches: [numberPatch(document, 'graphics', mechanism.offset + 6, -4, 2)] });
  expect(() => setMechanismRange(moving, 'door', mechanism.id, top, bottom)).not.toThrow();
  const ungated = applyCommand(document, { label: 'Ungated door fixture', patches: [numberPatch(document, 'graphics', mechanism.offset + 14, 0, 2)] });
  expect(() => setMechanismRange(ungated, 'door', mechanism.id, top, bottom)).not.toThrow();
  const intermediate = applyCommand(ungated, setMechanismRange(ungated, 'door', mechanism.id, top, bottom));
  expect(() => setMechanismRules(intermediate, 'door', mechanism.id, 1, mechanism.openRule, mechanism.closeRule)).toThrow(/Keep one destination/);
  expect(() => setMechanismRules(intermediate, 'door', mechanism.id, 0, mechanism.openRule, mechanism.closeRule)).not.toThrow();
});

it('preserves range edits in projects and rejects invalid ranges before mutation', () => {
  const document = openLevel('Range', originalFiles(1)), mechanism = document.mechanisms.find(item => item.kind === 'door')!;
  const top = Math.min(mechanism.top, mechanism.current) - 0.25, bottom = Math.max(mechanism.bottom, mechanism.current);
  const history = execute({ document, past: [], future: [] }, setMechanismRange(document, 'door', mechanism.id, top, bottom));
  expect(decodeProject(encodeProject(history.document)).mechanisms.find(item => item.kind === 'door' && item.id === mechanism.id)!.top).toBe(top);
  for (const [a, b] of [[1, 0], [mechanism.current + 0.25, bottom + 10], [top, mechanism.current - 0.25], [0.1, bottom], [-8193, bottom], [top, 8192], [top, 249.75], [NaN, bottom]])
    expect(() => setMechanismRange(document, 'door', mechanism.id, a!, b!)).toThrow();
});
