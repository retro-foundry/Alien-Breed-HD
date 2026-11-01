import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { setMechanismRange } from '../src/model/mechanismRanges';
import { execute, undo, redo } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';

it('changes native quarter-unit movement destinations atomically across all original levels', () => {
  let checked = 0;
  for (let level = 1; level <= 16; level++) {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    for (const mechanism of document.mechanisms) {
      const top = Math.min(mechanism.top, mechanism.current) - 0.25;
      const bottom = Math.max(mechanism.bottom, mechanism.current) + 0.5;
      if (top < -8192 || bottom > 8191.75 || bottom === 249.75) continue;
      checked++;
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
  }
  expect(checked).toBeGreaterThan(0);
});

it('preserves range edits in projects and rejects invalid ranges before mutation', () => {
  const document = openLevel('Range', originalFiles(1)), mechanism = document.mechanisms.find(item => item.kind === 'door')!;
  const top = Math.min(mechanism.top, mechanism.current) - 0.25, bottom = Math.max(mechanism.bottom, mechanism.current);
  const history = execute({ document, past: [], future: [] }, setMechanismRange(document, 'door', mechanism.id, top, bottom));
  expect(decodeProject(encodeProject(history.document)).mechanisms.find(item => item.kind === 'door' && item.id === mechanism.id)!.top).toBe(top);
  for (const [a, b] of [[1, 0], [mechanism.current + 0.25, bottom + 10], [top, mechanism.current - 0.25], [0.1, bottom], [-8193, bottom], [top, 8192], [top, 249.75], [NaN, bottom]])
    expect(() => setMechanismRange(document, 'door', mechanism.id, a!, b!)).toThrow();
});
