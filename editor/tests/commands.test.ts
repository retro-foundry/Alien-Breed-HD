import { describe, expect, it } from 'vitest';
import { openLevel, exportLevel } from '../src/model/level';
import { execute, undo, redo, setZoneProperty, setStart, setTeleport, moveObject, removeObject, setMechanismRules, type History } from '../src/model/commands';
import { originalFiles } from './helpers';

describe('transactional map edits', () => {
  it('rejects teleporter destinations outside their room without losing the previous destination', () => {
    const document = openLevel('Level 9', originalFiles(9));
    const start = document.starts[0]!;
    const history = execute({ document, past: [], future: [] }, setTeleport(document, 0, start.zone, start.x, start.z));
    expect(history.document.zones[0]?.teleport).toEqual({ zone: start.zone, x: start.x, z: start.z });
    expect(() => setTeleport(history.document, 0, start.zone, 32000, 32000)).toThrow(/inside/);
    expect(history.document.zones[0]?.teleport.zone).toBe(start.zone);
    expect(exportLevel(undo(history).document)).toEqual(originalFiles(9));
  });
  it('keeps the complete undo chain beyond the old 100-action limit', () => {
    const original = originalFiles(1);
    let history: History = { document: openLevel('Level 1', original), past: [], future: [] };
    for (let i = 0; i < 250; i++) history = execute(history, setZoneProperty(history.document, 3, 'brightness', i % 2 === 0 ? 6 : 7));
    expect(history.past).toHaveLength(250);
    for (let i = 0; i < 250; i++) history = undo(history);
    expect(exportLevel(history.document)).toEqual(original);
    for (let i = 0; i < 250; i++) history = redo(history);
    expect(history.document.zones[3]?.brightness).toBe(7);
  });
  it('changes only the intended brightness word and restores the original compressed file on undo', () => {
    const files = originalFiles(1), document = openLevel('Level 1', files);
    let history: History = { document, past: [], future: [] };
    const zone = document.zones[3]!;
    history = execute(history, setZoneProperty(document, 3, 'brightness', 8));
    expect(history.document.zones[3]?.brightness).toBe(8);
    const changed = [...history.document.files.data].flatMap((byte, i) => byte === document.files.data[i] ? [] : [i]);
    expect(changed.every(i => i === zone.offset + 22 || i === zone.offset + 23)).toBe(true);
    expect(history.document.files.graphics).toBe(document.files.graphics);
    history = undo(history);
    expect(exportLevel(history.document)).toEqual(files);
    expect(redo(history).document.zones[3]?.brightness).toBe(8);
  });
  it('preserves command history when an invalid edit is rejected', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const history: History = { document, past: [], future: [] };
    expect(() => setZoneProperty(document, 3, 'brightness', 100000)).toThrow(/whole number/);
    expect(() => setStart(document, 1, 32000, 32000, 3)).toThrow(/inside/);
    expect(history.document).toBe(document);
    expect(history.past).toHaveLength(0);
  });
  it('synchronizes player spawn and its multiplayer avatar', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const start = document.starts[0]!;
    const history = execute({ document, past: [], future: [] }, setStart(document, 1, start.x + 1, start.z, start.zone));
    expect(history.document.starts[0]?.x).toBe(start.x + 1);
    const avatar = new DataView(document.files.data.buffer).getInt32(46);
    const cid = new DataView(document.files.data.buffer).getInt16(avatar);
    expect(new DataView(history.document.files.data.buffer).getInt32(document.objectPointsOffset + cid * 8)).toBe((start.x + 1) * 65536);
    expect(exportLevel(undo(history).document)).toEqual(originalFiles(1));
  });
  it('moves objects within actual collision bounds and makes deletion undoable', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const object = document.objects.find(object => object.zone >= 0)!;
    let history = execute({ document, past: [], future: [] }, moveObject(document, object.id, object.x, object.z, object.zone));
    history = execute(history, removeObject(history.document, object.id));
    expect(history.document.objects.find(candidate => candidate.id === object.id)?.zone).toBe(-1);
    expect(undo(history).document.objects.find(candidate => candidate.id === object.id)?.zone).toBe(object.zone);
  });
  it('edits original door rules without changing texture or motion records', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const door = document.mechanisms.find(mechanism => mechanism.kind === 'door')!;
    const history = execute({ document, past: [], future: [] }, setMechanismRules(document, 'door', door.id, 0x100, 1, 0));
    const edited = history.document.mechanisms.find(mechanism => mechanism.kind === 'door' && mechanism.id === door.id)!;
    expect(edited.conditions).toBe(0x100);
    expect(edited.openRule).toBe(1);
    expect(edited.walls).toEqual(door.walls);
    expect(edited.top).toBe(door.top);
    expect(exportLevel(undo(history).document)).toEqual(originalFiles(1));
  });
});
