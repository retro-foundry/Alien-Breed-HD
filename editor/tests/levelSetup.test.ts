import { describe, expect, it } from 'vitest';
import { unzipSync, zipSync, strToU8 } from 'fflate';
import { encodeLevelSetup, decodeLevelSetup } from '../src/model/levelSetup';
import { openLevel } from '../src/model/level';
import { applyCommand, execute, undo, redo, setExitZone, setZoneProperty } from '../src/model/commands';
import { encodeProject, decodeProject, encodeGameLevel, openArchive } from '../src/model/project';
import { originalFiles } from './helpers';

describe('portable port exit setup', () => {
  it('encodes the exact checked sidecar format and rejects malformed inputs', () => {
    expect(encodeLevelSetup(132)).toEqual(new Uint8Array([65, 66, 69, 88, 0, 1, 0, 132]));
    for (const zone of [null, -1, 0, 132, 255]) expect(decodeLevelSetup(encodeLevelSetup(zone))).toBe(zone);
    expect(() => encodeLevelSetup(256)).toThrow(/Exit zone/);
    expect(() => encodeLevelSetup(0.5)).toThrow(/Exit zone/);
    expect(() => decodeLevelSetup(new Uint8Array(8))).toThrow(/Unsupported/);
    expect(() => decodeLevelSetup(encodeLevelSetup(0).subarray(0, 7))).toThrow(/Unsupported/);
    const future = encodeLevelSetup(0); future[5] = 2;
    expect(() => decodeLevelSetup(future)).toThrow(/Unsupported/);
  });
  it('retains exit edits through history, binary edits, project recovery and game exports', () => {
    const document = openLevel('Reactor', originalFiles(1), 8), initial = { document, past: [], future: [] };
    let history = execute(initial, setExitZone(document, 0));
    expect(history.document.exitZone).toBe(0);
    history = undo(history); expect(history.document.exitZone).toBeNull();
    history = redo(history); expect(history.document.exitZone).toBe(0);
    const edited = applyCommand(history.document, setZoneProperty(history.document, 3, 'brightness', 6));
    expect(edited.exitZone).toBe(0);
    expect(decodeProject(encodeProject(edited)).exitZone).toBe(0);
    const game = encodeGameLevel(edited), entries = unzipSync(game);
    expect(decodeLevelSetup(entries['levels/level_i/twolev.cfg']!)).toBe(0);
    expect(openArchive(game, 'level.zip')[0]!.exitZone).toBe(0);
    expect(() => setExitZone(document, 250)).toThrow(/existing exit zone/);
    expect(() => applyCommand(history.document, setExitZone(document, 1))).toThrow(/exit zone changed/);
    expect(execute(history, setExitZone(history.document, 0))).toBe(history);
  });
  it('keeps v2 slots while migrating unspecified exits and rejects dangling saved exits', () => {
    const document = openLevel('Old setup', originalFiles(1), 8), entries = unzipSync(encodeProject(document));
    entries['alien-breed-project.json'] = strToU8(JSON.stringify({ format: 'alien-breed-level-project', version: 2, name: 'Old setup', playtestSlot: 8 }));
    expect(decodeProject(zipSync(entries))).toMatchObject({ playtestSlot: 8, exitZone: null });
    entries['alien-breed-project.json'] = strToU8(JSON.stringify({ format: 'alien-breed-level-project', version: 3, name: 'Bad setup', playtestSlot: 8, exitZone: 250 }));
    expect(() => decodeProject(zipSync(entries))).toThrow(/existing exit zone/);
  });
  it('rejects corrupt imported setup and always exports an explicit reset for original exits', () => {
    const document = openLevel('original', originalFiles(1));
    const entries = unzipSync(encodeGameLevel(document));
    expect(decodeLevelSetup(entries['levels/level_a/twolev.cfg']!)).toBeNull();
    entries['levels/level_a/twolev.cfg'] = new Uint8Array(8);
    expect(() => openArchive(zipSync(entries), 'damaged.zip')).toThrow(/Unsupported or damaged/);
  });
});
