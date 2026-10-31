import { describe, expect, it } from 'vitest';
import { zipSync, strToU8, unzipSync, strFromU8 } from 'fflate';
import { openLevel, exportLevel } from '../src/model/level';
import { encodeProject, decodeProject, encodeGameLevel, openArchive } from '../src/model/project';
import { applyCommand, setZoneProperty, setPlaytestSlot, execute, undo, redo } from '../src/model/commands';
import { originalFiles } from './helpers';

describe('portable project files', () => {
  it('round trips edited binary records and project name without changing unrelated files', () => {
    const original = openLevel('My reactor level', originalFiles(1));
    const edited = applyCommand(original, setZoneProperty(original, 3, 'brightness', 6));
    const restored = decodeProject(encodeProject(edited));
    expect(restored.name).toBe(edited.name);
    expect(restored.zones[3]?.brightness).toBe(6);
    expect(exportLevel(restored)).toEqual(exportLevel(edited));
  });
  it('exports a level ZIP containing the exact original game bytes before edits', () => {
    const original = originalFiles(1);
    const opened = openArchive(encodeGameLevel(openLevel('Level 1', original)), 'level.zip');
    expect(opened).toHaveLength(1);
    expect(opened[0]?.files).toEqual(original);
  });
  it('finds separate levels in a game release archive', () => {
    const one = originalFiles(1), two = originalFiles(2);
    const bytes = zipSync({ 'data/levels/level_a/twolev.bin': one.data, 'data/levels/level_a/twolev.graph.bin': one.graphics,
      'data/levels/level_a/twolev.clips': one.clips, 'data/levels/level_b/twolev.bin': two.data,
      'data/levels/level_b/twolev.graph.bin': two.graphics, 'data/levels/level_b/twolev.clips': two.clips });
    expect(openArchive(bytes, 'game.zip').map(project => project.name)).toEqual(['Level 1', 'Level 2']);
  });
  it('reports incomplete sets, corrupt archives and unsupported project versions', () => {
    expect(() => openArchive(new Uint8Array([1, 2, 3]), 'broken.zip')).toThrow(/Could not open/);
    expect(() => openArchive(zipSync({ 'twolev.bin': originalFiles(1).data }), 'incomplete.zip')).toThrow(/all three/);
    expect(() => openArchive(zipSync({ 'alien-breed-project.json': strToU8('{"format":"alien-breed-level-project","version":999}') }), 'future.ab3d')).toThrow(/not supported/);
  });
  it('saves and restores the selected target slot independently of the project name', () => {
    const original = openLevel('Level 16', originalFiles(1), 2);
    let history = execute({ document: original, past: [], future: [] }, setPlaytestSlot(original, 8));
    const edited = applyCommand(history.document, setZoneProperty(history.document, 3, 'brightness', 6));
    expect(edited.playtestSlot).toBe(8);
    expect(decodeProject(encodeProject(edited)).playtestSlot).toBe(8);
    history = undo(history); expect(history.document.playtestSlot).toBe(2);
    history = redo(history); expect(history.document.playtestSlot).toBe(8);
    expect(exportLevel(history.document)).toEqual(exportLevel(original));
    expect(execute(history, setPlaytestSlot(history.document, 8))).toBe(history);
    expect(() => applyCommand(history.document, setPlaytestSlot(original, 7))).toThrow(/slot changed/);
    expect(() => setPlaytestSlot(original, 16)).toThrow(/Game level slot/);
    expect(() => setPlaytestSlot(original, -1)).toThrow(/Game level slot/);
    expect(() => setPlaytestSlot(original, 1.5)).toThrow(/Game level slot/);
  });
  it('places game exports in the selected actual level directory with installation instructions', () => {
    const document = openLevel('Custom reactor', originalFiles(1), 8), bytes = encodeGameLevel(document), entries = unzipSync(bytes);
    expect(entries['levels/level_i/twolev.bin']).toEqual(originalFiles(1).data);
    expect(strFromU8(entries['INSTALL.txt']!)).toContain('level slot 9');
    const candidate = openArchive(bytes, 'reactor.zip')[0]!;
    expect(candidate.playtestSlot).toBe(8);
  });
  it('migrates v1 projects explicitly and rejects malformed saved slots', () => {
    const document = openLevel('Level 16', originalFiles(1)), entries = unzipSync(encodeProject(document));
    entries['alien-breed-project.json'] = strToU8(JSON.stringify({ format: 'alien-breed-level-project', version: 1, name: 'Level 16' }));
    expect(decodeProject(zipSync(entries)).playtestSlot).toBe(0);
    for (const slot of [null, '8', -1, 16, 0.5]) {
      entries['alien-breed-project.json'] = strToU8(JSON.stringify({ format: 'alien-breed-level-project', version: 2, name: 'Test', playtestSlot: slot }));
      expect(() => decodeProject(zipSync(entries))).toThrow(/invalid game level slot/);
    }
  });
});
