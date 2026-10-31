import { describe, expect, it } from 'vitest';
import { zipSync, strToU8 } from 'fflate';
import { openLevel, exportLevel } from '../src/model/level';
import { encodeProject, decodeProject, encodeGameLevel, openArchive } from '../src/model/project';
import { applyCommand, setZoneProperty } from '../src/model/commands';
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
});
