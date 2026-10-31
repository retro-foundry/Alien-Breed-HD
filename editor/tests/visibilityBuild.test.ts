import { describe, expect, it } from 'vitest';
import { buildVisibilityFiles } from '../src/model/visibilityBuild';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';

describe('visibility binary assembly', () => {
  it('builds all sixteen real maps deterministically while retaining game content', () => {
    for (let level = 1; level <= 16; level++) {
      const original = openLevel(`Level ${level}`, originalFiles(level));
      const files = buildVisibilityFiles(original), rebuilt = openLevel(original.name, files);
      expect(rebuilt.points).toEqual(original.points);
      expect(rebuilt.lines).toEqual(original.lines);
      expect(rebuilt.objects).toEqual(original.objects);
      expect(rebuilt.mechanisms).toEqual(original.mechanisms);
      expect(rebuilt.switches).toEqual(original.switches);
      expect(rebuilt.starts).toEqual(original.starts);
      for (const zone of original.zones) {
        const changed = rebuilt.zones.find(z => z.id === zone.id)!;
        expect(changed.boundaryIds).toEqual(zone.boundaryIds); expect(changed.wallIds).toEqual(zone.wallIds);
        expect(changed.lower).toEqual(zone.lower); expect(changed.upper).toEqual(zone.upper);
        expect(changed.floor).toBe(zone.floor); expect(changed.ceiling).toBe(zone.ceiling);
        expect(changed.upperFloor).toBe(zone.upperFloor); expect(changed.upperCeiling).toBe(zone.upperCeiling);
        expect(changed.water).toBe(zone.water); expect(changed.teleport).toEqual(zone.teleport);
      }
      expect(buildVisibilityFiles(rebuilt), `Level ${level} build must not accumulate old data`).toEqual(files);
    }
  });
});
