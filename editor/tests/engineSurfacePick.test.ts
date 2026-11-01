import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { resolveEngineSurface } from '../src/model/engineSurfacePick';
import { originalFiles } from './helpers';

it('resolves actual native wall and polygon identities on both floors of all sixteen levels', () => {
  const counts = new Map<number, number>();
  for (let level = 1; level <= 16; level++) {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    for (const zone of document.zones) for (const upper of [false, true]) for (const record of upper ? zone.upper : zone.lower) {
      const pick = { offset: record.offset, upper, type: record.type }, result = resolveEngineSurface(document, zone.id, pick);
      if (![0, 13, 5, 1, 2, 7, 8, 9, 10, 11].includes(record.type)) { expect(result).toBeUndefined(); continue; }
      counts.set(record.type, (counts.get(record.type) ?? 0) + 1);
      expect(result).toMatchObject({ ...pick, zone: zone.id });
      expect(result!.texture).toBe(record.texture);
      expect(result!.height).toBe(record.type === 1 ? upper ? zone.upperFloor : zone.floor
        : record.type === 2 ? upper ? zone.upperCeiling : zone.ceiling : record.height);
    }
  }
  expect([...counts.keys()].sort((a, b) => a - b)).toEqual([0, 1, 2, 7]);
});

it('rejects absent, mismatched and malformed provenance instead of treating offsets as editable source', () => {
  const document = openLevel('Native picks', originalFiles(1)), zone = document.zones[0]!, record = zone.lower.find(record => record.wall)!;
  const pick = { offset: record.offset, upper: false, type: record.type };
  expect(resolveEngineSurface(document, zone.id, pick)).toBeDefined();
  for (const invalid of [null, undefined, { ...pick, offset: -1 }, { ...pick, offset: .5 }, { ...pick, type: 99 }, { ...pick, upper: true }])
    expect(resolveEngineSurface(document, zone.id, invalid)).toBeUndefined();
  expect(resolveEngineSurface(document, -1, pick)).toBeUndefined();
  expect(resolveEngineSurface(document, 10000, pick)).toBeUndefined();
});
