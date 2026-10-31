import { describe, expect, it } from 'vitest';
import oracles from './fixtures/original-levels.json';
import { exportLevel, openLevel, zoneContainsPoint } from '../src/model/level';
import { originalFiles } from './helpers';

describe('current runtime map model', () => {
  for (const oracle of oracles.levels) it(`opens and exports level ${oracle.level} without losing any bytes`, () => {
    const files = originalFiles(oracle.level);
    const document = openLevel(`Level ${oracle.level}`, files);
    const header = oracle.files['twolev.bin'].header;
    const graphHeader = oracle.files['twolev.graph.bin'].header;
    expect(document.points.length).toBe((header[7] ?? -1) + 1);
    expect(document.zones.length).toBe(((graphHeader[3] ?? 0) - 16) / 4);
    expect(document.objects.length).toBeGreaterThan(0);
    expect(document.lines.length).toBeGreaterThan(0);
    expect(document.zones.some(zone => zoneContainsPoint(document, zone, document.starts[0]?.x ?? 0, document.starts[0]?.z ?? 0))).toBe(true);
    const exported = exportLevel(document);
    for (const name of ['data', 'graphics', 'clips'] as const) expect(exported[name]).toEqual(files[name]);
  });
  it('rejects invalid offsets and missing input instead of reading arbitrary bytes', () => {
    expect(() => openLevel('Bad level', { data: new Uint8Array(), graphics: new Uint8Array(), clips: new Uint8Array() })).toThrow(/past/);
    const document = openLevel('Level 1', originalFiles(1));
    const invalid = { ...document.files, data: document.files.data.slice() };
    new DataView(invalid.data.buffer).setInt32(22, -100);
    expect(() => openLevel('Bad offset', invalid)).toThrow(/byte -100/);
  });
  it('accepts every original player start and teleport inside its primary room outline', () => {
    for (let level = 1; level <= 16; level++) {
      const document = openLevel(`Level ${level}`, originalFiles(level));
      for (const start of document.starts) expect(zoneContainsPoint(document, document.zones.find(zone => zone.id === start.zone)!, start.x, start.z)).toBe(true);
      for (const zone of document.zones) if (zone.teleport.zone >= 0)
        expect(zoneContainsPoint(document, document.zones.find(item => item.id === zone.teleport.zone)!, zone.teleport.x, zone.teleport.z)).toBe(true);
    }
    const document = openLevel('Level 3', originalFiles(3)), start = document.starts[0]!, zone = document.zones[start.zone]!;
    expect(zone.wallIds.length).toBeGreaterThan(0);
    expect(zoneContainsPoint(document, zone, start.x + 1, start.z)).toBe(true);
    expect(zoneContainsPoint(document, zone, NaN, start.z)).toBe(false);
    expect(zoneContainsPoint(document, zone, start.x, Infinity)).toBe(false);
  });
});
