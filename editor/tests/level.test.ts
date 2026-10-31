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
});
