import { expect, it } from 'vitest';
import { openLevel, zoneContainsPoint } from '../src/model/level';
import { originalFiles } from './helpers';

it('uses the closed room outline rather than adjoining secondary collision fences', () => {
  const document = openLevel('Level 1', originalFiles(1));
  expect(zoneContainsPoint(document, document.zones[1]!, -482.56, 278.08)).toBe(true);
  expect(zoneContainsPoint(document, document.zones[0]!, -482.56, 278.08)).toBe(false);
  expect(zoneContainsPoint(document, document.zones[1]!, NaN, 278.08)).toBe(false);
});

it.each(Array.from({ length: 16 }, (_, index) => index + 1))('keeps original starts and teleport destinations inside their room in level %i', level => {
  const document = openLevel(`Level ${level}`, originalFiles(level));
  for (const start of document.starts)
    expect(zoneContainsPoint(document, document.zones[start.zone]!, start.x, start.z)).toBe(true);
  for (const zone of document.zones) if (zone.teleport.zone >= 0)
    expect(zoneContainsPoint(document, document.zones[zone.teleport.zone]!, zone.teleport.x, zone.teleport.z)).toBe(true);
});
