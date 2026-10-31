import { describe, expect, it } from 'vitest';
import { zoneOrderMask, classifyPortalPoints, clipsSeparateRooms, runtimeOrderingMask, type OrderingRoom } from '../src/model/visibilityGeometry';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import type { Position } from '../src/model/geometry';
import fixtures from './fixtures/original-geometry.json';

const positions = (points: number[][]): Position[] => points.map(p => ({ x: p[0]!, z: p[1]! }));

describe('original 68020 geometry contracts', () => {
  it('independently rebuilds every current-format ordering mask in all sixteen shipped levels', () => {
    let compared = 0;
    for (let level = 1; level <= 16; level++) {
      const document = openLevel(`Level ${level}`, originalFiles(level));
      const rooms = new Map(document.zones.map(zone => [zone.id, {
        id: zone.id,
        outline: zone.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })),
        neighbours: zone.boundaryIds.map(id => document.lines[id]!.connect),
      } satisfies OrderingRoom]));
      for (const zone of document.zones) {
        const source = rooms.get(zone.id)!, visible = new Set(zone.visibleZones);
        for (const entry of zone.visibility) {
          const target = rooms.get(entry.zone)!;
          expect(runtimeOrderingMask(source, target, visible), `Level ${level}: zone ${zone.id} → ${entry.zone}`).toBe(entry.mask);
          compared++;
        }
      }
    }
    expect(compared).toBe(49797);
  });
  it('reproduces every original CALCORDER mask, including signed-word arithmetic', () => {
    for (const fixture of fixtures.masks)
      expect(zoneOrderMask(positions(fixture.source), positions(fixture.target), fixture.walls)).toBe(fixture.mask);
  });
  it('reproduces every original BETPTS classification and collapsed portal case', () => {
    for (const fixture of fixtures.classifications)
      expect(classifyPortalPoints(positions(fixture.portal), positions(fixture.points), fixture.used, fixture.mode as 0 | 1 | 2)).toEqual(fixture.result);
  });
  it('reproduces the original separating-line rejection of room pairs', () => {
    for (const fixture of fixtures.separations)
      expect(clipsSeparateRooms(positions(fixture.source), positions(fixture.target), positions(fixture.left), positions(fixture.right))).toBe(fixture.result);
  });
});
