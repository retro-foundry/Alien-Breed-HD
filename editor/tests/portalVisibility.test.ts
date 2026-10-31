import { describe, expect, it } from 'vitest';
import { portalVisibility } from '../src/model/portalVisibility';
import { roomGeometry, originalFiles } from './helpers';
import { openLevel } from '../src/model/level';
import type { RoomOutline } from '../src/model/geometry';
import { visibilityCells } from '../src/model/visibilityImport';
import { BinaryReader } from '../src/model/binary';

const square = (id: number, x: number, z: number): RoomOutline => ({ id: String(id), vertices: [{ x, z }, { x: x + 128, z }, { x: x + 128, z: z + 128 }, { x, z: z + 128 }] });
function visible(rooms: RoomOutline[]) { const geometry = roomGeometry(rooms); return portalVisibility(geometry.points, geometry.rooms).visible; }

describe('exact portal sightline feasibility', () => {
  it('rebuilds all original maps including concave zones, within clip limits and with original wall links', () => {
    for (let level = 1; level <= 16; level++) {
      const document = openLevel(`Level ${level}`, originalFiles(level)), geometry = visibilityCells(document);
      const result = portalVisibility(geometry.points, geometry.rooms, geometry.owners);
      expect(result.visible.size).toBe(document.zones.length);
      let words = 0;
      for (const [source, targets] of result.clips) for (const [target, clips] of targets) {
        if (source !== target) words += 2 + clips.left.length + clips.right.length;
        expect(result.visible.get(source)?.has(target)).toBe(true);
        if (source !== target) {
          const ray = result.witnesses.get(source)!.get(target)!;
          const value = (id: number) => ray.a * geometry.points[id]!.x + ray.b * geometry.points[id]!.z + ray.c;
          expect(clips.left.every(id => value(id) > 0), `Level ${level}: left clipping ${source} → ${target}`).toBe(true);
          expect(clips.right.every(id => value(id) < 0), `Level ${level}: right clipping ${source} → ${target}`).toBe(true);
        }
      }
      expect(words, `Level ${level} clip word offsets`).toBeLessThan(32768);
      const original = new BinaryReader(document.files.clips, 'twolev.clips');
      const count = (original.bytes.length - document.connectTableOffset) / 4;
      for (let point = 0; point < count; point++) {
        expect(result.previous[point], `Level ${level}, previous wall at point ${point}`).toBe(original.i16(document.connectTableOffset + point * 4));
        expect(result.next[point], `Level ${level}, next wall at point ${point}`).toBe(original.i16(document.connectTableOffset + point * 4 + 2));
      }
    }
  });
  it('keeps both openings around an occluder in the common legacy clip window', () => {
    const rooms: RoomOutline[] = [
      { id: '0', vertices: [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 384 }, { x: 0, z: 384 }] },
      square(1, 128, 0), square(2, 128, 256),
      { id: '3', vertices: [{ x: 256, z: 0 }, { x: 384, z: 0 }, { x: 384, z: 384 }, { x: 256, z: 384 }] },
    ];
    const geometry = roomGeometry(rooms), result = portalVisibility(geometry.points, geometry.rooms);
    const clips = result.clips.get(0)!.get(3)!;
    expect(result.visible.get(0)?.has(3)).toBe(true);
    for (const z of [64, 320]) {
      const from = { x: 64, z }, to = { x: 320, z };
      const cross = (id: number) => (to.x - from.x) * (geometry.points[id]!.z - from.z) - (to.z - from.z) * (geometry.points[id]!.x - from.x);
      expect(clips.left.every(id => cross(id) > 0)).toBe(true);
      expect(clips.right.every(id => cross(id) < 0)).toBe(true);
    }
  });
  it('sees a straight corridor in both directions and excludes a disconnected room', () => {
    const result = visible([...Array.from({ length: 5 }, (_, i) => square(i, i * 128, 0)), square(5, 1024, 0)]);
    expect([...result.get(0)!].sort()).toEqual([0, 1, 2, 3, 4]);
    expect([...result.get(4)!].sort()).toEqual([0, 1, 2, 3, 4]);
    expect([...result.get(5)!]).toEqual([5]);
  });
  it('rejects sightlines around two bends even when a walking path exists', () => {
    const rooms = [[0, 0], [128, 0], [256, 0], [256, 128], [256, 256], [128, 256], [0, 256]].map(([x, z], i) => square(i, x!, z!));
    const result = visible(rooms);
    expect(result.get(0)?.has(3)).toBe(true);
    expect(result.get(0)?.has(4)).toBe(false);
    expect(result.get(0)?.has(6)).toBe(false);
  });
  it('preserves the independently witnessed Level 1 sightline that the old traversal lost', () => {
    const document = openLevel('Level 1', originalFiles(1));
    const coordinates = new Map(document.points.map(point => [`${point.x},${point.z}`, point.id]));
    const rooms = document.zones.map(zone => ({ id: zone.id, vertices: zone.boundaryIds.map(id => { const line = document.lines[id]!; return coordinates.get(`${line.x},${line.z}`)!; }), neighbours: zone.boundaryIds.map(id => document.lines[id]!.connect) }));
    const result = portalVisibility(document.points, rooms).visible;
    expect(result.get(71)?.has(91)).toBe(true);
    expect(result.get(91)?.has(71)).toBe(true);
  });
});
