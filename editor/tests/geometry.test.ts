import { describe, expect, it } from 'vitest';
import { normaliseRoomOutline, signedArea, isConvex, convexParts, buildRoomEdges } from '../src/model/geometry';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';

const square = [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 128 }, { x: 0, z: 128 }];

describe('drawing direction is owned by the editor', () => {
  it('splits larger rooms within mask limits and merges unnecessary triangles automatically', () => {
    const room = Array.from({ length: 20 }, (_, i) => ({ x: Math.round(1000 * Math.cos(i * Math.PI / 10)), z: Math.round(1000 * Math.sin(i * Math.PI / 10)) }));
    const parts = convexParts(room);
    expect(parts.every(part => part.length <= 10 && isConvex(part))).toBe(true);
    expect(parts.length).toBeLessThan(18);
    expect(parts.reduce((area, part) => area + Math.abs(signedArea(part)), 0)).toBe(Math.abs(signedArea(room)));
    expect(convexParts(room.toReversed())).toEqual(parts);
    expect(convexParts(room, 3).every(part => part.length === 3)).toBe(true);
  });
  it('removes straight intermediate clicks before producing strictly convex pieces', () => {
    const room = [{ x: 0, z: 0 }, { x: 64, z: 0 }, { x: 128, z: 0 }, ...square.slice(2)];
    expect(convexParts(room)).toEqual([normaliseRoomOutline(square)]);
  });
  it('makes every start corner and both drawing directions identical', () => {
    const expected = normaliseRoomOutline(square);
    for (let start = 0; start < square.length; start++) {
      const rotated = [...square.slice(start), ...square.slice(0, start)];
      expect(normaliseRoomOutline(rotated)).toEqual(expected);
      expect(normaliseRoomOutline(rotated.toReversed())).toEqual(expected);
    }
    expect(signedArea(expected)).toBeLessThan(0);
  });
  it('accepts clicking the first corner again and removes repeated clicks', () => {
    expect(normaliseRoomOutline([square[0]!, square[0]!, ...square.slice(1), square[0]!])).toEqual(normaliseRoomOutline(square));
  });
  it('automatically pairs portal walls when either adjacent room is drawn in either direction', () => {
    const next = square.map(point => ({ x: point.x + 128, z: point.z }));
    for (const left of [square, square.toReversed()]) for (const right of [next, next.toReversed()]) {
      const edges = buildRoomEdges([{ id: 'a', vertices: left }, { id: 'b', vertices: right }]);
      const portals = edges.filter(edge => edge.neighbour !== null);
      expect(portals).toHaveLength(2);
      expect(portals[0]?.start).toEqual(portals[1]?.end);
      expect(portals[0]?.end).toEqual(portals[1]?.start);
    }
  });
  it('pairs a shorter adjacent wall without requiring matching hand-drawn corner counts', () => {
    const room = [{ x: 128, z: 32 }, { x: 256, z: 32 }, { x: 256, z: 96 }, { x: 128, z: 96 }];
    const edges = buildRoomEdges([{ id: 'large', vertices: square }, { id: 'small', vertices: room.toReversed() }]);
    expect(edges.filter(edge => edge.neighbour !== null)).toHaveLength(2);
    expect(edges.filter(edge => edge.room === 'large')).toHaveLength(6);
  });
  it('splits concave rooms while preserving area, internal portals and direction independence', () => {
    const room = [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 48 }, { x: 48, z: 48 }, { x: 48, z: 128 }, { x: 0, z: 128 }];
    const pieces = convexParts(room);
    expect(pieces.every(isConvex)).toBe(true);
    expect(pieces.reduce((area, piece) => area + Math.abs(signedArea(piece)), 0)).toBe(Math.abs(signedArea(room)));
    expect(convexParts(room.toReversed())).toEqual(pieces);
    expect(buildRoomEdges(pieces.map((vertices, i) => ({ id: String(i), vertices }))).filter(edge => edge.neighbour !== null)).toHaveLength((pieces.length - 1) * 2);
  });
  it('uses the orientation observed in every shipped collision outline', () => {
    let rooms = 0;
    for (let level = 1; level <= 16; level++) {
      const document = openLevel(`Level ${level}`, originalFiles(level));
      for (const zone of document.zones) {
        const outline = zone.boundaryIds.map(id => document.lines[id]!).map(line => ({ x: line.x, z: line.z }));
        expect(signedArea(outline)).toBeLessThan(0);
        expect(signedArea(normaliseRoomOutline(outline.toReversed()))).toBeLessThan(0);
        rooms++;
      }
    }
    expect(rooms).toBe(2496);
  });
  it('gives useful errors for crossing outlines, zero-area rooms and overlapping walls', () => {
    expect(() => normaliseRoomOutline([square[0]!, square[2]!, square[1]!, square[3]!])).toThrow(/crosses itself/);
    expect(() => normaliseRoomOutline([{ x: 0, z: 0 }, { x: 20, z: 0 }, { x: 40, z: 0 }])).toThrow(/form a line/);
    expect(() => buildRoomEdges([{ id: 'a', vertices: square }, { id: 'b', vertices: square.toReversed() }])).toThrow(/overlap/);
  });
});
