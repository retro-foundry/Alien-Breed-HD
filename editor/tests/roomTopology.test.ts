import { describe, expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { importRoomSource, beginRoomDraft, commitRoomDraft, setDraftCorners, removeSourceRooms } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { visibilityCells } from '../src/model/visibilityImport';
import { signedArea } from '../src/model/geometry';

describe('source room topology compilation', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('retains level %i portal cells and point identities', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    const topology = compileRoomTopology(document, importRoomSource(document)), original = visibilityCells(document);
    expect(topology.points).toEqual(document.points.map(({ x, z }) => ({ x, z })));
    expect(topology.cells.map(({ id, vertices, neighbours }) => ({ id, vertices, neighbours }))).toEqual(original.rooms);
    expect(topology.cells.map(cell => cell.originZone)).toEqual([...original.owners.values()]);
    expect(topology.sourceCells.size).toBe(document.zones.length);
  });
  it('turns a concave new outline into joined convex cells without changing imported rooms', () => {
    const document = openLevel('Level 1', originalFiles(1)), source = importRoomSource(document);
    const outline = [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20064 }, { x: 20064, z: 20064 }, { x: 20064, z: 20256 }, { x: 20000, z: 20256 }];
    const build = (vertices: typeof outline) => compileRoomTopology(document, commitRoomDraft(source,
      setDraftCorners(beginRoomDraft(document, source, 0), vertices)).after);
    const topology = build(outline), id = `room-${source.nextId}`, indices = topology.sourceCells.get(id)!;
    expect(build(outline.toReversed())).toEqual(topology);
    expect(indices.length).toBeGreaterThan(1);
    const cells = indices.map(index => topology.cells[index]!);
    expect(cells.every(cell => cell.originZone === null && cell.recordZone === 0)).toBe(true);
    expect(cells.reduce((area, cell) => area + signedArea(cell.vertices.map(id => topology.points[id]!)), 0)).toBe(-Math.abs(signedArea(outline)));
    expect(cells.flatMap(cell => cell.neighbours).filter(id => id >= 0)).toHaveLength((cells.length - 1) * 2);
    expect(topology.points.slice(0, document.points.length)).toEqual(document.points.map(({ x, z }) => ({ x, z })));
  });
  it('refuses empty geometry and invalid source references before producing output', () => {
    const document = openLevel('Level 1', originalFiles(1)), source = importRoomSource(document);
    const empty = removeSourceRooms(source, source.rooms.map(room => room.id)).after;
    expect(() => compileRoomTopology(document, empty)).toThrow(/Create a room/);
    expect(() => compileRoomTopology(document, { ...source, rooms: [{ ...source.rooms[0]!, recordZone: -1 }] })).toThrow(/missing surface record/);
    expect(() => compileRoomTopology(document, { ...source, rooms: [source.rooms[0]!, { ...source.rooms[0]!, id: 'duplicate-origin' }] })).toThrow(/duplicate original zone/);
  });
  it('joins a shorter new wall automatically and closes portals when a neighbouring source room is deleted', () => {
    const document = openLevel('Level 1', originalFiles(1)), imported = importRoomSource(document);
    const empty = removeSourceRooms(imported, imported.rooms.map(room => room.id)).after;
    const add = (current: typeof empty, vertices: { x: number; z: number }[]) => commitRoomDraft(current,
      setDraftCorners(beginRoomDraft(document, current, 0), vertices)).after;
    const left = [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }];
    const right = [{ x: 20128, z: 20032 }, { x: 20256, z: 20032 }, { x: 20256, z: 20096 }, { x: 20128, z: 20096 }];
    const source = add(add(empty, left), right.toReversed());
    const joined = compileRoomTopology(document, source);
    expect(joined.cells).toHaveLength(2);
    expect(joined.cells[0]!.vertices).toHaveLength(6);
    expect(joined.cells.flatMap(cell => cell.neighbours).filter(id => id >= 0)).toHaveLength(2);
    const deleted = removeSourceRooms(source, [source.rooms[1]!.id]).after;
    expect(compileRoomTopology(document, deleted).cells[0]!.neighbours.every(id => id === -1)).toBe(true);
  });
});
