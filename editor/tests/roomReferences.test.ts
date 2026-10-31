import { describe, expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { importRoomSource, removeSourceRooms, beginRoomDraft, setDraftCorners, commitRoomDraft } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { assertRoomReferences, cellContainsPoint, planRoomReferences } from '../src/model/roomReferences';
import { originalFiles } from './helpers';

describe('source room dependency remapping', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('resolves level %i dependencies to their original logical rooms after convex splitting', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level)), source = importRoomSource(document), topology = compileRoomTopology(document, source);
    const plan = planRoomReferences(document, source, topology);
    expect(plan.diagnostics.filter(item => item.severity === 'error')).toEqual([]);
    expect(() => assertRoomReferences(plan)).not.toThrow();
    for (const start of document.starts) {
      const cell = plan.starts.find(item => item.player === start.player)!.cell!;
      expect(topology.cells[cell]!.originZone).toBe(start.zone);
      expect(cellContainsPoint(topology, topology.cells[cell]!, start)).toBe(true);
    }
    for (const object of document.objects) {
      const cell = plan.objects.get(object.id)!;
      if (object.zone < 0) expect(cell).toBe(object.zone);
      else expect(topology.cells[cell]!.originZone).toBe(object.zone);
    }
    for (const zone of document.zones) {
      const cell = plan.teleports.get(zone.id)!;
      if (zone.teleport.zone < 0) expect(cell).toBe(zone.teleport.zone);
      else expect(topology.cells[cell]!.originZone).toBe(zone.teleport.zone);
    }
    for (const item of document.switches) {
      const cell = plan.switches.get(item.id)!;
      if (item.zone < 0) expect(cell).toBe(item.zone);
      else expect(topology.cells[cell]!.originZone).toBe(item.zone);
    }
    const warnings: Record<number, number> = { 1: 1, 2: 1, 6: 2, 9: 1, 14: 1 };
    expect(plan.diagnostics.filter(item => item.severity === 'warning')).toHaveLength(warnings[level] ?? 0);
    expect(plan.mechanisms).toHaveLength(document.mechanisms.length);
  });
  it('reports dependencies on a deleted room instead of disabling or moving them', () => {
    const document = openLevel('Level 1', originalFiles(1)), source = importRoomSource(document), start = document.starts[0]!;
    const id = source.rooms.find(room => room.originZone === start.zone)!.id;
    const deleted = removeSourceRooms(source, [id]).after, plan = planRoomReferences(document, deleted);
    expect(plan.starts[0]!.cell).toBeNull();
    expect(() => assertRoomReferences(plan)).toThrow(/Player 1 start.*deleted/);
    expect(document.starts[0]).toEqual(start);
  });
  it('blocks edited geometry that leaves a player start outside its room', () => {
    const document = openLevel('Level 1', originalFiles(1)), source = importRoomSource(document), start = document.starts[0]!;
    const room = source.rooms.find(room => room.originZone === start.zone)!;
    const edited = commitRoomDraft(source, setDraftCorners(beginRoomDraft(document, source, room.recordZone, room.id),
      [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }])).after;
    const plan = planRoomReferences(document, edited);
    expect(plan.starts[0]!.cell).toBeNull();
    expect(() => assertRoomReferences(plan)).toThrow(/outside the remaining geometry/);
  });
  it('includes shared boundaries and fractional positions without rounding a point into a room', () => {
    const document = openLevel('Level 1', originalFiles(1)), source = importRoomSource(document);
    const empty = removeSourceRooms(source, source.rooms.map(room => room.id)).after;
    const created = commitRoomDraft(empty, setDraftCorners(beginRoomDraft(document, empty, 0),
      [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 128 }, { x: 0, z: 128 }])).after;
    const topology = compileRoomTopology(document, created), cell = topology.cells[0]!;
    expect(cellContainsPoint(topology, cell, { x: 0, z: 64.5 })).toBe(true);
    expect(cellContainsPoint(topology, cell, { x: -0.25, z: 64.5 })).toBe(false);
    expect(cellContainsPoint(topology, cell, { x: NaN, z: 64 })).toBe(false);
  });
  it('reports dangling teleporter destinations and deleted controlled rooms specifically', () => {
    const document = openLevel('Level 9', originalFiles(9)), source = importRoomSource(document);
    const teleporter = document.zones.find(zone => zone.teleport.zone >= 0 && zone.teleport.zone !== zone.id)!;
    const room = source.rooms.find(room => room.originZone === teleporter.teleport.zone)!;
    const plan = planRoomReferences(document, removeSourceRooms(source, [room.id]).after);
    expect(plan.teleports.get(teleporter.id)).toBeNull();
    expect(plan.diagnostics.some(item => item.subject === `Room ${teleporter.id} teleport` && item.severity === 'error')).toBe(true);
    const mechanism = document.mechanisms[0]!, controlled = source.rooms.find(room => room.originZone === mechanism.zone)!;
    const deleted = planRoomReferences(document, removeSourceRooms(source, [controlled.id]).after);
    expect(deleted.mechanisms.find(item => item.kind === mechanism.kind && item.id === mechanism.id)!.cells).toEqual([]);
    expect(deleted.diagnostics.some(item => item.subject === `${mechanism.kind} ${mechanism.id}` && item.severity === 'error')).toBe(true);
  });
});
