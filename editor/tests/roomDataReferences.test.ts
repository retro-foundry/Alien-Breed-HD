import { describe, expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { openLevel } from '../src/model/level';
import { buildRoomDataReferences } from '../src/model/roomDataReferences';
import { compileRoomTopology } from '../src/model/roomTopology';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners } from '../src/model/roomSource';
import { originalFiles } from './helpers';

describe('source room data reference assembly', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('writes level %i references after room reordering and convex decomposition without changing other data', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level));
    const source = { ...document.roomSource, rooms: document.roomSource.rooms.toReversed() };
    const topology = compileRoomTopology(document, source), built = buildRoomDataReferences(document, source, topology);
    const reader = new BinaryReader(built.data, 'compiled references'), expected = document.files.data.slice(), view = new DataView(expected.buffer);
    for (const start of document.starts) {
      const id = reader.i16(start.player === 1 ? 4 : 10);
      expect(topology.cells[id]!.originZone).toBe(start.zone);
      view.setInt16(start.player === 1 ? 4 : 10, id);
    }
    for (const object of document.objects) {
      const id = reader.i16(object.offset + 12);
      if (object.zone < 0) expect(id).toBe(object.zone);
      else expect(topology.cells[id]!.originZone).toBe(object.zone);
      view.setInt16(object.offset + 12, id);
    }
    expect(built.data).toEqual(expected);
    for (const cell of topology.cells) {
      const template = document.zones[cell.recordZone]!, header = built.headers.get(cell.id)!;
      const headerReader = new BinaryReader(header, 'compiled room header');
      expect(headerReader.i16(0)).toBe(cell.id);
      const destination = headerReader.i16(38);
      if (template.teleport.zone < 0) expect(destination).toBe(template.teleport.zone);
      else expect(topology.cells[destination]!.originZone).toBe(template.teleport.zone);
      const retained = document.files.data.slice(template.offset, template.offset + 48), retainedView = new DataView(retained.buffer);
      retainedView.setInt16(0, cell.id); retainedView.setInt16(38, destination);
      expect(header).toEqual(retained);
    }
    expect(document.files.data).toEqual(openLevel('unchanged', originalFiles(level)).files.data);
  });

  it('inherits and remaps a new room template teleport instead of retaining its old zone number', () => {
    const document = openLevel('Level 9', originalFiles(9));
    const template = document.zones.find(zone => zone.teleport.zone >= 0 && zone.teleport.zone !== zone.id)!;
    const created = commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, template.id),
      [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }])).after;
    const source = { ...created, rooms: created.rooms.toReversed() }, topology = compileRoomTopology(document, source);
    const built = buildRoomDataReferences(document, source, topology), added = topology.cells.find(cell => cell.originZone === null)!;
    const target = new BinaryReader(built.headers.get(added.id)!, 'new room').i16(38);
    expect(topology.cells[target]!.originZone).toBe(template.teleport.zone);
    expect(target).not.toBe(template.teleport.zone);
    const deleted = removeSourceRooms(source, [source.rooms.find(room => room.originZone === template.teleport.zone)!.id]).after;
    const invalidTopology = compileRoomTopology(document, deleted);
    expect(() => buildRoomDataReferences(document, deleted, invalidTopology)).toThrow(/template teleport.*deleted/);
  });

  it('rejects unresolved starts and stale object offsets before mutating the supplied data', () => {
    const document = openLevel('Level 1', originalFiles(1)), input = document.files.data.slice();
    const startRoom = document.roomSource.rooms.find(room => room.originZone === document.starts[0]!.zone)!;
    const deleted = removeSourceRooms(document.roomSource, [startRoom.id]).after;
    expect(() => buildRoomDataReferences(document, deleted, compileRoomTopology(document, deleted), input)).toThrow(/Player 1 start.*deleted/);
    expect(input).toEqual(document.files.data);
    new DataView(input.buffer).setInt32(30, 54);
    const before = input.slice();
    expect(() => buildRoomDataReferences(document, document.roomSource, compileRoomTopology(document, document.roomSource), input)).toThrow(/prefix offsets/);
    expect(input).toEqual(before);
  });
});
