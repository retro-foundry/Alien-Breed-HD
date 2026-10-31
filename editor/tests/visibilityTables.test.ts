import { describe, expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { openLevel } from '../src/model/level';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildSourceVisibility, buildVisibilityTables } from '../src/model/visibilityTables';
import { originalFiles } from './helpers';

function corridor() {
  const document = openLevel('original', originalFiles(1));
  let source = removeSourceRooms(document.roomSource, document.roomSource.rooms.map(room => room.id)).after;
  for (let x = 20000; x < 20384; x += 128) source = commitRoomDraft(source,
    setDraftCorners(beginRoomDraft(document, source, 0), [{ x, z: 20000 }, { x: x + 128, z: 20000 },
      { x: x + 128, z: 20128 }, { x, z: 20128 }])).after;
  return compileRoomTopology(document, source);
}

describe('generated visibility table assembly', () => {
  it('emits a joined corridor, self-first masks, complete rotation lists and a continuous outer solid boundary', () => {
    const topology = corridor(), points = new Map(topology.cells.map(cell => [cell.id, cell.vertices]));
    const tables = buildSourceVisibility(topology, points), reader = new BinaryReader(tables.clips, 'generated clips');
    expect(tables.rooms.size).toBe(3);
    let cursor = 0;
    for (const cell of topology.cells) {
      const record = tables.rooms.get(cell.id)!;
      expect(record.visibility.map(entry => entry.zone)).toEqual([cell.id, ...[0, 1, 2].filter(id => id !== cell.id)]);
      expect(record.visibility[0]).toEqual({ zone: cell.id, clip: -1, mask: cell.id === 1 ? 63 : 7 });
      expect(record.rotationPoints).toEqual([...new Set(topology.cells.flatMap(target => target.vertices))].sort((a, b) => a - b));
      // Walk the actual native left/-1/right/-2 contract, not parser metadata.
      for (const entry of record.visibility.slice(1)) {
        expect(entry.clip).toBe(0);
        while (reader.i16(cursor) >= 0) { expect(record.rotationPoints).toContain(reader.i16(cursor)); cursor += 2; }
        expect(reader.i16(cursor)).toBe(-1); cursor += 2;
        while (reader.i16(cursor) >= 0) { expect(record.rotationPoints).toContain(reader.i16(cursor)); cursor += 2; }
        expect(reader.i16(cursor)).toBe(-2); cursor += 2;
      }
    }
    expect(cursor).toBe(tables.connectTableOffset);
    expect(reader.bytes.length - cursor).toBe(topology.points.length * 4);
    const outline = [{ x: 20000, z: 20000 }, { x: 20000, z: 20128 }, { x: 20128, z: 20128 }, { x: 20256, z: 20128 },
      { x: 20384, z: 20128 }, { x: 20384, z: 20000 }, { x: 20256, z: 20000 }, { x: 20128, z: 20000 }];
    const ids = outline.map(point => topology.points.findIndex(candidate => candidate.x === point.x && candidate.z === point.z));
    ids.forEach((id, i) => {
      expect(reader.i16(cursor + id * 4)).toBe(ids[(i + ids.length - 1) % ids.length]);
      expect(reader.i16(cursor + id * 4 + 2)).toBe(ids[(i + 1) % ids.length]);
    });
  });

  it('rejects missing graphics, invalid rotation points and mismatched runtime identities', () => {
    const topology = corridor(), points = new Map(topology.cells.map(cell => [cell.id, cell.vertices]));
    const before = JSON.stringify(topology);
    expect(() => buildSourceVisibility(topology, new Map())).toThrow(/missing.*graphics point/);
    expect(() => buildSourceVisibility(topology, new Map(points).set(1, [topology.points.length]))).toThrow(/rotation point/);
    expect(() => buildVisibilityTables({ points: topology.points, cells: topology.cells, graphicPoints: points,
      rooms: topology.cells.slice(0, 2).map(cell => ({ id: cell.id, outline: cell.vertices.map(id => topology.points[id]!), neighbours: cell.neighbours })) }))
      .toThrow(/disagree.*identities/);
    expect(JSON.stringify(topology)).toBe(before);
  });
});
