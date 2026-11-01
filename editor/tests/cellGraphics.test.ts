import { expect, it } from 'vitest';
import { BinaryReader } from '../src/model/binary';
import { buildCellGraphics } from '../src/model/cellGraphics';
import { readGraphicRecord } from '../src/model/graphics';
import { openLevel } from '../src/model/level';
import { compileRoomTopology } from '../src/model/roomTopology';
import { originalFiles } from './helpers';
import { buildGraphicsStreams, importedGraphicsStreams } from '../src/model/graphicsBuild';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../src/model/roomSource';
import { encodePolygonSurface } from '../src/model/surfaces';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('assembles level %i cell graphics in source draw order with complete point references', level => {
  const document = openLevel('original', originalFiles(level)), topology = compileRoomTopology(document, document.roomSource);
  const built = buildCellGraphics(document, document.roomSource, topology);
  for (const cell of topology.cells) {
    const template = document.zones[cell.recordZone]!, stream = built.streams.get(cell.id)!;
    const wholeRoom = topology.sourceCells.get(cell.sourceId)!.length === 1;
    for (const [records, generated] of [[template.lower, stream.lower], [template.upper, stream.upper ?? []]] as const) {
      if (wholeRoom) expect(generated.map(item => item.bytes)).toEqual(records.map(record => document.files.graphics.slice(record.offset, record.offset + record.size)));
      const order = generated.map(item => records.findIndex(record => record.offset === item.sourceOffset));
      expect(order).toEqual(order.toSorted((a, b) => a - b));
      for (const item of generated) {
        const record = readGraphicRecord(new BinaryReader(item.bytes, 'compiled'), 0, built.topology.points.length);
        record.points.forEach(id => expect(built.pointReferences.get(cell.id)).toContain(id));
      }
    }
    const reader = new BinaryReader(document.files.graphics, 'source');
    expect(stream.upper !== undefined).toBe(reader.i32(reader.i32(12) + template.id * 8 + 4) !== 0);
  }
});

it('recognises and validates both LightDraw point references', () => {
  const bytes = new Uint8Array([0, 6, 0, 2, 0, 3]);
  expect(readGraphicRecord(new BinaryReader(bytes, 'beam'), 0, 4).points).toEqual([2, 3]);
  expect(() => readGraphicRecord(new BinaryReader(bytes, 'beam'), 0, 3)).toThrow(/outside the point table/);
});

it('translates arc and beam points while retaining surrounding control order and payloads', () => {
  const original = openLevel('controls', originalFiles(1)), streams = importedGraphicsStreams(original), stream = streams.get(0)!;
  const ids = original.zones[0]!.lower.find(record => record.wall)!.points;
  const arc = new Uint8Array(30), arcView = new DataView(arc.buffer);
  arcView.setInt16(0, 5); arcView.setInt16(2, ids[0]!); arcView.setInt16(4, ids[1]!);
  arcView.setInt16(6, 17); arcView.setInt16(8, 83); arcView.setInt16(12, 3);
  const beam = new Uint8Array(6), beamView = new DataView(beam.buffer);
  beamView.setInt16(0, 6); beamView.setInt16(2, ids[0]!); beamView.setInt16(4, ids[1]!);
  streams.set(0, { ...stream, lower: [{ bytes: new Uint8Array([0, 3]) }, { bytes: beam }, ...stream.lower,
    { bytes: new Uint8Array([0, 4, 0, 1]) }, { bytes: arc }, { bytes: new Uint8Array([0, 12]) }] });
  const document = openLevel('controls', { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined, { streams }) });
  const source = { ...document.roomSource,
    rooms: document.roomSource.rooms.map(room => ({ ...room, vertices: room.vertices.map(point => ({ x: point.x + 8192, z: point.z + 8192 })) })) };
  const topology = compileRoomTopology(document, source), before = structuredClone(topology);
  const built = buildCellGraphics(document, source, topology), generated = built.streams.get(0)!.lower;
  expect(generated.map(item => new BinaryReader(item.bytes, 'record').i16(0))).toEqual(document.zones[0]!.lower.map(record => record.type));
  for (const item of generated) {
    const record = document.zones[0]!.lower.find(record => record.offset === item.sourceOffset)!;
    if (record.type !== 5 && record.type !== 6) continue;
    const decoded = readGraphicRecord(new BinaryReader(item.bytes, 'translated'), 0, built.topology.points.length);
    expect(decoded.points.map(id => built.topology.points[id])).toEqual(record.points.map(id => ({ x: document.points[id]!.x + 8192, z: document.points[id]!.z + 8192 })));
    expect(item.bytes.slice(6)).toEqual(document.files.graphics.slice(record.offset + 6, record.offset + record.size));
    decoded.points.forEach(id => expect(built.pointReferences.get(0)).toContain(id));
  }
  expect(topology).toEqual(before);
  const copies = { ...source, rooms: source.rooms.map(room => ({ ...room, originZone: null })) };
  const copiedTopology = compileRoomTopology(document, copies), copied = buildCellGraphics(document, copies, copiedTopology);
  expect(copiedTopology.cells.every(cell => cell.originZone === null)).toBe(true);
  expect(copied.streams.get(0)!.lower).toEqual(generated);
  const reshaped = { ...source, rooms: document.roomSource.rooms.map(room => ({ ...room, vertices: room.vertices.map(point => ({ x: point.x * 2, z: point.z * 2 })) })) };
  expect(() => buildCellGraphics(document, reshaped, compileRoomTopology(document, reshaped))).toThrow(/explicit geometry mapping/);
});

it('uses the explicit new-wall position relative to water and object passes', () => {
  const original = openLevel('water ordering', originalFiles(1)), streams = importedGraphicsStreams(original), template = original.zones[0]!;
  const wall = template.lower.find(record => record.wall)!.wall!, floor = template.lower.find(record => record.polygon)!.polygon!;
  const originalStream = streams.get(0)!;
  streams.set(0, { ...originalStream, lower: [{ bytes: new Uint8Array([0, 4, 0, 0]) },
    ...originalStream.lower.filter(item => template.lower.find(record => record.offset === item.sourceOffset)!.wall),
    { bytes: encodePolygonSurface({ ...floor, type: 7 }) }, { bytes: new Uint8Array([0, 4, 0, 1]) },
    ...originalStream.lower.filter(item => template.lower.find(record => record.offset === item.sourceOffset)!.polygon)] });
  const document = openLevel('water ordering', { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined, { streams }) });
  const drawn = commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }])).after;
  const room = drawn.rooms.at(-1)!, source = { ...drawn, rooms: [room] }, topology = compileRoomTopology(document, source);
  const insertion = document.zones[0]!.lower.findIndex(record => record.type === 7) + 1;
  const settings = { walls: { lower: [wall] }, wallInsertion: { lower: insertion } };
  const built = buildCellGraphics(document, source, topology, new Map([[room.id, settings]]));
  const types = built.streams.get(0)!.lower.map(item => new BinaryReader(item.bytes, 'record').i16(0));
  expect(types.slice(0, 7)).toEqual([4, 7, wall.type, wall.type, wall.type, wall.type, 4]);
  expect(() => buildCellGraphics(document, source, topology, new Map([[room.id,
    { ...settings, wallInsertion: { lower: 9999 } }]]))).toThrow(/wall insertion position/);
  expect(() => buildCellGraphics(document, source, topology, new Map([['missing', settings]]))).toThrow(/missing room missing/);
});
