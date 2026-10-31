import { describe, expect, it } from 'vitest';
import { buildZoneRecords, type ZoneRecordInput } from '../src/model/zoneBuild';
import { BinaryReader } from '../src/model/binary';
import { openLevel, parseLevel, type LevelDocument } from '../src/model/level';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { originalFiles } from './helpers';

function inputs(document: LevelDocument): ZoneRecordInput[] {
  return document.zones.map(zone => ({ id: zone.id, header: document.files.data.slice(zone.offset, zone.offset + 48),
    boundaries: zone.boundaryIds, cornerWalls: zone.wallIds, visibility: zone.visibility, rotationPoints: zone.rotationPointIds }));
}
function prefix(document: LevelDocument) { return document.files.data.subarray(0, Math.min(...document.zones.map(zone => zone.exitListOffset))); }
function relocateGraphics(graphics: Uint8Array, offsets: number[]) {
  const copy = graphics.slice(), view = new DataView(copy.buffer);
  offsets.forEach((offset, id) => view.setInt32(16 + id * 4, offset)); return copy;
}

describe('runtime room-record assembly', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('reconstructs every byte of level %i original room records and safely relocates reversed records', level => {
    const document = openLevel(`Level ${level}`, originalFiles(level)), records = inputs(document);
    const built = buildZoneRecords(prefix(document), records);
    expect(built.data).toEqual(document.files.data);
    expect(built.offsets).toEqual(document.zones.map(zone => zone.offset));
    const reversed = buildZoneRecords(prefix(document), records.toReversed());
    const rebuilt = parseLevel('reversed rooms', { ...document.files, data: reversed.data, graphics: relocateGraphics(document.files.graphics, reversed.offsets) });
    for (const zone of document.zones) {
      const changed = rebuilt.zones[zone.id]!;
      expect(reversed.data.subarray(changed.offset, changed.offset + 32)).toEqual(document.files.data.subarray(zone.offset, zone.offset + 32));
      expect(changed.boundaryIds).toEqual(zone.boundaryIds); expect(changed.wallIds).toEqual(zone.wallIds);
      expect(changed.rotationPointIds).toEqual(zone.rotationPointIds);
      expect(changed.visibility.map(({ offset: _, ...entry }) => entry)).toEqual(zone.visibility.map(({ offset: _, ...entry }) => entry));
    }
    expect(buildZoneRecords(prefix(rebuilt), inputs(rebuilt)).data).toEqual(document.files.data);
  });
  it('assembles an additional room header and updates both count locations and its graphics address', () => {
    const document = openLevel('Level 1', originalFiles(1)), records = inputs(document), id = records.length;
    const extra: ZoneRecordInput = { ...records[0]!, id, visibility: [{ zone: id, clip: -1, mask: 0 }] };
    const built = buildZoneRecords(prefix(document), [...records, extra]);
    const graphics = relocateGraphics(buildGraphicsStreams(document, undefined, id + 1), built.offsets);
    const rebuilt = parseLevel('extra header', { ...document.files, data: built.data, graphics });
    expect(rebuilt.zones).toHaveLength(id + 1); expect(rebuilt.zones.at(-1)!.fileId).toBe(id);
    const reader = new BinaryReader(built.data, 'generated room data');
    expect(reader.i16(16)).toBe(id); expect(reader.i16(prefix(document).length - 2)).toBe(id + 1);
    expect(rebuilt.zones.at(-1)!.visibleZones).toEqual([id]);
    // This is an assembly test, not a new playable-room claim: surface and
    // dependency generation are still required before exposing room creation.
  });
  it('rejects invalid references, duplicate rooms and missing self visibility without changing source', () => {
    const document = openLevel('Level 1', originalFiles(1)), records = inputs(document), before = document.files.data.slice();
    expect(() => buildZoneRecords(prefix(document), [...records.slice(0, -1), records[0]!])).toThrow(/duplicate runtime identities/);
    expect(() => buildZoneRecords(prefix(document), [{ ...records[0]!, id: 0, visibility: [{ zone: 0, clip: -1, mask: 0 }], rotationPoints: [2047] }])).toThrow(/Rotation point/);
    expect(() => buildZoneRecords(prefix(document), [{ ...records[0]!, boundaries: [0, 1, 32767] }])).toThrow(/Room collision line/);
    expect(() => buildZoneRecords(prefix(document), records.map((record, i) => i ? record : { ...record, visibility: [{ zone: 1, clip: 0, mask: 0 }] }))).toThrow(/own visibility/);
    expect(document.files.data).toEqual(before);
  });
});
