import { BinaryReader, checkedInteger } from './binary';
import { type LevelDocument, type LevelFiles } from './level';
import { visibilityCells } from './visibilityImport';
import { buildVisibilityTables } from './visibilityTables';
import { buildZoneRecords, type ZoneRecordInput } from './zoneBuild';

/** Rebuild derived visibility, clipping and rotation lists from imported source
 * geometry. Every original zone block is accounted for before replacement.
 * All other level records and graphics streams remain byte-preserved. */
export function buildVisibilityFiles(document: LevelDocument): LevelFiles {
  const zones = [...document.zones].sort((a, b) => a.exitListOffset - b.exitListOffset);
  if (!zones.length) throw new Error('A level needs a room before visibility can be built.');
  const prefixEnd = zones[0]!.exitListOffset;
  for (let i = 0; i < zones.length; i++) {
    const zone = zones[i]!, end = zones[i + 1]?.exitListOffset ?? document.files.data.length;
    if (zone.exitListOffset >= zone.offset || zone.dataEnd !== end)
      throw new Error(`Zone ${zone.id} has additional source data outside the supported layout. The original files have been kept.`);
  }
  const sourceData = new BinaryReader(document.files.data, 'twolev.bin');
  for (const offset of [22, 26, 30, 34, 38, 42, 46, 50])
    if (sourceData.i32(offset) >= prefixEnd) throw new Error('A non-zone table overlaps the zone data being rebuilt.');
  const graphics = document.files.graphics.slice(), graphicView = new DataView(graphics.buffer);
  const sourceGraphics = new BinaryReader(graphics, 'twolev.graph.bin'), table = sourceGraphics.i32(12);
  for (const zone of zones) {
    const stream = sourceGraphics.i32(table + zone.id * 8);
    if (stream <= 0 || sourceGraphics.i16(stream) !== zone.id || zone.fileId !== zone.id)
      throw new Error(`Zone ${zone.id} needs its graphics references normalised before visibility can be rebuilt.`);
  }
  const geometry = visibilityCells(document);
  const ordering = document.zones.map(zone => ({
    id: zone.id, outline: zone.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })),
    neighbours: zone.boundaryIds.map(id => document.lines[id]!.connect),
  }));
  const pointLinks = (document.files.clips.length - document.connectTableOffset) / 4;
  checkedInteger(pointLinks, 0, document.points.length, 'Solid-wall point link count');
  const tables = buildVisibilityTables({ points: geometry.points, cells: geometry.rooms, owners: geometry.owners, rooms: ordering,
    graphicPoints: new Map(document.zones.map(zone => [zone.id, [...zone.lower, ...zone.upper].flatMap(record => record.points)])),
    previousOrder: new Map(document.zones.map(zone => [zone.id, zone.visibility.map(entry => entry.zone)])), pointLinkCount: pointLinks });
  const records: ZoneRecordInput[] = [];
  for (const zone of document.zones) {
    records.push({ id: zone.id, header: sourceData.bytes.slice(zone.offset, zone.offset + 48), boundaries: zone.boundaryIds,
      cornerWalls: zone.wallIds, ...tables.rooms.get(zone.id)! });
  }
  const data = buildZoneRecords(document.files.data.subarray(0, prefixEnd), records);
  data.offsets.forEach((offset, id) => graphicView.setInt32(16 + id * 4, offset));
  // Door/lift +8 pointers address LEVELGRAPHICS (Anims.s), which is unchanged
  // apart from ZoneAdds. They must never be relocated as level-data pointers.
  return { data: data.data, graphics, clips: tables.clips };
}
