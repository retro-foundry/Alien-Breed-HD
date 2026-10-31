/** Integration fixture for a bijective runtime-room rebuild. Geometry and
 * existing visibility membership stay fixed; every room identity changes. */
import { BinaryReader } from '../src/model/binary';
import { buildDataPrefix, importedDataTables } from '../src/model/dataBuild';
import { importedGraphicsStreams } from '../src/model/graphicsBuild';
import { buildGeneratedGraphics } from '../src/model/generatedGraphics';
import { importedAnimationTables, remapAnimationRooms } from '../src/model/animationBuild';
import { openLevel, type LevelDocument } from '../src/model/level';
import { BinaryWriter } from '../src/model/writer';
import { buildZoneRecords } from '../src/model/zoneBuild';

export function reindexRooms(document: LevelDocument, ids: ReadonlyMap<number, number>, resolvedExit: number): LevelDocument {
  const remap = (id: number): number => {
    if (id < 0) return id;
    const value = ids.get(id);
    if (value === undefined) throw new Error(`Room ${id} has no generated identity.`);
    return value;
  };
  const source = importedDataTables(document);
  const prefix = buildDataPrefix(document, { ...source, collision: source.collision.map(line => ({ ...line, neighbour: remap(line.neighbour) })) }, document.zones.length);
  const view = new DataView(prefix.buffer);
  document.starts.forEach(start => view.setInt16(start.player === 1 ? 4 : 10, remap(start.zone)));
  document.objects.forEach(object => view.setInt16(object.offset + 12, remap(object.zone)));
  const clips = new BinaryWriter(), reader = new BinaryReader(document.files.clips, 'source clipping');
  const clipRegions = new Map<number, Uint8Array>(); let cursor = 0;
  for (const zone of document.zones) {
    const start = cursor;
    for (const entry of zone.visibility) if (entry.clip >= 0) cursor += (reader.wordsUntil(cursor, -2, 8192).length + 1) * 2;
    clipRegions.set(zone.id, reader.bytes.slice(start, cursor));
  }
  if (cursor !== document.connectTableOffset) throw new Error('Original clipping regions were not fully accounted for.');
  document.zones.toSorted((a, b) => remap(a.id) - remap(b.id)).forEach(zone => clips.write(clipRegions.get(zone.id)!));
  clips.write(document.files.clips.subarray(cursor));
  const rooms = buildZoneRecords(prefix, document.zones.toReversed().map(zone => {
    const header = document.files.data.slice(zone.offset, zone.offset + 48), headerView = new DataView(header.buffer);
    headerView.setInt16(38, remap(zone.teleport.zone));
    return { id: remap(zone.id), header, boundaries: zone.boundaryIds, cornerWalls: zone.wallIds,
      visibility: zone.visibility.map(entry => ({ ...entry, zone: remap(entry.zone) })), rotationPoints: zone.rotationPointIds };
  }));
  const streams = importedGraphicsStreams(document);
  const graphics = buildGeneratedGraphics(document, document.zones.toReversed().map(zone => ({ id: remap(zone.id),
    dataOffset: rooms.offsets[remap(zone.id)]!, ...streams.get(zone.id)! })),
  remapAnimationRooms(document, importedAnimationTables(document).tables, ids));
  return openLevel(`${document.name} renumbered`, { data: rooms.data, graphics, clips: clips.finish() }, document.playtestSlot, remap(resolvedExit));
}
