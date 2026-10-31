/** Compiler integration fixture: explicit new geometry with materials selected
 * from original records, not a replacement renderer or a full-source compiler. */
import { BinaryReader } from '../src/model/binary';
import { buildDataTables, importedDataTables } from '../src/model/dataBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { buildZoneRecords } from '../src/model/zoneBuild';
import { polygonForCell, encodePolygonSurface, encodeWallSurface } from '../src/model/surfaces';
import { normaliseRoomOutline } from '../src/model/geometry';
import { openLevel, type LevelDocument } from '../src/model/level';
import { applyCommand, setStart } from '../src/model/commands';

export function buildAuthoredRoom(original: LevelDocument): LevelDocument {
  const template = original.zones[0]!, floor = template.lower.find(record => record.polygon && [1, 8, 9, 10, 11].includes(record.type))!.polygon!;
  const roof = template.lower.find(record => record.type === 2)!.polygon!;
  const wall = template.lower.find(record => record.wall?.topFixed === template.ceiling * 256 && record.wall?.bottomFixed === template.floor * 256)!.wall!;
  if (!floor || !roof || !wall) throw new Error('The fixture requires explicit original floor, roof and full-height wall materials.');
  const vertices = normaliseRoomOutline([{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]);
  const source = importedDataTables(original), pointIds = vertices.map((_, index) => source.points.length + index), id = original.zones.length;
  const sample = source.points[floor.points[0]!]!;
  const input = { points: [...source.points, ...vertices.map(point => ({ ...point, lowerBrightness: sample.lowerBrightness, upperBrightness: sample.upperBrightness }))],
    collision: [...source.collision, ...vertices.map((start, index) => ({ start, end: vertices[(index + 1) % vertices.length]!, neighbour: -1 }))] };
  const tables = buildDataTables(original, input), displacement = tables.data.length - original.files.data.length;
  const prefixEnd = Math.min(...original.zones.map(zone => zone.exitListOffset)) + displacement;
  const records = original.zones.map(zone => ({ id: zone.id, header: original.files.data.slice(zone.offset, zone.offset + 48),
    boundaries: zone.boundaryIds, cornerWalls: zone.wallIds, visibility: zone.visibility, rotationPoints: zone.rotationPointIds }));
  const rooms = buildZoneRecords(tables.data.subarray(0, prefixEnd), [...records, { id,
    header: original.files.data.slice(template.offset, template.offset + 48), boundaries: vertices.map((_, index) => original.lines.length + index),
    cornerWalls: [], visibility: [{ zone: id, clip: -1, mask: 0 }], rotationPoints: pointIds }]);
  const lower = [
    ...pointIds.map((point, index) => encodeWallSurface({ ...wall, points: [point, pointIds[(index + 1) % pointIds.length]!] })),
    encodePolygonSurface(polygonForCell(floor, pointIds, floor.heightQuarter / 4)),
    encodePolygonSurface(polygonForCell(roof, pointIds, roof.heightQuarter / 4)),
  ];
  const graphics = buildGraphicsStreams(original, undefined, id + 1, { additions: [{ id, lower }] });
  const graphView = new DataView(graphics.buffer); rooms.offsets.forEach((offset, zone) => graphView.setInt32(16 + zone * 4, offset));
  const links = new Uint8Array(original.connectTableOffset + input.points.length * 4).fill(255);
  links.set(original.files.clips.subarray(0, original.connectTableOffset));
  const old = new BinaryReader(original.files.clips, 'original clips'), oldCount = (old.bytes.length - original.connectTableOffset) / 4;
  links.set(old.bytes.subarray(original.connectTableOffset, original.connectTableOffset + oldCount * 4), original.connectTableOffset);
  const clipView = new DataView(links.buffer);
  pointIds.forEach((point, i) => {
    clipView.setInt16(original.connectTableOffset + point * 4, pointIds[(i + pointIds.length - 1) % pointIds.length]!);
    clipView.setInt16(original.connectTableOffset + point * 4 + 2, pointIds[(i + 1) % pointIds.length]!);
  });
  const compiled = openLevel('Generated room', { data: rooms.data, graphics, clips: links }, original.playtestSlot, -1);
  return applyCommand(compiled, setStart(compiled, 1, 20064, 20064, id));
}
