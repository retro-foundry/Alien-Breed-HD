/** Compiler integration fixture: explicit new geometry with materials selected
 * from original records, not a replacement renderer or a full-source compiler. */
import { BinaryReader } from '../src/model/binary';
import { buildDataPrefix, importedDataTables } from '../src/model/dataBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { buildZoneRecords } from '../src/model/zoneBuild';
import { buildCellGraphics } from '../src/model/cellGraphics';
import { normaliseRoomOutline } from '../src/model/geometry';
import { openLevel, type LevelDocument } from '../src/model/level';
import { applyCommand, setStart } from '../src/model/commands';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../src/model/roomSource';
import { compileRoomTopology } from '../src/model/roomTopology';
import { buildRoomDataReferences } from '../src/model/roomDataReferences';
import { buildSourceVisibility } from '../src/model/visibilityTables';
import { buildGeometryLighting } from '../src/model/geometryLighting';

export function buildAuthoredRoom(original: LevelDocument): LevelDocument {
  const template = original.zones[0]!, floor = template.lower.find(record => record.polygon && [1, 8, 9, 10, 11].includes(record.type))!.polygon!;
  const roof = template.lower.find(record => record.type === 2)!.polygon!;
  const wall = template.lower.find(record => record.wall?.topFixed === template.ceiling * 256 && record.wall?.bottomFixed === template.floor * 256)!.wall!;
  if (!floor || !roof || !wall) throw new Error('The fixture requires explicit original floor, roof and full-height wall materials.');
  const vertices = normaliseRoomOutline([{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]);
  const source = importedDataTables(original), pointIds = vertices.map((_, index) => source.points.length + index), id = original.zones.length;
  const roomSource = commitRoomDraft(original.roomSource,
    setDraftCorners(beginRoomDraft(original, original.roomSource, template.id), vertices)).after;
  const topology = compileRoomTopology(original, roomSource);
  const geometry = buildCellGraphics(original, roomSource, topology,
    new Map([[topology.cells.find(cell => cell.id === id)!.sourceId, { walls: { lower: [wall] },
      wallInsertion: { lower: template.lower.findIndex(record => record.wall) } }]]));
  const lighting = buildGeometryLighting(original, roomSource, geometry,
    new Map([[topology.cells.find(cell => cell.id === id)!.sourceId, { kind: 'copy', source: floor.points[0]! }]]));
  const cellGraphics = lighting.graphics;
  const input = { points: lighting.points,
    collision: [...source.collision, ...vertices.map((start, index) => ({ start, end: vertices[(index + 1) % vertices.length]!, neighbour: -1 }))] };
  const prefix = buildDataPrefix(original, input, id + 1);
  const remapped = buildRoomDataReferences(original, roomSource, topology, prefix);
  const addedCells = topology.cells.filter(cell => cell.id === id);
  const visibility = buildSourceVisibility({ ...cellGraphics.topology, cells: addedCells }, new Map([[id, cellGraphics.pointReferences.get(id)!]]));
  const records = original.zones.map(zone => ({ id: zone.id, header: remapped.headers.get(zone.id)!,
    boundaries: zone.boundaryIds, cornerWalls: zone.wallIds, visibility: zone.visibility, rotationPoints: zone.rotationPointIds }));
  const rooms = buildZoneRecords(remapped.data, [...records, { id,
    header: remapped.headers.get(id)!, boundaries: vertices.map((_, index) => original.lines.length + index),
    cornerWalls: [], ...visibility.rooms.get(id)! }]);
  const lower = cellGraphics.streams.get(id)!.lower.map(record => record.bytes);
  const graphics = buildGraphicsStreams(original, undefined, id + 1, { additions: [{ id, lower }] });
  const graphView = new DataView(graphics.buffer); rooms.offsets.forEach((offset, zone) => graphView.setInt32(16 + zone * 4, offset));
  const links = new Uint8Array(original.connectTableOffset + input.points.length * 4).fill(255);
  links.set(original.files.clips.subarray(0, original.connectTableOffset));
  const old = new BinaryReader(original.files.clips, 'original clips'), oldCount = (old.bytes.length - original.connectTableOffset) / 4;
  links.set(old.bytes.subarray(original.connectTableOffset, original.connectTableOffset + oldCount * 4), original.connectTableOffset);
  const clipView = new DataView(links.buffer), generatedLinks = new BinaryReader(visibility.clips, 'generated solid-wall links');
  pointIds.forEach(point => {
    clipView.setInt16(original.connectTableOffset + point * 4, generatedLinks.i16(visibility.connectTableOffset + point * 4));
    clipView.setInt16(original.connectTableOffset + point * 4 + 2, generatedLinks.i16(visibility.connectTableOffset + point * 4 + 2));
  });
  const compiled = openLevel('Generated room', { data: rooms.data, graphics, clips: links }, original.playtestSlot, -1);
  return applyCommand(compiled, setStart(compiled, 1, 20064, 20064, id));
}
