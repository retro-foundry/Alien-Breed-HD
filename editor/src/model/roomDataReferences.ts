import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { RoomSource } from './roomSource';
import type { RoomTopology } from './roomTopology';
import { assertRoomReferences, planRoomReferences, type RoomReferencePlan } from './roomReferences';
import { runtimeObjectPosition } from './sourceObjects';
import { roomHeightFields, roomSettings } from './roomHeights';
import { writeRoomProperties } from './roomProperties';
import { waterStartHeight } from './sourceWater';

export interface RoomDataReferences {
  data: Uint8Array;
  headers: Map<number, Uint8Array>;
  references: RoomReferencePlan;
}

/** Write authored room relationships consumed by player.c's level starts,
 * objects.c's OBJ_ZONE (+12), and Defs.i's ToTelZone (+38). The data-table
 * writer preserves the object prefix, including shot pools and avatar slots.
 * Graphics-room state (+26), mechanisms and graphics pointers are separate
 * dependencies, not additional authored zone fields to guess or overwrite.
 * Call before assembling room records; their relative pointers are rebuilt
 * by buildZoneRecords, not by this reference pass.
 */
export function buildRoomDataReferences(document: LevelDocument, source: RoomSource, topology: RoomTopology,
  input = document.files.data): RoomDataReferences {
  const references = planRoomReferences(document, source, topology);
  assertRoomReferences(references);
  const original = new BinaryReader(document.files.data, 'source level data');
  const reader = new BinaryReader(input, 'generated level data');
  // Existing object records and object-point identities must still occupy
  // their preserved prefix. A later object-table editor must supply its own
  // relocation map rather than patching stale offsets through this pass.
  for (const field of [12, 20]) if (reader.i16(field) !== original.i16(field))
    throw new Error('Room reference assembly requires the preserved object/control counts.');
  for (const field of [22, 30, 34, 38, 42, 46, 50]) if (reader.i32(field) !== original.i32(field))
    throw new Error('Room reference assembly requires the preserved object prefix offsets.');
  reader.require(0, original.i32(22));
  const data = input.slice(), view = new DataView(data.buffer);
  for (const start of references.starts) {
    view.setInt16(start.player === 1 ? 4 : 10, checkedInteger(start.cell!, 0, topology.cells.length - 1, 'Player start room'));
  }
  for (const object of document.objects) {
    reader.require(object.offset, 64);
    view.setInt16(object.offset + 12, checkedInteger(references.objects.get(object.id)!, -32768, topology.cells.length - 1, 'Object room'));
  }
  for (const authored of document.sourceObjects ?? []) {
    const object = document.objects.find(item => item.id === authored.id)!;
    const cell = references.objects.get(object.id)!;
    // Authored moves use the same OBJ_ZONE (+12), OBJ_GROOM (+26) and
    // 16.16 CID point-table contract as commands.moveObject.
    view.setInt16(object.offset + 26, cell!);
    if (authored.room !== null) {
      const position = runtimeObjectPosition(authored);
      reader.require(object.pointOffset, 8);
      view.setInt32(object.pointOffset, position.x * 65536);
      view.setInt32(object.pointOffset + 4, position.z * 65536);
    }
  }
  for (const start of document.sourceStarts ?? []) {
    const offset = start.player === 1 ? 0 : 6;
    const cell = references.starts.find(item => item.player === start.player)!.cell!;
    view.setInt16(offset, start.x); view.setInt16(offset + 2, start.z);
    // Match commands.setStart and the native multiplayer avatar contract:
    // header LONG +46/+50 -> object CID -> shared 16.16 object point table.
    const avatar = reader.i32(start.player === 1 ? 46 : 50);
    reader.require(avatar, 64);
    const point = document.objectPointsOffset + reader.i16(avatar) * 8;
    reader.require(point, 8);
    view.setInt16(avatar + 12, cell); view.setInt16(avatar + 26, cell);
    view.setInt32(point, start.x * 65536); view.setInt32(point + 4, start.z * 65536);
  }
  const headers = new Map<number, Uint8Array>();
  for (const cell of topology.cells) {
    const template = document.zones.find(zone => zone.id === cell.recordZone);
    if (!template) throw new Error(`Room ${cell.sourceId} has no settings template.`);
    const header = document.files.data.slice(template.offset, template.offset + 48), headerView = new DataView(header.buffer);
    if (cell.properties) writeRoomProperties(header, cell.properties);
    headerView.setInt16(0, cell.id);
    const room = source.rooms.find(room => room.id === cell.sourceId)!;
    if (room.waterSurface !== undefined || source.waterGroups?.length) headerView.setInt32(18, waterStartHeight(document, room) * 256);
    if (room.sky?.some(Boolean)) headerView.setInt16(36, 1);
    // Defs.i / level.c: lower floor/roof and upper floor/roof are 24.8
    // LONGs at +2/+6/+10/+14. Water (+18) is separately authored.
    if (cell.heights || cell.staticMechanisms?.length) {
      const settings = roomSettings(document, cell);
      roomHeightFields.forEach((field, index) => headerView.setInt32(2 + index * 4, settings[field] * 256));
    }
    headerView.setInt16(38, checkedInteger(references.cellTeleports.get(cell.id)!, -32768, topology.cells.length - 1, 'Teleport destination room'));
    const teleport = document.sourceTeleports?.find(item => item.room === cell.sourceId);
    if (teleport) {
      headerView.setInt16(40, teleport.x); headerView.setInt16(42, teleport.z);
    }
    headers.set(cell.id, header);
  }
  return { data, headers, references };
}
