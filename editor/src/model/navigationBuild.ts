import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import { editableNavigation, navigationAssignment, navigationPosition } from './navigation';
import type { RoomDataReferences } from './roomDataReferences';
import type { RoomTopology } from './roomTopology';
import { BinaryWriter } from './writer';

/** Run after object/start relocation has consumed the original offsets, before
 * room assembly assigns absolute zone offsets. LEVELSAVE's control points are
 * eight bytes at +54; all subsequent absolute data-header pointers move together.
 * Object CID values, point coordinates, pools and runtime records stay intact. */
export function buildNavigation(document: LevelDocument, topology: RoomTopology, input: RoomDataReferences): RoomDataReferences {
  if (!document.roomSource.navigation) return input;
  const navigation = editableNavigation(document), reader = new BinaryReader(input.data, 'control-point compiler');
  const previousEnd = 54 + document.controlPoints.length * 8;
  if (reader.i32(42) !== previousEnd || reader.i16(12) !== document.controlPoints.length) throw new Error('Control-point relocation requires the preserved object-point prefix.');
  const output = new BinaryWriter(); output.write(input.data.subarray(0, 54));
  for (const point of navigation.points) {
    const current = navigationPosition(document, point);
    output.word(current.x); output.word(current.z); output.word(current.y); output.word(current.upper ? 1 : 0);
  }
  output.write(input.data.subarray(previousEnd));
  output.patchWord(12, navigation.points.length);
  const displacement = (navigation.points.length - document.controlPoints.length) * 8;
  for (const offset of [22, 26, 30, 34, 38, 42, 46, 50]) {
    const pointer = reader.i32(offset);
    if (pointer < previousEnd || pointer >= input.data.length) throw new Error(`Data pointer +${offset} lies outside the control-point relocation range.`);
    output.patchLong(offset, pointer + displacement);
  }
  const headers = new Map<number, Uint8Array>();
  for (const cell of topology.cells) {
    const header = input.headers.get(cell.id)!.slice(), view = new DataView(header.buffer);
    for (const upper of [false, true]) {
      const point = navigationAssignment(document, cell.sourceId, upper), authored = navigation.assignments.some(item => item.room === cell.sourceId && item.upper === upper);
      if (point === null && !authored) continue; // retain opaque original sentinel bytes
      const index = point === null ? -1 : navigation.points.findIndex(item => item.id === point);
      if (point !== null && index < 0) throw new Error(`Room ${cell.sourceId} uses deleted control point ${point}. Choose another point or None.`);
      view.setInt8(upper ? 27 : 26, checkedInteger(index, -1, 99, 'Room control-point index'));
    }
    headers.set(cell.id, header);
  }
  return { ...input, data: output.finish(), headers };
}
