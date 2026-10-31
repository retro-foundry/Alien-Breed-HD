import { checkedInteger } from './binary';
import type { Position } from './geometry';
import { portalVisibility } from './portalVisibility';
import type { RoomTopology } from './roomTopology';
import { runtimeOrderingMask, type OrderingRoom, type VisibilityRoom } from './visibilityGeometry';
import { BinaryWriter } from './writer';
import type { ZoneRecordInput } from './zoneBuild';

export interface VisibilityTableInput {
  points: readonly Position[];
  cells: readonly VisibilityRoom[];
  rooms: readonly OrderingRoom[];
  graphicPoints: ReadonlyMap<number, readonly number[]>;
  owners?: ReadonlyMap<number, number>;
  pointLinkCount?: number;
  previousOrder?: ReadonlyMap<number, readonly number[]>;
}
export interface RoomVisibilityTables {
  visibility: ZoneRecordInput['visibility'];
  rotationPoints: number[];
}
export interface VisibilityTables {
  rooms: Map<number, RoomVisibilityTables>;
  clips: Uint8Array;
  connectTableOffset: number;
}

/** LEVELSAVE's visibility/ordering/rotation lists and INITDEFORDER's solid
 * corner links, shared by imported logical rooms and generated convex rooms.
 * Legacy clip fields are flags here; level_assign_clips walks the concatenated
 * left/-1/right/-2 lists and assigns their actual offsets when loading.
 */
export function buildVisibilityTables(input: VisibilityTableInput): VisibilityTables {
  const ordering = new Map(input.rooms.map(room => [room.id, room]));
  if (!ordering.size || ordering.size !== input.rooms.length) throw new Error('Visibility tables require unique runtime room identities.');
  for (const room of input.rooms) {
    checkedInteger(room.id, 0, 255, 'Visibility room identity');
    const points = input.graphicPoints.get(room.id);
    if (!points) throw new Error(`Room ${room.id} is missing its generated graphics point references.`);
    points.forEach(point => checkedInteger(point, 0, input.points.length - 1, 'Graphics rotation point'));
  }
  const compiled = portalVisibility(input.points, input.cells, input.owners);
  if (compiled.visible.size !== ordering.size || [...compiled.visible.keys()].some(id => !ordering.has(id)))
    throw new Error('Visibility cells and runtime room records disagree about their identities.');
  const pointLinks = checkedInteger(input.pointLinkCount ?? input.points.length, 0, input.points.length, 'Solid-wall point link count');
  const rooms = new Map<number, RoomVisibilityTables>(), clips = new BinaryWriter();
  for (const room of input.rooms) {
    const visible = compiled.visible.get(room.id)!, previous = input.previousOrder?.get(room.id) ?? [];
    const oldOrder = new Map(previous.map((id, index) => [id, index]));
    const rank = (id: number) => id === room.id ? -1 : oldOrder.get(id) ?? previous.length + id;
    const targets = [...visible].sort((a, b) => rank(a) - rank(b));
    const rotate = new Set<number>(), visibility: ZoneRecordInput['visibility'][number][] = [];
    for (const target of targets) {
      visibility.push({ zone: target, clip: target === room.id ? -1 : 0,
        mask: runtimeOrderingMask(room, ordering.get(target)!, visible) });
      input.graphicPoints.get(target)!.forEach(point => rotate.add(point));
      if (target === room.id) continue;
      checkedInteger(clips.offset / 2, 0, 32767, 'Clip word offset');
      const bounds = compiled.clips.get(room.id)!.get(target)!;
      for (const id of bounds.left) { clips.word(checkedInteger(id, 0, pointLinks - 1, 'Left clip point')); rotate.add(id); }
      clips.word(-1);
      for (const id of bounds.right) { clips.word(checkedInteger(id, 0, pointLinks - 1, 'Right clip point')); rotate.add(id); }
      clips.word(-2);
    }
    rooms.set(room.id, { visibility, rotationPoints: [...rotate].sort((a, b) => a - b) });
  }
  const connectTableOffset = clips.offset;
  checkedInteger(connectTableOffset / 2, 0, 32767, 'Total clip word count');
  for (let id = 0; id < pointLinks; id++) {
    clips.word(checkedInteger(compiled.previous[id]!, -1, pointLinks - 1, 'Previous solid-wall point'));
    clips.word(checkedInteger(compiled.next[id]!, -1, pointLinks - 1, 'Next solid-wall point'));
  }
  return { rooms, clips: clips.finish(), connectTableOffset };
}

/** Generated cells are runtime rooms themselves; they do not inherit the
 * imported logical-room aliases or the sixteen reserved switch-point gap. */
export function buildSourceVisibility(topology: RoomTopology, graphicPoints: ReadonlyMap<number, readonly number[]>): VisibilityTables {
  return buildVisibilityTables({ points: topology.points, cells: topology.cells, graphicPoints,
    rooms: topology.cells.map(cell => ({ id: cell.id, outline: cell.vertices.map(id => topology.points[id]!), neighbours: cell.neighbours })) });
}
