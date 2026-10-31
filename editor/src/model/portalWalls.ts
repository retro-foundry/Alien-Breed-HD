import { BinaryReader } from './binary';
import type { LevelDocument, Zone } from './level';
import type { CompiledRoomCell, RoomTopology } from './roomTopology';
import type { WallSurface } from './surfaces';

/** Existing imported openings keep their authored draw records. New openings
 * are determined by the generated neighbour and original collision identity,
 * never by coincident wall coordinates alone. */
export function newPortalTarget(document: LevelDocument, topology: RoomTopology, cell: CompiledRoomCell, edge: number): CompiledRoomCell | undefined {
  const neighbour = cell.neighbours[edge]!;
  if (neighbour < 0) return undefined;
  const target = topology.cells.find(item => item.id === neighbour);
  if (!target) throw new Error(`Doorway ${cell.id} -> ${neighbour} has no draw-room identity.`);
  if (target.sourceId === cell.sourceId) return undefined;
  const line = cell.sourceLines[edge];
  return cell.originZone !== null && line !== null && line !== undefined && document.lines[line]!.connect === target.originZone ? undefined : target;
}

/** Match src/level.c level_zone_has_upper_layer and movement.c's lower/upper
 * floor/roof tests. Wall heights use the native signed 24.8 representation.
 * Subtract all destination air bands to leave steps, lintels and the solid
 * band between floors. Preserve every material/sampling/phase field. */
export function portalWallSpans(document: LevelDocument, material: WallSurface, destination: Zone): WallSurface[] {
  const reader = new BinaryReader(document.files.graphics, 'portal layer table');
  const upper = reader.i32(reader.i32(12) + destination.id * 8 + 4) > 0 && destination.upperFloor > destination.upperCeiling;
  const openings = [{ top: destination.ceiling * 256, bottom: destination.floor * 256 },
    ...(upper ? [{ top: destination.upperCeiling * 256, bottom: destination.upperFloor * 256 }] : [])]
    .filter(span => span.bottom > span.top).toSorted((a, b) => a.top - b.top);
  let spans = [{ top: material.topFixed, bottom: material.bottomFixed }];
  for (const opening of openings) spans = spans.flatMap(span => {
    if (opening.bottom <= span.top || opening.top >= span.bottom) return [span];
    return [{ top: span.top, bottom: Math.min(span.bottom, opening.top) },
      { top: Math.max(span.top, opening.bottom), bottom: span.bottom }].filter(piece => piece.bottom > piece.top);
  });
  return spans.map(span => ({ ...material, topFixed: span.top, bottomFixed: span.bottom }));
}
