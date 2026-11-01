import { BinaryReader } from './binary';
import type { LevelDocument, Zone } from './level';
import type { CompiledRoomCell, RoomTopology } from './roomTopology';
import type { WallSurface } from './surfaces';
import { airHeightsChanged } from './roomHeights';
import { hasNativeWallMaterials } from './roomMaterials';

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
  const targetOrigin = target.originZone ?? (hasNativeWallMaterials(target.originZone, target.sourceLines) ? target.recordZone : null);
  return hasNativeWallMaterials(cell.originZone, cell.sourceLines) && line !== null && line !== undefined && document.lines[line]!.connect === targetOrigin
    && !airHeightsChanged(document, cell) && !airHeightsChanged(document, target) ? undefined : target;
}

/** An existing doorway whose destination identity is unchanged only needs its
 * air bands updated. It must retain native materials, not demand a new style. */
export function heightPortalTarget(document: LevelDocument, topology: RoomTopology, cell: CompiledRoomCell, edge: number): CompiledRoomCell | undefined {
  const target=topology.cells.find(item=>item.id===cell.neighbours[edge]),line=cell.sourceLines[edge];
  if(!target || target.sourceId===cell.sourceId || line==null || !hasNativeWallMaterials(cell.originZone,cell.sourceLines))return;
  const origin=target.originZone??(hasNativeWallMaterials(target.originZone,target.sourceLines)?target.recordZone:null);
  return document.lines[line]!.connect===origin && (airHeightsChanged(document,cell)||airHeightsChanged(document,target)) ? target : undefined;
}

/** Original portals whose destination no longer shares this generated edge
 * need new solid graphics as well as closed collision geometry. */
export function closedPortalEdges(document: LevelDocument, cell: CompiledRoomCell): number[] {
  if (!hasNativeWallMaterials(cell.originZone, cell.sourceLines)) return [];
  return cell.neighbours.flatMap((neighbour, edge) => {
    const line = cell.sourceLines[edge];
    return neighbour < 0 && line != null && document.lines[line]!.connect >= 0 ? [edge] : [];
  });
}

/** Replace old portal materials when the generated boundary is solid or now
 * opens into a different room. The new destination supplies the air bands. */
export function changedPortalEdges(document: LevelDocument, topology: RoomTopology, cell: CompiledRoomCell): number[] {
  if (!hasNativeWallMaterials(cell.originZone, cell.sourceLines)) return [];
  return cell.neighbours.flatMap((neighbour, edge) => {
    const line = cell.sourceLines[edge];
    return line != null && document.lines[line]!.connect >= 0
      && (neighbour < 0 || newPortalTarget(document, topology, cell, edge) && !heightPortalTarget(document,topology,cell,edge)) ? [edge] : [];
  });
}

/** True new perimeter edges need explicit material/lighting; decomposition
 * seams are internal and must never acquire walls. */
export function unassignedBoundaryEdges(topology: RoomTopology, cell: CompiledRoomCell): number[] {
  if (!hasNativeWallMaterials(cell.originZone, cell.sourceLines)) return [];
  return cell.neighbours.flatMap((neighbour, edge) => cell.sourceLines[edge] == null
    && (neighbour < 0 || topology.cells.find(target => target.id === neighbour)?.sourceId !== cell.sourceId) ? [edge] : []);
}

export function hasUpperLayer(document: LevelDocument, zone: Zone): boolean {
  const reader = new BinaryReader(document.files.graphics, 'portal layer table');
  return reader.i32(reader.i32(12) + zone.id * 8 + 4) > 0 && zone.upperFloor > zone.upperCeiling;
}

/** Match src/level.c level_zone_has_upper_layer and movement.c's lower/upper
 * floor/roof tests. Wall heights use the native signed 24.8 representation.
 * Subtract all destination air bands to leave steps, lintels and the solid
 * band between floors. Preserve every material/sampling/phase field. */
export function portalWallSpans<T extends Pick<WallSurface,'topFixed'|'bottomFixed'>>(document: LevelDocument, material: T, destination: Zone): T[] {
  const upper = hasUpperLayer(document, destination);
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
