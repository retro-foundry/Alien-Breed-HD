import { buildAnimationAttachments } from './animationAttachments';
import { checkedInteger } from './binary';
import { buildCellGraphics, type NewRoomGraphics } from './cellGraphics';
import { compileCollisionGeometry } from './collisionBuild';
import { buildDataPrefix } from './dataBuild';
import { buildGeneratedGraphics } from './generatedGraphics';
import { buildGeometryLighting } from './geometryLighting';
import { openLevel, type LevelDocument } from './level';
import { buildPointLighting, type PointLightingBinding } from './pointLighting';
import { buildRoomDataReferences } from './roomDataReferences';
import { compileRoomTopology } from './roomTopology';
import { buildSourceVisibility } from './visibilityTables';
import { buildZoneRecords } from './zoneBuild';
import { buildDrawTopology } from './drawTopology';

export interface SourceBuildOptions {
  /** Resolved game configuration in source room identities; -1 disables exit.
   * The compiler does not duplicate the runtime's default exit table. */
  resolvedExit: number;
  newRoomGraphics?: ReadonlyMap<string, NewRoomGraphics>;
  newRoomLighting?: ReadonlyMap<string, PointLightingBinding>;
}

/** Atomic complete-source assembly. The source document/archive remains the
 * editable baseline; the returned document is a generated play/export artifact.
 * Every geometry-dependent table is rebuilt before validated openLevel.
 */
export function buildSourceLevel(document: LevelDocument, options: SourceBuildOptions) {
  const source = document.roomSource, topology = compileRoomTopology(document, source);
  for (const cell of topology.cells) cell.neighbours.forEach((neighbour, edge) => {
    if (neighbour < 0) return;
    const target = topology.cells.find(target => target.id === neighbour)!;
    if (target.sourceId === cell.sourceId) return;
    if (cell.originZone !== null) {
      const line = cell.sourceLines[edge];
      if (line === null || document.lines[line!]!.connect !== target.originZone)
        throw new Error('A new doorway needs regenerated wall openings before this geometry can be built.');
    } else if (target.originZone === null) {
      const a = document.zones[cell.recordZone]!, b = document.zones[target.recordZone]!;
      if (a.floor !== b.floor || a.ceiling !== b.ceiling || a.upperFloor !== b.upperFloor || a.upperCeiling !== b.upperCeiling)
        throw new Error('A new doorway between different room heights needs generated step and ceiling wall spans before building.');
    }
  });
  const geometry = buildCellGraphics(document, source, topology, options.newRoomGraphics);
  const lighting = buildGeometryLighting(document, source, geometry, options.newRoomLighting);
  const collision = compileCollisionGeometry(lighting.graphics.topology);
  const attachments = buildAnimationAttachments(document, lighting.graphics, collision);
  const graphics = attachments.graphics;
  const points = buildPointLighting(document, graphics.topology.points, new Map([...lighting.bindings, ...attachments.addedLighting]));
  const previousOrder = new Map(topology.cells.filter(cell => cell.originZone !== null).map(cell => [cell.id,
    document.zones.find(zone => zone.id === cell.originZone)!.visibleZones.flatMap(id => topology.cells.filter(target => target.originZone === id).map(target => target.id))]));
  const visibility = buildSourceVisibility(graphics.topology, graphics.pointReferences, previousOrder);
  const prefix = buildDataPrefix(document, { points, collision: collision.segments }, topology.cells.length);
  const references = buildRoomDataReferences(document, source, graphics.topology, prefix);
  const rooms = buildZoneRecords(references.data, topology.cells.map((cell, index) => ({ id: cell.id,
    header: references.headers.get(cell.id)!, boundaries: collision.boundaries[index]!, cornerWalls: collision.cornerWalls[index]!,
    ...visibility.rooms.get(cell.id)! })));
  const bytes = buildGeneratedGraphics(document, topology.cells.map(cell => ({ id: cell.id, dataOffset: rooms.offsets[cell.id]!,
    ...graphics.streams.get(cell.id)! })), attachments.tables, attachments.selectTarget);
  const originalExit = document.exitZone ?? checkedInteger(options.resolvedExit, -1, document.zones.length - 1, 'Source exit room');
  let exitZone = originalExit;
  if (originalExit >= 0) {
    const destinations = topology.cells.filter(cell => cell.originZone === originalExit);
    if (destinations.length !== 1) throw new Error('The exit room was deleted or split. Assign a single generated exit destination before building.');
    exitZone = destinations[0]!.id;
  }
  const owners = buildDrawTopology(source, graphics.topology).owners;
  const groups = topology.cells.map(cell => owners.get(cell.id)!);
  const compiled = openLevel(`${document.name} compiled`, { data: rooms.data, graphics: bytes, clips: visibility.clips }, document.playtestSlot, exitZone,
    groups.some((group, index) => group !== index) ? groups : undefined);
  return { document: compiled, topology: graphics.topology, references: references.references };
}
