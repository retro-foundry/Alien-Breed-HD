import { activeNativeSwitches, validateRemovedNativeSwitches } from './nativeSwitches';
import { buildWallSettings, syncWallSettingsAnimations } from './wallSettingsBuild';
import { authorSeeThroughBarriers, barrierCollisionTopology } from './seeThroughBarriers';
import type { AnimationGraphicTarget } from './animationBuild';
import { buildAuthoredSwitches, appendAuthoredSwitches } from './switchBuild';
import { authorWaterGraphics, appendSourceWater } from './waterBuild';
import { buildNavigation } from './navigationBuild';
import { authorRoomLayers, authorSkyCeilings } from './roomLayerGraphics';
import { compileSourceMovingRoom } from './sourceMovingRoom';
import { buildAuthoredMovingRooms, appendAuthoredMovingAnimations, initialMovingGraphics, type AuthoredMovingRoom } from './authoredMovingRoom';
import { buildAnimationAttachments } from './animationAttachments';
import { checkedInteger } from './binary';
import { buildCellGraphics, type NewRoomGraphics, type ClosingRoomGraphics } from './cellGraphics';
import { compileCollisionGeometry } from './collisionBuild';
import { buildDataPrefix } from './dataBuild';
import { compileGeneratedGraphics } from './generatedGraphics';
import { buildGeometryLighting } from './geometryLighting';
import { openLevel, type LevelDocument } from './level';
import { buildPointLighting, type PointLightingBinding } from './pointLighting';
import { buildRoomDataReferences } from './roomDataReferences';
import { compileRoomTopology } from './roomTopology';
import { buildSourceVisibility } from './visibilityTables';
import { buildZoneRecords } from './zoneBuild';
import { buildDrawTopology } from './drawTopology';
import { closedPortalEdges, heightPortalTarget, newPortalTarget, unassignedBoundaryEdges } from './portalWalls';
import { resolveAnimationRoom } from './animationRooms';
import { airHeightsChanged } from './roomHeights';
import { hasNativeWallMaterials } from './roomMaterials';
import { mechanismDependencies, mechanismsForRoom } from './mechanismDependencies';
import { compileControllerCopies, validateMovingControllerCopies, type MovingControllerCopy } from './movingControllerCopies';
import { compileSwitchCopies, validateSwitchCopies, type SwitchCopy } from './switchCopies';
import { paintCellWalls } from './wallMaterials';
import { paintCellPolygons } from './polygonMaterials';

export interface SourceBuildOptions {
  /** Resolved game configuration in source room identities; -1 disables exit.
   * The compiler does not duplicate the runtime's default exit table. */
  resolvedExit: number;
  newRoomGraphics?: ReadonlyMap<string, NewRoomGraphics>;
  /** Initial lighting for new rooms and newly authored boundary corners in
   * imported rooms. Existing material/corner identities remain authored. */
  newRoomLighting?: ReadonlyMap<string, PointLightingBinding>;
  closingRoomGraphics?: ReadonlyMap<string, ClosingRoomGraphics>;
  controllerCopies?: readonly MovingControllerCopy[];
  switchCopies?: readonly SwitchCopy[];
  /** Source-owned settings for independent new lower moving rooms. */
  authoredControllers?: readonly AuthoredMovingRoom[];
}

/** Atomic complete-source assembly. The source document/archive remains the
 * editable baseline; the returned document is a generated play/export artifact.
 * Every geometry-dependent table is rebuilt before validated openLevel.
 */
export function buildSourceLevel(document: LevelDocument, options: SourceBuildOptions) {
  validateRemovedNativeSwitches(document);
  const source = document.roomSource;
  const controllerCopies = options.controllerCopies ?? compileControllerCopies(source.controllerCopies ?? []);
  validateMovingControllerCopies(document, controllerCopies);
  const switchCopies = options.switchCopies ?? compileSwitchCopies(source.switchCopies ?? []);
  validateSwitchCopies(document, switchCopies);
  const plans = options.authoredControllers ?? source.rooms.flatMap(room => { const plan = compileSourceMovingRoom(document, room); return plan ? [plan] : []; });
  const topology = compileRoomTopology(document, source);
  const moving = new Set(document.mechanisms.map(item => resolveAnimationRoom(document, item.zone, item.kind === 'lift')));
  for (const room of source.rooms) if (room.originZone === null && hasNativeWallMaterials(room.originZone, room.wallSources)) {
    if (moving.has(room.recordZone) && mechanismsForRoom(document, room.recordZone).some(mechanism =>
      !controllerCopies.some(copy => copy.room === room.id && copy.kind === mechanism.kind && copy.id === mechanism.id))) {
      const dependencies = mechanismsForRoom(document, room.recordZone).map(item => mechanismDependencies(document, item.kind, item.id));
      const owners = [...new Set(dependencies.flatMap(item => item.graphicsRooms))].sort((a, b) => a - b);
      throw new Error(`Room ${room.id} copies a moving door or lift. Allocate its independent controller and wall attachments before building; the original controller cannot drive a room copy. Its live graphics belong to ${owners.map(id => `Zone ${id}`).join(', ')}. Inspect Attachments on the original mechanism to locate its moving walls and trigger boundaries.`);
    }
    if (activeNativeSwitches(document).some(item => item.zone === room.recordZone && !room.retiredCopiedSwitches?.includes(item.id) && !switchCopies.some(copy => copy.room === room.id && copy.id === item.id))) throw new Error(`Room ${room.id} copies a switch. Allocate and link its independent activation record before building.`);
  }
  for (const copy of controllerCopies) {
    const cells = topology.cells.filter(cell => cell.sourceId === copy.room);
    if (cells.length !== 1) throw new Error('A copied moving room needs one convex controller volume. Grouped moving cells require explicit synchronization.');
    const cell = cells[0]!;
    if (airHeightsChanged(document, cell) || closedPortalEdges(document, cell).length || unassignedBoundaryEdges(topology, cell).length)
      throw new Error('A copied moving room needs its original heights and boundary attachments. Copy its adjoining doorway rooms together before building.');
  }
  for (const cell of topology.cells) {
    if (!airHeightsChanged(document, cell)) continue;
    const template = document.zones.find(zone => zone.id === cell.recordZone)!;
    if (cell.originZone !== null && moving.has(cell.originZone))
      throw new Error('Changing a moving door or lift height needs synchronized animation ranges. Restore its original heights before building.');
    if ([...template.lower, ...template.upper].some(record => record.type === 5))
      throw new Error('Changing curved-wall heights needs explicit curve height attachments. Restore the original heights before building.');
  }
  for (const cell of topology.cells) if (closedPortalEdges(document, cell).length && cell.originZone !== null && moving.has(cell.originZone))
    throw new Error('Closing a doorway in a moving door or lift needs synchronized animation attachments. Restore the doorway before building.');
  for (const cell of topology.cells) if (unassignedBoundaryEdges(topology, cell).length && cell.originZone !== null && moving.has(cell.originZone))
    throw new Error('New boundaries in a moving door or lift need synchronized animation attachments. Restore its outline before building.');
  for (const cell of topology.cells) cell.neighbours.forEach((neighbour, edge) => {
    if (neighbour < 0) return;
    const target = topology.cells.find(target => target.id === neighbour)!;
    if (target.sourceId === cell.sourceId) return;
    if (!newPortalTarget(document, topology, cell, edge) || heightPortalTarget(document,topology,cell,edge)) return;
    if ((cell.originZone !== null && moving.has(cell.originZone)) || (target.originZone !== null && moving.has(target.originZone)))
      throw new Error('A new doorway into a moving door or lift needs synchronized wall animation attachments before building. Choose a static room boundary.');
  });
  const baseGraphics = buildCellGraphics(document, source, topology, options.newRoomGraphics, options.closingRoomGraphics);
  const layerGraphics = authorSeeThroughBarriers(document, authorRoomLayers(document, baseGraphics));
  const paintedGraphics = paintCellPolygons(source, paintCellWalls(document, source, layerGraphics));
  const geometry = authorWaterGraphics(document, authorSkyCeilings(document, paintedGraphics));
  const lighting = buildGeometryLighting(document, source, geometry, options.newRoomLighting);
  const collision = compileCollisionGeometry(barrierCollisionTopology(document, lighting.graphics.topology));
  const authored = buildAuthoredMovingRooms(document, initialMovingGraphics(document, lighting.graphics, plans), collision, plans);
  const aligned = buildWallSettings(document, authored.graphics);
  const switches = buildAuthoredSwitches(document, aligned, lighting.bindings);
  const rebindMoving = (target: AnimationGraphicTarget) => {
    const record = aligned.streams.get(target.room)!.lower[target.index]!;
    const index = switches.graphics.streams.get(target.room)!.lower.indexOf(record);
    if (index < 0) throw new Error('A switch replaced a moving controller attachment.');
    return { ...target, index };
  };
  const movingAnimations = authored.authored.map(item => ({ ...item, animation: { ...item.animation,
    target: rebindMoving(item.animation.target!), walls: item.animation.walls.map(wall => ({ ...wall, target: rebindMoving(wall.target!) })) } }));
  const attachments = buildAnimationAttachments(document, switches.graphics, collision, controllerCopies, switchCopies);
  const animations = syncWallSettingsAnimations(document, attachments.graphics, appendSourceWater(document, attachments.graphics, appendAuthoredSwitches(appendAuthoredMovingAnimations(attachments.tables, movingAnimations), switches.records)), attachments.selectTarget);
  const graphics = attachments.graphics;
  const points = buildPointLighting(document, graphics.topology.points, new Map([...lighting.bindings, ...switches.addedLighting, ...attachments.addedLighting]));
  const previousOrder = new Map(topology.cells.filter(cell => cell.originZone !== null).map(cell => [cell.id,
    document.zones.find(zone => zone.id === cell.originZone)!.visibleZones.flatMap(id => topology.cells.filter(target => target.originZone === id).map(target => target.id))]));
  const visibility = buildSourceVisibility(graphics.topology, graphics.pointReferences, previousOrder);
  const prefix = buildDataPrefix(document, { points, collision: collision.segments }, topology.cells.length);
  const references = buildNavigation(document, graphics.topology, buildRoomDataReferences(document, source, graphics.topology, prefix));
  const rooms = buildZoneRecords(references.data, topology.cells.map((cell, index) => ({ id: cell.id,
    header: references.headers.get(cell.id)!, boundaries: collision.boundaries[index]!, cornerWalls: collision.cornerWalls[index]!,
    ...visibility.rooms.get(cell.id)! })));
  const generated = compileGeneratedGraphics(document, topology.cells.map(cell => ({ sourceId: cell.sourceId, id: cell.id, dataOffset: rooms.offsets[cell.id]!,
    ...graphics.streams.get(cell.id)! })), animations, attachments.selectTarget);
  const originalExit = source.exitRoom !== undefined ? -1 : document.exitZone ?? checkedInteger(options.resolvedExit, -1, document.zones.length - 1, 'Source exit room');
  let exitZone = originalExit;
  if (originalExit >= 0) {
    const destinations = topology.cells.filter(cell => cell.originZone === originalExit);
    if (!destinations.length) throw new Error('The exit room was deleted. Choose another level exit before building.');
    exitZone = destinations[0]!.id;
  }
  if (source.exitRoom !== undefined) {
    const destination = topology.cells.find(cell => cell.sourceId === source.exitRoom);
    if (!destination) throw new Error('The exit room was deleted. Choose another level exit before building.');
    exitZone = destination.id;
  }
  const owners = buildDrawTopology(source, graphics.topology).owners;
  const groups = topology.cells.map(cell => owners.get(cell.id)!);
  const compiled = openLevel(`${document.name} compiled`, { data: rooms.data, graphics: generated.bytes, clips: visibility.clips }, document.playtestSlot, exitZone,
    groups.some((group, index) => group !== index) ? groups : undefined);
  return { document: compiled, topology: graphics.topology, references: references.references, surfaceSources: generated.records };
}
