import { partitionPortalStates } from './portalStates';
import { remapBoundaryMaterials, restoreBoundaryMaterials } from './boundaryMaterials';
import { roomPlaneRecords } from './roomPlanes';
import { polygonMaterialCurrent, polygonMaterialMode, setPolygonMaterial } from './polygonMaterials';
import { checkedInteger } from './binary';
import { convexParts, normaliseRoomOutline, pointInRoomOutline, signedArea, type Position } from './geometry';
import type { LevelDocument } from './level';
import type { EditCommand } from './commands';
import { roomSettings } from './roomHeights';
import { partitionNavigation } from './navigation';
import { restoreRoomSource, roomSourceSnapshot, sourceWallForEdge } from './roomSource';
import { assertStaticRoom, ceilingIsSky } from './roomStoreys';
import { editableStarts, placeSourceStart } from './sourceStarts';
import { editableObjects, objectIsActive, objectPlayer, placeSourceObject } from './sourceObjects';
import { editableTeleport, placeSourceTeleport } from './sourceTeleports';
import { setRoomAppearance } from './roomAppearance';
import { buildPlayableLevel } from './playableBuild';

export interface RoomOpening { left: number; right: number; top: number; bottom: number }

function clip(input: readonly Position[], axis: 'x' | 'z', limit: number, below: boolean): Position[] {
  const output: Position[] = [], inside = (p: Position) => below ? p[axis] <= limit : p[axis] >= limit;
  for (let i = 0; i < input.length; i++) {
    const a = input[i]!, b = input[(i + 1) % input.length]!, from = inside(a), to = inside(b);
    if (from) output.push({ x: a.x, z: a.z });
    if (from !== to) {
      const t = (limit - a[axis]) / (b[axis] - a[axis]);
      const point = { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t };
      for (const value of [point.x, point.z]) if (Math.abs(value - Math.round(value)) > 1e-7) throw new Error('The opening cuts a slanted wall between whole coordinates. Adjust its bounds to meet whole map coordinates.');
      output.push({ x: Math.round(point.x), z: Math.round(point.z) });
    }
  }
  return output;
}

export function openingParts(outline: readonly Position[], opening: RoomOpening): Position[][] {
  for (const [label, value] of Object.entries(opening)) checkedInteger(value, -32768, 32767, `Opening ${label}`);
  const { left, right, top, bottom } = opening;
  if (left >= right || top >= bottom) throw new Error('Opening width and depth must be positive.');
  const shaft = [{ x: left, z: top }, { x: left, z: bottom }, { x: right, z: bottom }, { x: right, z: top }];
  if (shaft.some(point => !pointInRoomOutline(outline, point))) throw new Error('All four corners of the opening must be inside the room.');
  if (shaft.some(point => outline.some((a, index) => {
    const b = outline[(index + 1) % outline.length]!;
    return (b.x - a.x) * (point.z - a.z) === (b.z - a.z) * (point.x - a.x);
  }))) throw new Error('Leave space between the opening and the room boundary.');
  const pieces = convexParts(outline);
  let covered = 0;
  const surrounding = pieces.flatMap(part => {
    const middle = clip(clip(part, 'x', left, false), 'x', right, true);
    const inside = clip(clip(middle, 'z', top, false), 'z', bottom, true);
    covered += Math.abs(signedArea(inside));
    return [clip(part, 'x', left, true), clip(part, 'x', right, false), clip(middle, 'z', top, true), clip(middle, 'z', bottom, false)].filter(part => part.length >= 3 && Math.abs(signedArea(part)) > 0).map(normaliseRoomOutline);
  });
  if (Math.abs(covered - (right - left) * (bottom - top)) > 1e-7) throw new Error('The opening crosses outside the room. Move it clear of inward corners.');
  return [normaliseRoomOutline(shaft), ...surrounding];
}

/** Native movement.c checkotherwalls/FindRoom chooses air bands when crossing
 * a zone boundary. A playable opening is a tall single-storey shaft surrounded
 * by two-storey zones, rather than an invisible floor with unchanged collision.
 * No new collision rule or vertical-transition behavior is added to the game. */
export function createRoomOpening(document: LevelDocument, id: string, opening: RoomOpening,
  appearance: { lowerWall: number; upperWall: number | null; lightingPoint: number }, kind: 'storeys' | 'sky' = 'storeys'): { command: EditCommand; rooms: readonly string[] } {
  const original = document.roomSource.rooms.find(room => room.id === id);
  if (!original) throw new Error('Select a two-storey room before creating an opening.');
  assertStaticRoom(document, original);
  const settings = roomSettings(document, original);
  const hasUpper = settings.upper.length > 0 && settings.upperFloor > settings.upperCeiling;
  if (kind === 'storeys' && !hasUpper) throw new Error('Add an upper storey before creating an opening between storeys.');
  if (document.roomSource.exitRoom === original.id || (document.roomSource.exitRoom === undefined && document.exitZone !== null && document.exitZone === original.originZone)) throw new Error('Choose another level exit before dividing this exit room into an opening and surrounding zones.');
  const parts = openingParts(original.vertices, opening);
  const ids = parts.map((_, index) => index === 0 ? id : `room-${document.roomSource.nextId + index - 1}`);
  const rooms = parts.map((vertices, index) => ({ ...original, ...(original.cornerLights ? { cornerLights: original.cornerLights.filter(light => vertices.some(point => point.x === light.x && point.z === light.z)) } : {}), id: ids[index]!, originZone: index === 0 ? original.originZone : null, vertices,
    wallSources: vertices.map((a, i) => sourceWallForEdge(original, a, vertices[(i + 1) % vertices.length]!)),
    ...(() => {
      const boundary = remapBoundaryMaterials(original, vertices.map((start, index) => ({ start, end: vertices[(index + 1) % vertices.length]! })));
      const active = new Set(boundary?.edges ?? []);
      return { wallSettings: original.wallSettings?.filter(item => !(index === 0 && kind === 'storeys' && item.upper) && (!('boundary' in item) || active.has(item.boundary))), boundaryMaterials: boundary ? restoreBoundaryMaterials({ ...boundary, overrides: boundary.overrides.filter(item => active.has(item.edge)) }, vertices.length) : undefined,
        portals: original.portals?.filter(item => active.has(item.boundary)) };
    })(),
    ...(index === 0 && kind === 'sky' ? { sky: hasUpper ? [original.sky?.[0] ?? false, true] : [true, false] } : {}),
    ...(index === 0 && kind === 'storeys' ? { upperStorey: false, heights: { floor: settings.floor, ceiling: settings.upperCeiling, upperFloor: settings.upperCeiling, upperCeiling: settings.upperCeiling },
      sky: [ceilingIsSky(document, original, true), false], wallMaterials: original.wallMaterials?.filter(item => !item.upper), polygonMaterials: original.polygonMaterials?.filter(item => !item.upper), planeSources: original.planeSources?.filter(item => !item.upper) } : {}) }));
  const snapshot = roomSourceSnapshot(document.roomSource);
  const source = restoreRoomSource(document, { ...snapshot, version: Math.max(snapshot.version, 12), controllerCopies: snapshot.controllerCopies ?? [], switchCopies: snapshot.switchCopies ?? [],
    rooms: [...document.roomSource.rooms.filter(room => room.id !== id), ...rooms.map(room => Object.fromEntries(Object.entries(room).filter(([, value]) => value !== undefined)))], nextId: document.roomSource.nextId + parts.length - 1 });
  let working = { ...document, roomSource: partitionPortalStates(source, id, ids) };
  const navigation = partitionNavigation(working, id, ids);
  if (navigation) working = { ...working, roomSource: { ...working.roomSource, navigation } };
  if (kind === 'storeys') {
    const upperRoofs = roomPlaneRecords(document, original, true).filter(record => [2, 9, 11].includes(record.type));
    const lowerRoofs = roomPlaneRecords(document, original, false).filter(record => [2, 9, 11].includes(record.type));
    if (upperRoofs.length === 1 && lowerRoofs.length === 1) {
      const target = { room: id, upper: true, record: upperRoofs[0]!.offset };
      const material = polygonMaterialCurrent(document, target)!;
      working = { ...working, roomSource: setPolygonMaterial(working, { room: id, upper: false, record: lowerRoofs[0]!.offset },
        { texture: material.texture, scale: material.scale, mode: polygonMaterialMode(document, target) }).after };
    }
  }
  const starterSurfaces = document.roomAppearances?.find(item => item.room === id)?.surfaces;
  for (const room of rooms) working = { ...working, roomAppearances: setRoomAppearance(working, { ...(starterSurfaces ? { surfaces: starterSurfaces } : {}), ...appearance, room: room.id, upperWall: room.upperStorey ? null : room.upperStorey === false ? null : appearance.upperWall }) };
  const destination = (point: Position) => rooms.find(room => pointInRoomOutline(room.vertices, point))!;
  const belongs = (room: string | null | undefined, zone: number) => room ? room === id : original.originZone !== null && zone === original.originZone;
  for (const start of editableStarts(document)) if (belongs(start.room, start.zone)) {
    const room = destination(start); if (!room) throw new Error('A player start is outside the original room. Move it inside before creating the opening.');
    working = { ...working, sourceStarts: placeSourceStart(working, start.player, room.id, start.x, start.z) };
  }
  for (const object of editableObjects(document)) if (objectIsActive(object) && objectPlayer(document, object) === undefined && belongs(object.room, object.zone)) {
    const room = destination(object); if (!room) throw new Error(`Object ${object.id} is outside the original room. Move it before creating the opening.`);
    if (kind === 'storeys' && object.upper && room.id === id) throw new Error(`Move object ${object.id} away from the proposed opening on the upper floor first.`);
    working = { ...working, sourceObjects: placeSourceObject(working, object.id, room.id, object.x, object.z) };
  }
  for (const room of document.roomSource.rooms) {
    const link = editableTeleport(document, room);
    if (link.destination !== id) continue;
    const target = destination(link); if (!target) throw new Error('An incoming teleporter lands outside the room. Repair its destination first.');
    working = { ...working, sourceTeleports: placeSourceTeleport(working, room.id, target.id, link.x, link.z) };
  }
  if (original.wallSettings?.length || original.cornerLights?.some(light => light.record !== undefined)) {
    // Partition only valid source settings, and retain each face override on
    // the parts which actually draw it. New cut walls have their own identity.
    buildPlayableLevel(document, -1);
    const planned = new Map(working.roomSource.rooms.filter(room => ids.includes(room.id)).map(room => [room.id, room.wallSettings]));
    const lights = new Map(working.roomSource.rooms.filter(room => ids.includes(room.id)).map(room => [room.id, room.cornerLights]));
    const bare = { ...working, roomSource: { ...working.roomSource, rooms: working.roomSource.rooms.map(room => {
      if (!planned.has(room.id)) return room;
      // Material masks and column overrides can need each other. Provenance
      // depends only on geometry, so inspect the unpainted streams together.
      const { wallSettings: _settings, wallMaterials: _materials, cornerLights: _lights, ...retained } = room;
      const generic = room.cornerLights?.filter(light => light.record === undefined);
      return { ...retained, ...(generic?.length ? { cornerLights: generic } : {}), ...(room.boundaryMaterials ? { boundaryMaterials: { ...room.boundaryMaterials, overrides: [] } } : {}) };
    }) } };
    const artifact = buildPlayableLevel(bare, -1);
    working = { ...working, roomSource: { ...working.roomSource, rooms: working.roomSource.rooms.map(room => {
      if (!planned.has(room.id)) return room;
      const cells = new Set(artifact.topology!.cells.filter(cell => cell.sourceId === room.id).map(cell => cell.id));
      const settings = planned.get(room.id)?.filter(item => artifact.surfaceSources!.some(record => cells.has(record.room) && record.upper === item.upper
        && ('record' in item ? record.sourceOffset === item.record : record.sourceOffset === undefined && record.boundaryIndex !== undefined && room.boundaryMaterials?.edges[record.boundaryIndex] === item.boundary)));
      const cornerLights = lights.get(room.id)?.filter(light => light.record === undefined || artifact.surfaceSources!.some(identity => {
        if (!cells.has(identity.room) || identity.sourceOffset !== light.record) return false;
        const zone = artifact.document.zones[identity.room]!, record = (identity.upper ? zone.upper : zone.lower).find(item => item.offset === identity.offset)!;
        return record.points.some(id => artifact.document.points[id]!.x === light.x && artifact.document.points[id]!.z === light.z);
      }));
      const { wallSettings: _settings, cornerLights: _lights, ...retained } = room;
      return { ...retained, ...(settings?.length ? { wallSettings: settings } : {}), ...(cornerLights?.length ? { cornerLights } : {}) };
    }) } };
  }
  // Validate the complete exported artifact before publishing one history entry.
  buildPlayableLevel(working, -1);
  const command: EditCommand = { label: kind === 'sky' ? 'Create sky opening' : 'Create opening between storeys', patches: [],
    roomSource: { before: document.roomSource, after: working.roomSource }, roomAppearances: { before: document.roomAppearances, after: working.roomAppearances },
    sourceStarts: { before: document.sourceStarts, after: working.sourceStarts }, sourceObjects: { before: document.sourceObjects, after: working.sourceObjects },
    sourceTeleports: { before: document.sourceTeleports, after: working.sourceTeleports } };
  return { command, rooms: ids };
}
