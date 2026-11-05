/* AMOS leveld.asc PLACECONTPT, CPTMOVE, CPTNEARTO[UPPER]ZONE and
 * CONNECTCONT. twolev.links is a source-only 100x100 byte bank, not the
 * renderer's CONNECT_TABLE. Point IDs below survive table compaction. */
import { checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { EditCommand } from './commands';
import { pointInRoomOutline } from './geometry';
import { roomSettings } from './roomHeights';
import { navigationSightlines } from './navigationSightlines';

export interface NavigationPoint { readonly id: number; readonly x: number; readonly z: number; readonly y: number; readonly upper: boolean; readonly room?: string }
export interface NavigationAssignment { readonly room: string; readonly upper: boolean; readonly point: number | null }
export interface Navigation { readonly points: readonly NavigationPoint[]; readonly nextId: number; readonly assignments: readonly NavigationAssignment[]; readonly links?: readonly number[] }

function keys(item: object, allowed: readonly string[]) { if (Object.keys(item).some(key => !allowed.includes(key))) throw new Error('Unsupported control-point source fields.'); }
export function restoreNavigation(value: unknown): Navigation {
  if (!value || typeof value !== 'object' || !('points' in value) || !Array.isArray(value.points) || value.points.length > 100 || !('assignments' in value) || !Array.isArray(value.assignments) || value.assignments.length > 512 || !('nextId' in value)) throw new Error('Damaged control-point source.');
  keys(value, ['points', 'nextId', 'assignments', 'links']);
  const nextId = checkedInteger(value.nextId as number, 0, Number.MAX_SAFE_INTEGER - 1, 'Next control-point identity'), ids = new Set<number>();
  const points = value.points.map((item: unknown): NavigationPoint => {
    if (!item || typeof item !== 'object' || !('id' in item) || !('x' in item) || !('z' in item) || !('y' in item) || !('upper' in item) || typeof item.upper !== 'boolean') throw new Error('Damaged control-point position.');
    keys(item, ['id', 'x', 'z', 'y', 'upper', 'room']);
    const id = checkedInteger(item.id as number, 0, nextId - 1, 'Control-point identity');
    if (ids.has(id)) throw new Error('Duplicate control-point identity.'); ids.add(id);
    const room = 'room' in item ? item.room : undefined;
    if (room !== undefined && (typeof room !== 'string' || !room.length || room.length > 128)) throw new Error('Invalid control-point room.');
    return Object.freeze({ id, x: checkedInteger(item.x as number, -32768, 32767, 'Control-point X'), z: checkedInteger(item.z as number, -32768, 32767, 'Control-point Z'), y: checkedInteger(item.y as number, -32768, 32767, 'Control-point height'), upper: item.upper, ...(room === undefined ? {} : { room: room as string }) });
  });
  const assigned = new Set<string>();
  const assignments = value.assignments.map((item: unknown): NavigationAssignment => {
    if (!item || typeof item !== 'object' || !('room' in item) || typeof item.room !== 'string' || !item.room.length || item.room.length > 128 || !('upper' in item) || typeof item.upper !== 'boolean' || !('point' in item) || (item.point !== null && !ids.has(item.point as number))) throw new Error('Invalid room control-point assignment.');
    keys(item, ['room', 'upper', 'point']); const key = `${item.room}:${item.upper}`;
    if (assigned.has(key)) throw new Error('Duplicate room control-point assignment.'); assigned.add(key);
    return Object.freeze({ room: item.room, upper: item.upper, point: item.point as number | null });
  });
  let links: readonly number[] | undefined;
  if ('links' in value) {
    if (!Array.isArray(value.links) || value.links.length !== 10000 || value.links.some(byte => !Number.isInteger(byte) || byte < 0 || byte > 255)) throw new Error('twolev.links needs exactly 10000 bytes.');
    links = Object.freeze([...value.links]);
    for (let a = 0; a < points.length; a++) for (let b = 0; b < points.length; b++) {
      const x = links[a * 100 + b]!, y = links[b * 100 + a]!;
      if (a === b ? x !== 0 : x > 2 || y > 2 || (x === 0) !== (y === 0)) throw new Error(`Invalid control-point link ${a + 1} / ${b + 1}. Use AMOS link pairs 0/0, 1/1, 1/2, 2/1 or 2/2.`);
    }
  }
  return Object.freeze({ points: Object.freeze(points), nextId, assignments: Object.freeze(assignments), ...(links ? { links } : {}) });
}

export function editableNavigation(document: LevelDocument): Navigation {
  return document.roomSource.navigation ?? restoreNavigation({ nextId: document.controlPoints.length, points: document.controlPoints.map(point => ({ id: point.id, x: point.x, z: point.z, y: point.y, upper: point.upper !== 0 })), assignments: [] });
}

export function navigationAssignment(document: LevelDocument, room: string, upper: boolean): number | null {
  const navigation = editableNavigation(document), assigned = navigation.assignments.find(item => item.room === room && item.upper === upper);
  if (assigned) return assigned.point;
  const source = document.roomSource.rooms.find(item => item.id === room);
  if (!source) throw new Error('Choose an existing room for the control-point assignment.');
  const template = document.zones.find(zone => zone.id === source.recordZone)!;
  const id = upper ? template.upperControl : template.control;
  return id >= 0 && id < document.controlPoints.length ? id : null;
}

function position(document: LevelDocument, room: string, upper: boolean, x: number, z: number) {
  checkedInteger(x, -32768, 32767, 'Control-point X'); checkedInteger(z, -32768, 32767, 'Control-point Z');
  const owner = document.roomSource.rooms.find(item => item.id === room);
  if (!owner || !pointInRoomOutline(owner.vertices, { x, z })) throw new Error('Place the control point inside its selected room.');
  const settings = roomSettings(document, owner);
  if (upper && (!settings.upper.length || settings.upperFloor <= settings.upperCeiling)) throw new Error('This room has no upper storey for the control point.');
  // LEVELSAVE computes each point's height from its owning storey's midpoint.
  const y = Math.trunc(((upper ? settings.upperFloor : settings.floor) + (upper ? settings.upperCeiling : settings.ceiling)) / 2);
  return { x, z, room, upper, y: checkedInteger(y, -32768, 32767, 'Control-point height') };
}

function change(document: LevelDocument, value: Navigation, label: string): EditCommand {
  const navigation = restoreNavigation(value);
  if (JSON.stringify(navigation) === JSON.stringify(document.roomSource.navigation)) return { label, patches: [] };
  return { label, patches: [], roomSource: { before: document.roomSource, after: Object.freeze({ ...document.roomSource, navigation }) } };
}

export function placeNavigationPoint(document: LevelDocument, id: number | null, room: string, upper: boolean, x: number, z: number): EditCommand {
  const navigation = editableNavigation(document), location = position(document, room, upper, x, z);
  if (id === null && navigation.points.length >= 100) throw new Error('The original control-point table allows at most 100 points.');
  if (id !== null && !navigation.points.some(point => point.id === id)) throw new Error('Select an existing control point to move.');
  const point = { id: id ?? navigation.nextId, ...location };
  // A previously unused bank row must not introduce links from old inactive bytes.
  const links = navigation.links && [...navigation.links];
  if (id === null && links) for (let i = 0; i < 100; i++) { links[navigation.points.length * 100 + i] = 0; links[i * 100 + navigation.points.length] = 0; }
  return change(document, { ...navigation, points: id === null ? [...navigation.points, point] : navigation.points.map(old => old.id === id ? point : old), nextId: navigation.nextId + (id === null ? 1 : 0), ...(links ? { links } : {}) }, id === null ? 'Add control point' : 'Move control point');
}

export function assignNavigationPoint(document: LevelDocument, room: string, upper: boolean, point: number | null): EditCommand {
  const navigation = editableNavigation(document), owner = document.roomSource.rooms.find(item => item.id === room);
  if (!owner) throw new Error('Choose an existing room for the control-point assignment.');
  const settings = roomSettings(document, owner);
  if (upper && (!settings.upper.length || settings.upperFloor <= settings.upperCeiling)) throw new Error('This room has no upper storey to assign.');
  if (point !== null && !navigation.points.some(item => item.id === point)) throw new Error('Choose an existing control point.');
  return change(document, { ...navigation, assignments: [...navigation.assignments.filter(item => item.room !== room || item.upper !== upper), { room, upper, point }] }, 'Assign room control point');
}

export function removeNavigationPoint(document: LevelDocument, id: number): EditCommand {
  const navigation = editableNavigation(document), index = navigation.points.findIndex(point => point.id === id);
  if (index < 0) throw new Error('Select an existing control point to delete.');
  const assignments = [...navigation.assignments];
  for (const room of document.roomSource.rooms) for (const upper of [false, true]) if (navigationAssignment(document, room.id, upper) === id) {
    const current = assignments.findIndex(item => item.room === room.id && item.upper === upper);
    if (current >= 0) assignments.splice(current, 1);
    assignments.push({ room: room.id, upper, point: null });
  }
  const retained = navigation.points.map((_, i) => i).filter(i => i !== index), links = navigation.links && [...navigation.links];
  if (links) for (let a = 0; a < 100; a++) for (let b = 0; b < 100; b++) links[a * 100 + b] = a < retained.length && b < retained.length ? navigation.links![retained[a]! * 100 + retained[b]!]! : 0;
  return change(document, { ...navigation, points: navigation.points.filter(point => point.id !== id), assignments, ...(links ? { links } : {}) }, 'Delete control point and its links');
}

export function setNavigationGraph(document: LevelDocument, bytes: Uint8Array): EditCommand {
  return change(document, { ...editableNavigation(document), links: [...bytes] }, 'Import control-point link graph');
}
export function setNavigationLink(document: LevelDocument, from: number, to: number, forward: number, reverse: number): EditCommand {
  const navigation = editableNavigation(document), a = navigation.points.findIndex(point => point.id === from), b = navigation.points.findIndex(point => point.id === to);
  if (a < 0 || b < 0 || a === b) throw new Error('Choose two different existing control points.');
  if (!navigation.links) throw new Error('Import twolev.links or create a new link graph first.');
  const links = [...navigation.links]; links[a * 100 + b] = forward; links[b * 100 + a] = reverse;
  return change(document, { ...navigation, links }, 'Change control-point link');
}

export function autoLinkNavigation(document: LevelDocument): EditCommand {
  const navigation = editableNavigation(document);
  if (!navigation.links) throw new Error('Import twolev.links or create a new link graph first.');
  const links = [...navigation.links], visible = navigationSightlines(document);
  for (let a = 0; a < navigation.points.length; a++) for (let b = a + 1; b < navigation.points.length; b++) {
    // AUTOLINK never replaces authored physical links (including one-way pairs).
    if (links[a * 100 + b] === 1 || links[b * 100 + a] === 1) continue;
    const value = visible(navigation.points[a]!, navigation.points[b]!) ? 2 : 0;
    links[a * 100 + b] = value; links[b * 100 + a] = value;
  }
  return change(document, { ...navigation, links }, 'Rebuild visual control-point links');
}

export function autoAssignNavigation(document: LevelDocument, rooms: readonly string[]): EditCommand {
  const navigation = editableNavigation(document), visible = navigationSightlines(document), assignments = [...navigation.assignments];
  if (!rooms.length || new Set(rooms).size !== rooms.length) throw new Error('Choose different rooms for automatic control-point assignment.');
  for (const id of rooms) {
    const room = document.roomSource.rooms.find(room => room.id === id);
    if (!room) throw new Error('Choose existing rooms for automatic control-point assignment.');
    const center = { x: Math.trunc(room.vertices.reduce((sum, p) => sum + p.x, 0) / room.vertices.length), z: Math.trunc(room.vertices.reduce((sum, p) => sum + p.z, 0) / room.vertices.length) };
    let nearest: number | null = null, distance = 100000000; // CPTNEARTOZONE's DDD
    for (const point of navigation.points) {
      const squared = (point.x - center.x) ** 2 + (point.z - center.z) ** 2;
      if (squared < distance && visible(point, center)) { nearest = point.id; distance = squared; }
    }
    if (nearest === null) throw new Error(`No visible control point was found for ${id}. Place a nearby point or choose its lower assignment manually.`);
    const previous = assignments.findIndex(item => item.room === id && !item.upper);
    if (previous >= 0) assignments.splice(previous, 1);
    assignments.push({ room: id, upper: false, point: nearest });
  }
  return change(document, { ...navigation, assignments }, 'Assign nearest visible lower control points');
}

export function navigationPosition(document: LevelDocument, point: NavigationPoint) {
  return point.room === undefined ? point : { ...point, ...position(document, point.room, point.upper, point.x, point.z) };
}

export function partitionNavigation(document: LevelDocument, replaced: string, members: readonly string[]): Navigation | undefined {
  const navigation = document.roomSource.navigation;
  if (!navigation) return undefined;
  const points = navigation.points.map(point => {
    if (point.room !== replaced) return point;
    const owner = members.map(id => document.roomSource.rooms.find(room => room.id === id)!).find(room => pointInRoomOutline(room.vertices, point));
    if (!owner) throw new Error('A control point lies outside the partitioned room. Move it inside before creating the opening.');
    if (point.upper && owner.upperStorey === false) throw new Error('Move upper control points outside the shaft before removing their upper floor.');
    return navigationPosition(document, { ...point, room: owner.id });
  });
  const assignments = navigation.assignments.flatMap(item => item.room !== replaced ? [item] : members.map(room => ({ ...item, room })));
  return restoreNavigation({ ...navigation, points, assignments });
}

/** Clipboard source is captured. Duplicate owned points and internal links;
 * external assignments still refer to the same existing project point. */
export function pasteNavigation(document: LevelDocument, captured: LevelDocument, rooms: ReadonlyMap<string, string>, x: number, z: number): Navigation | undefined {
  if (!captured.roomSource.navigation) return document.roomSource.navigation;
  const original = editableNavigation(captured), target = editableNavigation(document);
  const mapping = new Map<number, number>(); let nextId = target.nextId;
  const added = original.points.filter(point => point.room !== undefined && rooms.has(point.room)).map(point => {
    const id = nextId++; mapping.set(point.id, id);
    return { ...point, id, room: rooms.get(point.room!)!, x: point.x + x, z: point.z + z };
  });
  if (added.length + target.points.length > 100) throw new Error('Copying these rooms exceeds the 100 control-point limit.');
  const assignments = [...target.assignments];
  for (const [room, mapped] of rooms) {
    const owner = captured.roomSource.rooms.find(item => item.id === room)!, settings = roomSettings(captured, owner);
    for (const upper of settings.upper.length && settings.upperFloor > settings.upperCeiling ? [false, true] : [false]) {
      const point = navigationAssignment(captured, room, upper), mappedPoint = point === null ? null : mapping.get(point) ?? point;
      if (mappedPoint !== null && ![...target.points, ...added].some(point => point.id === mappedPoint)) throw new Error('A copied room uses a deleted external control point. Restore or reassign it before pasting.');
      assignments.push({ room: mapped, upper, point: mappedPoint });
    }
  }
  const links = target.links && [...target.links];
  if (added.length && original.links && !links) throw new Error('Restore or import the captured control-point link graph before pasting these linked rooms.');
  if (links) {
    for (let i = target.points.length; i < target.points.length + added.length; i++) for (let j = 0; j < 100; j++) { links[i * 100 + j] = 0; links[j * 100 + i] = 0; }
    if (original.links) {
      const points = [...target.points, ...added];
      for (const [id, mapped] of mapping) for (const other of original.points) {
        const a = original.points.findIndex(point => point.id === id), b = original.points.findIndex(point => point.id === other.id);
        const row = points.findIndex(point => point.id === mapped), column = points.findIndex(point => point.id === (mapping.get(other.id) ?? other.id));
        if (column < 0) { if (original.links[a * 100 + b] || original.links[b * 100 + a]) throw new Error('A copied control-point link uses a deleted external point. Restore or remove that link before pasting.'); continue; }
        links[row * 100 + column] = original.links[a * 100 + b]!; links[column * 100 + row] = original.links[b * 100 + a]!;
      }
    }
  }
  return restoreNavigation({ points: [...target.points, ...added], nextId, assignments, ...(links ? { links } : {}) });
}
