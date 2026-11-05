import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, redo, setSourceStart, undo } from '../src/model/commands';
import { autoAssignNavigation, autoLinkNavigation, assignNavigationPoint, editableNavigation, navigationAssignment, placeNavigationPoint, removeNavigationPoint, restoreNavigation, setNavigationGraph, setNavigationLink } from '../src/model/navigation';
import { navigationSightlines } from '../src/model/navigationSightlines';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { decodeProject, encodeGameLevel, encodeProject, openArchive } from '../src/model/project';
import { removeSourceRooms, roomSourceSnapshot, setRoomHeights, translateSourceRooms } from '../src/model/roomSource';
import { createRoomOpening } from '../src/model/roomOpening';
import { storeyMaterials } from '../src/model/roomStoreys';
import { pointInRoomOutline } from '../src/model/geometry';
import { roomSettings } from '../src/model/roomHeights';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { BinaryReader } from '../src/model/binary';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('keeps level %i original control coordinates, objects and sentinel bytes when adding only a source graph', level => {
  const original = openLevel('Navigation', originalFiles(level)), source = applyCommand(original, setNavigationGraph(original, new Uint8Array(10000)));
  const artifact = buildPlayableLevel(source, -1), built = artifact.document;
  expect(built.controlPoints).toEqual(original.controlPoints);
  expect(built.objects.map(o => [o.id, o.x, o.z, o.y])).toEqual(original.objects.map(o => [o.id, o.x, o.z, o.y]));
  for (const zone of built.zones) {
    const template = original.zones.find(z => z.id === artifact.topology!.cells.find(cell => cell.id === zone.id)!.recordZone)!;
    expect([zone.control, zone.upperControl]).toEqual([template.control, template.upperControl]);
  }
});

it('adds independent lower/upper controls, relocates every absolute pointer and emits their assigned native indices', () => {
  const { document, from } = roomHeightFixture(twoStoreyHeightTemplate()), first = document.controlPoints.length;
  let source = applyCommand(document, placeNavigationPoint(document, null, from.id, false, 20064, 20064));
  source = applyCommand(source, placeNavigationPoint(source, null, from.id, true, 20192, 20192));
  source = applyCommand(source, assignNavigationPoint(source, from.id, false, first));
  source = applyCommand(source, assignNavigationPoint(source, from.id, true, first + 1));
  const built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === from.id)!, native = built.document;
  expect(native.zones.find(zone => zone.id === cell.id)).toMatchObject({ control: first, upperControl: first + 1 });
  const settings = roomSettings(source, from);
  expect(native.controlPoints.slice(-2).map(({ x, z, y, upper }) => ({ x, z, y, upper }))).toEqual([
    { x: 20064, z: 20064, y: Math.trunc((settings.floor + settings.ceiling) / 2), upper: 0 },
    { x: 20192, z: 20192, y: Math.trunc((settings.upperFloor + settings.upperCeiling) / 2), upper: 1 },
  ]);
  const before = new BinaryReader(document.files.data, 'source'), after = new BinaryReader(native.files.data, 'compiled');
  for (const field of [22, 30, 34, 38, 42, 46, 50]) expect(after.i32(field)).toBe(before.i32(field) + 16);
  const withoutNavigation = buildPlayableLevel(document, -1).document;
  expect(native.objects.map(o => [o.id, o.x, o.z, o.y, o.zone])).toEqual(withoutNavigation.objects.map(o => [o.id, o.x, o.z, o.y, o.zone]));
  const archive = openArchive(encodeGameLevel(native), 'navigation.zip')[0]!;
  expect(openLevel(archive.name, archive.files).controlPoints).toEqual(native.controlPoints);
});

it('recomputes authored midpoint heights on builds and rejects misplaced/deleted owners instead of retaining stale coordinates', () => {
  const { document, from } = roomHeightFixture();
  let source = applyCommand(document, placeNavigationPoint(document, null, from.id, false, 20064, 20064));
  const heights = roomSettings(source, from);
  source = applyCommand(source, editRoomSource(setRoomHeights(source, source.roomSource, from.id, { floor: heights.floor + 8, ceiling: heights.ceiling, upperFloor: heights.upperFloor, upperCeiling: heights.upperCeiling })));
  expect(buildPlayableLevel(source, -1).document.controlPoints.at(-1)!.y).toBe(Math.trunc((heights.floor + 8 + heights.ceiling) / 2));
  expect(() => removeSourceRooms(source.roomSource, [from.id])).toThrow(/control points/);
  expect(() => placeNavigationPoint(source, null, from.id, true, 20064, 20064)).toThrow(/no upper storey/);
  expect(() => placeNavigationPoint(source, null, from.id, false, 100, 100)).toThrow(/inside/);
  expect(() => buildPlayableLevel({ ...source, roomSource: { ...source.roomSource, rooms: source.roomSource.rooms.filter(room => room.id !== from.id) } }, -1)).toThrow();
});

it('preserves graph direction, stable identities, compaction, assignments and atomic Undo/project history', () => {
  const { document, from } = roomHeightFixture(), first = document.controlPoints.length;
  let source = document;
  for (const x of [20064, 20128, 20192]) source = applyCommand(source, placeNavigationPoint(source, null, from.id, false, x, 20064));
  const bank = new Uint8Array(10000); bank[9999] = 239;
  source = applyCommand(source, setNavigationGraph(source, bank));
  source = applyCommand(source, setNavigationLink(source, first, first + 2, 1, 2));
  source = applyCommand(source, assignNavigationPoint(source, from.id, false, first + 1));
  const command = removeNavigationPoint(source, first + 1), history = execute({ document: source, past: [], future: [] }, command);
  const removed = history.document.roomSource.navigation!;
  expect(removed.points.at(-1)!.id).toBe(first + 2);
  expect(removed.links![first * 100 + first + 1]).toBe(1); expect(removed.links![(first + 1) * 100 + first]).toBe(2);
  expect(navigationAssignment(history.document, from.id, false)).toBeNull();
  expect(undo(history).document.roomSource).toBe(source.roomSource);
  expect(redo(undo(history)).document.roomSource).toBe(history.document.roomSource);
  const restored = decodeProject(encodeProject(history.document));
  expect(restored.roomSource.navigation).toEqual(removed); expect(roomSourceSnapshot(restored.roomSource).version).toBe(15);
  expect(buildPlayableLevel(restored, -1).document.controlPoints).toHaveLength(document.controlPoints.length + 2);
  expect(() => applyCommand(history.document, command)).toThrow(/room source changed/);
});

it('copies owned control points, both-storey assignments and internal/external links with their room group', () => {
  const { document, from, to } = roomHeightFixture(), first = document.controlPoints.length;
  let source = applyCommand(document, placeNavigationPoint(document, null, from.id, false, 20064, 20064));
  source = applyCommand(source, placeNavigationPoint(source, null, to.id, false, 21064, 20064));
  source = applyCommand(source, setNavigationGraph(source, new Uint8Array(10000)));
  source = applyCommand(source, setNavigationLink(source, first, first + 1, 1, 2));
  source = applyCommand(source, assignNavigationPoint(source, from.id, false, first));
  const result = pasteRooms(source, copyRooms(source, [from.id]), -10000, 0), pasted = applyCommand(source, result.command), nav = pasted.roomSource.navigation!;
  expect(nav.points.at(-1)).toMatchObject({ id: first + 2, room: result.ids[0], x: 10064, z: 20064 });
  expect(navigationAssignment(pasted, result.ids[0]!, false)).toBe(first + 2);
  expect(nav.links![(first + 2) * 100 + first + 1]).toBe(1); expect(nav.links![(first + 1) * 100 + first + 2]).toBe(2);
  expect(buildPlayableLevel(pasted, -1).document.controlPoints).toHaveLength(first + 3);
});

it('enforces the native 100-point table and exact source link-bank states', () => {
  const { document, from } = roomHeightFixture();
  let source = document;
  while (editableNavigation(source).points.length < 100) source = applyCommand(source, placeNavigationPoint(source, null, from.id, false, 20064, 20064));
  expect(() => placeNavigationPoint(source, null, from.id, false, 20064, 20064)).toThrow(/100 points/);
  expect(() => setNavigationGraph(source, new Uint8Array(9999))).toThrow(/10000 bytes/);
  expect(() => setNavigationLink(source, 0, 1, 1, 1)).toThrow(/graph first/);
  source = applyCommand(source, setNavigationGraph(source, new Uint8Array(10000)));
  expect(() => setNavigationLink(source, 0, 1, 1, 0)).toThrow(/Invalid control-point link/);
  expect(() => setNavigationLink(source, 0, 0, 1, 1)).toThrow(/two different/);
  expect(() => restoreNavigation({ ...source.roomSource.navigation, surprise: 1 })).toThrow(/Unsupported/);
  expect(() => restoreNavigation({ ...source.roomSource.navigation, points: [{ id: 0, x: 0, z: 0, y: 0, upper: false }, { id: 0, x: 0, z: 0, y: 0, upper: false }] })).toThrow(/Duplicate/);
});

it('moves owned controls with whole rooms and retains graph identities', () => {
  const { document, from } = roomHeightFixture(), id = document.controlPoints.length;
  let source = applyCommand(document, placeNavigationPoint(document, null, from.id, false, 20064, 20064));
  source = applyCommand(source, editRoomSource(translateSourceRooms(source.roomSource, [from.id], -1000, 0)));
  source = applyCommand(source, setSourceStart(source, 1, from.id, 19128, 20128));
  expect(source.roomSource.navigation!.points.at(-1)).toMatchObject({ id, x: 19064, z: 20064, room: from.id });
  expect(buildPlayableLevel(source, -1).document.controlPoints.at(-1)).toMatchObject({ x: 19064, z: 20064 });
});

it('rebinds points and room assignments across skylight partitions while rejecting stranded upper shaft controls', () => {
  const { document, from } = roomHeightFixture(), materials = storeyMaterials(document, from), first = document.controlPoints.length;
  let source = applyCommand(document, placeNavigationPoint(document, null, from.id, false, 20032, 20032));
  source = applyCommand(source, placeNavigationPoint(source, null, from.id, false, 20128, 20128));
  source = applyCommand(source, assignNavigationPoint(source, from.id, false, first));
  const bounds = { left: 20064, right: 20192, top: 20064, bottom: 20192 };
  const proposal = createRoomOpening(source, from.id, bounds, { lowerWall: materials.walls[0]!.offset, upperWall: null, lightingPoint: materials.walls[0]!.points[0]! }, 'sky');
  const opened = applyCommand(source, proposal.command), nav = opened.roomSource.navigation!;
  for (const point of nav.points.filter(p => p.room !== undefined)) expect(pointInRoomOutline(opened.roomSource.rooms.find(r => r.id === point.room)!.vertices, point)).toBe(true);
  expect(new Set(nav.points.slice(-2).map(p => p.room)).size).toBe(2);
  expect(nav.assignments.filter(item => item.point === first).length).toBeGreaterThan(1);
  expect(buildPlayableLevel(opened, -1).document.controlPoints).toHaveLength(document.controlPoints.length + 2);
  const upper = roomHeightFixture(twoStoreyHeightTemplate()), styles = storeyMaterials(upper.document, upper.from);
  const withUpper = applyCommand(upper.document, placeNavigationPoint(upper.document, null, upper.from.id, true, 20128, 20128));
  expect(() => createRoomOpening(withUpper, upper.from.id, bounds, { lowerWall: styles.walls[0]!.offset, upperWall: styles.walls[0]!.offset, lightingPoint: styles.walls[0]!.points[0]! })).toThrow(/upper control points outside the shaft/);
});

it('rebuilds visual links while preserving physical directions and atomically assigns nearest visible lower references', () => {
  const { document, from, to } = roomHeightFixture(), first = document.controlPoints.length;
  let source = applyCommand(document, placeNavigationPoint(document, null, from.id, false, 20064, 20064));
  source = applyCommand(source, placeNavigationPoint(source, null, to.id, false, 21064, 20064));
  source = applyCommand(source, setNavigationGraph(source, new Uint8Array(10000)));
  expect(applyCommand(source, autoLinkNavigation(source)).roomSource.navigation!.links![first * 100 + first + 1]).toBe(0);
  source = applyCommand(source, editRoomSource(translateSourceRooms(source.roomSource, [to.id], -744, 0)));
  source = applyCommand(source, autoLinkNavigation(source));
  expect(source.roomSource.navigation!.links![first * 100 + first + 1]).toBe(2);
  source = applyCommand(source, autoAssignNavigation(source, [from.id, to.id]));
  expect(navigationAssignment(source, from.id, false)).toBe(first);
  expect(navigationAssignment(source, to.id, false)).toBe(first + 1);
  source = applyCommand(source, setNavigationLink(source, first, first + 1, 1, 2));
  source = applyCommand(source, editRoomSource(translateSourceRooms(source.roomSource, [to.id], 1000, 0)));
  expect(applyCommand(source, autoLinkNavigation(source)).roomSource.navigation!.links![first * 100 + first + 1]).toBe(1);
  expect(() => autoAssignNavigation(source, [source.roomSource.rooms[0]!.id])).toThrow(/No visible control point/);
});

it('tests visual paths through multiple shared edges, occluded bends, concavity and corner-only contact', () => {
  const rectangle = (x: number, z: number, w = 100, h = 100) => [{ x, z }, { x: x + w, z }, { x: x + w, z: z + h }, { x, z: z + h }];
  const visible = (outlines: { x: number; z: number }[][]) => navigationSightlines({ roomSource: { nextId: outlines.length, rooms: outlines.map((vertices, id) => ({ id: String(id), originZone: null, recordZone: 0, vertices, wallSources: vertices.map(() => null) })) } });
  const straight = visible([rectangle(0, 0), rectangle(100, 0), rectangle(200, 0)]);
  expect(straight({ x: 20, z: 50 }, { x: 280, z: 50 })).toBe(true);
  expect(straight({ x: 280, z: 50 }, { x: 20, z: 50 })).toBe(true);
  expect(straight({ x: 20, z: 50 }, { x: 20, z: 50 })).toBe(true);
  expect(straight({ x: -1, z: 50 }, { x: -1, z: 50 })).toBe(false);
  expect(visible([rectangle(0, 0), rectangle(101, 0)])({ x: 50, z: 50 }, { x: 150, z: 50 })).toBe(false);
  expect(visible([rectangle(0, 0), rectangle(100, 100)])({ x: 50, z: 50 }, { x: 150, z: 150 })).toBe(false);
  expect(visible([rectangle(0, 0), rectangle(100, 0), rectangle(100, 100)])({ x: 20, z: 50 }, { x: 150, z: 180 })).toBe(false);
  const concave = visible([[{ x: 0, z: 0 }, { x: 200, z: 0 }, { x: 200, z: 50 }, { x: 50, z: 50 }, { x: 50, z: 200 }, { x: 0, z: 200 }]]);
  expect(concave({ x: 150, z: 25 }, { x: 25, z: 150 })).toBe(false);
  expect(concave({ x: 25, z: 25 }, { x: 25, z: 150 })).toBe(true);
});

it('remaps inherited native room references after deleting an earlier imported point', () => {
  const { document, from } = roomHeightFixture(), first = document.controlPoints.length;
  let source = document;
  for (const x of [20064, 20128, 20192]) source = applyCommand(source, placeNavigationPoint(source, null, from.id, false, x, 20064));
  source = applyCommand(source, assignNavigationPoint(source, from.id, false, first + 2));
  const native = buildPlayableLevel(source, -1).document, room = native.roomSource.rooms.find(room => native.zones.find(zone => zone.id === room.originZone)!.control === first + 2)!;
  const changed = applyCommand(native, removeNavigationPoint(native, first)), artifact = buildPlayableLevel(changed, -1), cell = artifact.topology!.cells.find(cell => cell.sourceId === room.id)!;
  expect(artifact.document.zones.find(zone => zone.id === cell.id)!.control).toBe(first + 1);
  expect(artifact.document.controlPoints.at(-1)).toMatchObject({ x: 20192, z: 20064 });
});
