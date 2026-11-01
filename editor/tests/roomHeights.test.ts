import { expect, it } from 'vitest';
import { applyCommand, editRoomSource, execute, redo, undo } from '../src/model/commands';
import { beginRoomDraft, commitRoomDraft, moveSourceCorner, roomSourceSnapshot, setDraftCorners, setRoomHeights } from '../src/model/roomSource';
import { roomSettings } from '../src/model/roomHeights';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { roomHeightFields } from '../src/model/roomHeights';
import { closedPortalFixture } from './portalFixture';
import { editRoomAppearance } from '../src/model/commands';

it('builds room-specific quarter heights into native headers, polygons and full walls without changing the shared template', () => {
  const fixture = roomHeightFixture(), before = structuredClone(fixture.document.files);
  const template = fixture.document.zones[fixture.from.recordZone]!;
  const heights = { floor: 32.25, ceiling: -160.5, upperFloor: template.upperFloor, upperCeiling: template.upperCeiling };
  const command = editRoomSource(setRoomHeights(fixture.document, fixture.document.roomSource, fixture.from.id, heights));
  let history = execute({ document: fixture.document, past: [], future: [] }, command);
  const restored = decodeProject(encodeProject(history.document));
  expect(roomSourceSnapshot(restored.roomSource).version).toBe(3);
  expect(restored.roomSource.rooms.find(room => room.id === fixture.from.id)!.heights).toEqual(heights);
  const built = buildPlayableLevel(restored, -1);
  const zone = built.document.zones.find(zone => zone.id === built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!.id)!;
  const other = built.document.zones.find(zone => zone.id === built.topology!.cells.find(cell => cell.sourceId === fixture.to.id)!.id)!;
  expect(zone).toMatchObject(heights);
  for (const record of zone.lower.filter(record => record.wall)) expect(record).toMatchObject({ top: heights.ceiling, bottom: heights.floor });
  for (const record of zone.lower.filter(record => record.polygon && record.type !== 7)) expect(record.height).toBe(record.type === 2 ? heights.ceiling : heights.floor);
  expect(other.floor).toBe(fixture.original.zones[0]!.floor); expect(other.ceiling).toBe(fixture.original.zones[0]!.ceiling);
  expect(restored.files).toEqual(before);
  history = undo(history); expect(history.document.roomSource).toBe(fixture.document.roomSource);
  history = redo(history); expect(buildPlayableLevel(history.document, -1).document.files).toEqual(built.document.files);
  const room = history.document.roomSource.rooms.find(room => room.id === fixture.from.id)!;
  const moved = moveSourceCorner(history.document.roomSource, room.id, 0, { x: 19992, z: 20000 });
  expect(moved.after.rooms.find(item => item.id === room.id)!.heights).toEqual(heights);
  const redrawn = commitRoomDraft(history.document.roomSource, setDraftCorners(beginRoomDraft(history.document, history.document.roomSource, room.recordZone, room.id), moved.after.rooms.find(item => item.id === room.id)!.vertices));
  expect(redrawn.after.rooms.find(item => item.id === room.id)!.heights).toEqual(heights);
});

it('rejects silently downgrading height source to an older schema', () => {
  const { document, from } = roomHeightFixture(), template = document.zones[from.recordZone]!;
  const source = applyCommand(document, editRoomSource(setRoomHeights(document, document.roomSource, from.id,
    { floor: template.floor + 16, ceiling: template.ceiling, upperFloor: template.upperFloor, upperCeiling: template.upperCeiling })));
  const entries = unzipSync(encodeProject(source)), snapshot = JSON.parse(strFromU8(entries['room-source.json']!));
  expect(snapshot.version).toBe(3); entries['room-source.json'] = strToU8(JSON.stringify({ ...snapshot, version: 2 }));
  expect(() => decodeProject(zipSync(entries))).toThrow(/unsupported source fields/);
});

it('can restore native closed-door heights after rejecting an unsupported moving-volume height change', () => {
  const document = openLevel('door heights', originalFiles(1)), moving = document.mechanisms.find(item => item.kind === 'door')!;
  const room = document.roomSource.rooms.find(room => room.originZone === moving.zone)!, template = document.zones[room.recordZone]!;
  const original = Object.fromEntries(roomHeightFields.map(field => [field, template[field]]));
  const heights = { ...original, floor: template.floor + 32, ceiling: template.ceiling - 32 };
  const source = applyCommand(document, editRoomSource(setRoomHeights(document, document.roomSource, room.id, heights)));
  expect(() => buildPlayableLevel(source, -1)).toThrow(/moving door or lift height/);
  const restored = applyCommand(source, editRoomSource(setRoomHeights(source, source.roomSource, room.id, original)));
  expect(restored.roomSource.rooms).toEqual(document.roomSource.rooms);
  expect(buildPlayableLevel(restored, -1).document).toBe(restored);
});

it('rebuilds both sides of an existing imported doorway after its destination height changes', () => {
  const fixture = closedPortalFixture(), template = fixture.baseline.zones[fixture.removed.recordZone]!;
  let source = applyCommand(fixture.baseline, editRoomSource(setRoomHeights(fixture.baseline, fixture.baseline.roomSource, fixture.removed.id,
    { floor: -32, ceiling: -160, upperFloor: template.upperFloor, upperCeiling: template.upperCeiling })));
  expect(() => buildPlayableLevel(source, -1)).not.toThrow();
  source = applyCommand(source, editRoomAppearance(source, fixture.appearance));
  const ownWall = source.zones[fixture.removed.recordZone]!.lower.find(record => record.wall)!;
  source = applyCommand(source, editRoomAppearance(source, { room: fixture.removed.id, lowerWall: ownWall.offset, upperWall: null, lightingPoint: ownWall.points[0]! }));
  const built = buildPlayableLevel(source, -1), target = built.document.zones.find(zone => zone.id === built.topology!.cells.find(cell => cell.sourceId === fixture.removed.id)!.id)!;
  expect(target).toMatchObject({ floor: -32, ceiling: -160 });
  const owner = built.document.zones.find(zone => zone.id === built.topology!.cells.find(cell => cell.sourceId === fixture.owner.id)!.id)!;
  const crossing = (record: typeof owner.lower[number]) => record.wall && record.points.every(id => {
    const p = built.document.points[id]!; return (p.x - fixture.x) * fixture.normal.x + (p.z - fixture.z) * fixture.normal.z === 0;
  });
  expect(owner.lower.some(record => crossing(record) && record.top === -32 && record.bottom === 0)).toBe(true);
  expect(target.lower.some(record => crossing(record) && record.top === -160 && record.bottom === owner.ceiling)).toBe(true);
});

it('rejects crossed boundaries, non-quarter heights, unsupported upper layers and unknown fields atomically', () => {
  const { document, from } = roomHeightFixture(), settings = roomSettings(document, from);
  const base = { floor: settings.floor, ceiling: settings.ceiling, upperFloor: settings.upperFloor, upperCeiling: settings.upperCeiling };
  for (const heights of [{ ...base, floor: base.ceiling }, { ...base, floor: 0.1 }, { ...base, floor: 8192 },
    { ...base, upperCeiling: -300 }, { ...base, unknown: 1 }]) expect(() => setRoomHeights(document, document.roomSource, from.id, heights)).toThrow();
  expect(setRoomHeights(document, document.roomSource, from.id, base).after).toBe(document.roomSource);
});

it('writes independent upper and lower height bands from an actual two-storey material template', () => {
  const original = openLevel('upper template', originalFiles(1));
  const zone = original.zones[twoStoreyHeightTemplate()]!;
  expect(zone).toBeDefined();
  const { document, from } = roomHeightFixture(zone.id);
  const heights = { floor: zone.floor + 16, ceiling: zone.ceiling, upperFloor: zone.upperFloor, upperCeiling: zone.upperCeiling - 16 };
  const source = applyCommand(document, editRoomSource(setRoomHeights(document, document.roomSource, from.id, heights)));
  const built = buildPlayableLevel(source, -1), actual = built.document.zones.find(zone => zone.id === built.topology!.cells.find(cell => cell.sourceId === from.id)!.id)!;
  expect(actual).toMatchObject(heights);
  expect(actual.upper.filter(record => record.type === 2).every(record => record.height === heights.upperCeiling)).toBe(true);
  expect(actual.upper.some(record => record.wall && record.top === heights.upperCeiling && record.bottom === heights.upperFloor)).toBe(true);
});
