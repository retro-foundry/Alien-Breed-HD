import { expect, it } from 'vitest';
import { resolveAnimationRoom } from '../src/model/animationRooms';
import { importedAnimationTables } from '../src/model/animationBuild';
import { openLevel } from '../src/model/level';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { planRoomReferences } from '../src/model/roomReferences';
import { removeSourceRooms } from '../src/model/roomSource';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { originalFiles } from './helpers';

it('preserves native raw-slot preference, rejects holes, and permits only lift/water file aliases', () => {
  const original = openLevel('original', originalFiles(1)), data = original.files.data.slice();
  const view = new DataView(data.buffer), last = original.zones.at(-1)!;
  view.setInt16(last.offset, 0);
  const conflict = openLevel('in-range alias', { ...original.files, data });
  expect(resolveAnimationRoom(conflict, 0, true)).toBe(0);
  expect(resolveAnimationRoom({ ...conflict, zones: conflict.zones.filter(zone => zone.id !== 0) }, 0, true)).toBeUndefined();
  view.setInt16(last.offset, 300);
  const alias = openLevel('out-of-range alias', { ...original.files, data });
  expect(resolveAnimationRoom(alias, 300, true)).toBe(last.id);
  expect(resolveAnimationRoom(alias, 300)).toBeUndefined();
  for (const id of [-1, 0.5, NaN, 301]) expect(resolveAnimationRoom(alias, id, true)).toBeUndefined();
});

it('compiles a lift file alias exactly like its canonical source and diagnoses removal of the resolved room', () => {
  const original = openLevel('original', originalFiles(1)), lift = original.mechanisms.find(item => item.kind === 'lift')!;
  const data = original.files.data.slice(), graphics = original.files.graphics.slice();
  new DataView(data.buffer).setInt16(original.zones[lift.zone]!.offset, 300);
  new DataView(graphics.buffer).setInt16(lift.offset + 12, 300);
  const alias = openLevel('lift alias', { ...original.files, data, graphics });
  const plan = planRoomReferences(alias, alias.roomSource);
  expect(plan.diagnostics.filter(item => item.severity === 'error')).toEqual([]);
  expect(plan.mechanisms.find(item => item.kind === 'lift' && item.id === lift.id)!.cells).not.toHaveLength(0);
  expect(buildSourceLevel(alias, { resolvedExit: -1 }).document.files)
    .toEqual(buildSourceLevel(original, { resolvedExit: -1 }).document.files);
  const room = alias.roomSource.rooms.find(room => room.originZone === lift.zone)!;
  const deleted = removeSourceRooms(alias.roomSource, [room.id]).after;
  expect(planRoomReferences(alias, deleted).diagnostics).toContainEqual(expect.objectContaining({
    severity: 'error', subject: `lift ${lift.id}`, message: expect.stringContaining('300 was deleted'),
  }));
  expect(alias.mechanisms.find(item => item.kind === 'lift' && item.id === lift.id)!.zone).toBe(300);
});

it('compiles water file aliases exactly like canonical source without inventing alias support for doors or switches', () => {
  const imported = openLevel('water', originalFiles(1));
  const selected = imported.zones.flatMap(zone => [...zone.lower, ...zone.upper].map(record => ({ zone: zone.id, record }))).find(item => item.record.type === 7)!;
  const importedTables = importedAnimationTables(imported).tables;
  const original = openLevel('canonical water control', { ...imported.files, graphics: buildGraphicsStreams(imported, undefined, undefined,
    { animations: { ...importedTables, water: importedTables.water.map((group, index) => ({ ...group,
      entries: index ? group.entries : [{ zone: selected.zone, graphic: selected.record.offset }] })) } }) });
  const { tables } = importedAnimationTables(original), entry = tables.water.flatMap(group => group.entries)[0]!;
  const data = original.files.data.slice(); new DataView(data.buffer).setInt16(original.zones[entry.zone]!.offset, 300);
  const graphics = original.files.graphics.slice();
  const view = new DataView(graphics.buffer);
  const waterStart = view.getInt32(4) + (tables.lifts ?? []).reduce((bytes, record) => bytes + 20 + record.walls.length * 10, 0) + 2;
  view.setInt16(waterStart + 14, 300);
  const alias = openLevel('water alias', { ...original.files, data, graphics });
  expect(buildSourceLevel(alias, { resolvedExit: -1 }).document.files)
    .toEqual(buildSourceLevel(original, { resolvedExit: -1 }).document.files);
  expect(resolveAnimationRoom(alias, 300)).toBeUndefined();
  expect(resolveAnimationRoom(alias, 300, true)).toBe(entry.zone);
});
