import { expect, it } from 'vitest';
import { setRoomHeadroom } from '../src/model/roomHeadroom';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { roomSettings } from '../src/model/roomHeights';
import { editRoomSource, execute, undo, redo } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';
import { buildPlayableLevel } from '../src/model/playableBuild';
it('a positive room size raises only the ceiling, saves native planes and is one undoable edit', () => {
  const { document, from } = roomHeightFixture(), before = roomSettings(document, from), size = before.floor - before.ceiling + 20;
  const command = editRoomSource(setRoomHeadroom(document, document.roomSource, from.id, false, size));
  let history = execute({ document, past: [], future: [] }, command);
  const saved = decodeProject(encodeProject(history.document)), settings = roomSettings(saved, saved.roomSource.rooms.find(r => r.id === from.id)!);
  expect(settings).toMatchObject({ floor: before.floor, ceiling: before.ceiling - 20, upperFloor: before.upperFloor, upperCeiling: before.upperCeiling, water: before.water });
  expect(saved.roomSource.rooms.find(r => r.id === from.id)!.heights).toEqual({ floor: before.floor, ceiling: before.ceiling - 20, upperFloor: before.upperFloor, upperCeiling: before.upperCeiling });
  expect(history.past).toHaveLength(1); history = undo(history); expect(history.document.roomSource).toBe(document.roomSource);
  history = redo(history); expect(roomSettings(history.document, history.document.roomSource.rooms.find(r => r.id === from.id)!).ceiling).toBe(before.ceiling - 20);
});
it('lower and upper headroom preserve both floors, the other room and independent water', () => {
  const { document, from } = roomHeightFixture(twoStoreyHeightTemplate()), before = roomSettings(document, from);
  for (const upper of [false, true]) {
    const source = setRoomHeadroom(document, document.roomSource, from.id, upper, before[upper ? 'upperFloor' : 'floor'] - before[upper ? 'upperCeiling' : 'ceiling'] - .25).after;
    const settings = roomSettings(document, source.rooms.find(r => r.id === from.id)!);
    expect(settings.floor).toBe(before.floor); expect(settings.upperFloor).toBe(before.upperFloor); expect(settings.water).toBe(before.water);
    expect(settings[upper ? 'ceiling' : 'upperCeiling']).toBe(before[upper ? 'ceiling' : 'upperCeiling']);
    expect(settings[upper ? 'upperCeiling' : 'ceiling']).toBe(before[upper ? 'upperCeiling' : 'ceiling'] + .25);
  }
});
it('rejects invalid sizes, missing storeys and overlapping planes before changing source', () => {
  const { document, from } = roomHeightFixture();
  for (const value of [0, -1, .1, NaN, Infinity]) expect(() => setRoomHeadroom(document, document.roomSource, from.id, false, value)).toThrow();
  expect(() => setRoomHeadroom(document, document.roomSource, from.id, true, 100)).toThrow('no upper');
  const two = roomHeightFixture(twoStoreyHeightTemplate()), before = roomSettings(two.document, two.from);
  expect(() => setRoomHeadroom(two.document, two.document.roomSource, two.from.id, false, before.floor - before.upperFloor + 1)).toThrow('upper floor');
});
it('the compiled native ceiling changes while its collision floor stays fixed', () => {
  const { document, from } = roomHeightFixture(), before = roomSettings(document, from);
  const source = setRoomHeadroom(document, document.roomSource, from.id, false, before.floor - before.ceiling + 20).after;
  const built = buildPlayableLevel({ ...document, roomSource: source }, -1), id = built.topology!.cells.find(c => c.sourceId === from.id)!.id;
  expect(built.document.zones.find(z => z.id === id)).toMatchObject({ ceiling: before.ceiling - 20, floor: before.floor, water: before.water });
});
