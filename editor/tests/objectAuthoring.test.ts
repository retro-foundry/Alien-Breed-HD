import { expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { BinaryReader } from '../src/model/binary';
import { applyCommand, execute, numberPatch, redo, undo } from '../src/model/commands';
import { createObject, floorMatchedObjectHeight, objectSlotLimit, objectTemplates } from '../src/model/objectAuthoring';
import { openLevel } from '../src/model/level';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeGameLevel, encodeProject } from '../src/model/project';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { originalFiles } from './helpers';
import { convexParts } from '../src/model/geometry';
import { runtimeObjectPosition } from '../src/model/sourceObjects';

it('appends native actors across every original prefix without changing pool/avatar records or authored geometry', () => {
  for (let level = 1; level <= 16; level++) {
    const document = openLevel(`Level ${level}`, originalFiles(level)), template = objectTemplates(document).find(item => item.zone >= 0 && !item.upper)!;
    expect(template).toBeDefined(); const room = document.roomSource.rooms.find(room => room.originZone === template.zone)!;
    const part = convexParts(room.vertices)[0]!;
    const position = runtimeObjectPosition({ x: part.reduce((sum, point) => sum + point.x, 0) / part.length, z: part.reduce((sum, point) => sum + point.z, 0) / part.length });
    const proposal = createObject(document, { template: template.id, room: room.id, ...position, y: template.y });
    const changed = applyCommand(document, proposal.command), before = new BinaryReader(document.files.data, 'data'), after = new BinaryReader(changed.files.data, 'data');
    expect(changed.objects).toHaveLength(document.objects.length + 1);
    expect(changed.objects.at(-1)).toMatchObject({ id: proposal.id, cid: before.i16(20), ...position, y: template.y, type: template.type });
    expect(after.i16(20)).toBe(before.i16(20) + 1);
    expect(changed.objects.at(-1)!.cid).toBeLessThan(after.i16(20)); // native pool expansion starts here
    for (const field of [34, 38, 46, 50]) {
      expect(after.i32(field)).toBe(before.i32(field) + 8);
      expect(changed.files.data.slice(after.i32(field), after.i32(field) + 64)).toEqual(document.files.data.slice(before.i32(field), before.i32(field) + 64));
    }
    for (const old of document.objects) {
      const actual = changed.objects[old.id]!;
      expect(actual.cid).toBe(old.cid === before.i16(20) ? old.cid + 1 : old.cid);
      expect({ ...actual, offset: old.offset, cid: old.cid, pointOffset: old.pointOffset }).toEqual(old);
    }
    expect(changed.points.map(({ x, z }) => ({ x, z }))).toEqual(document.points.map(({ x, z }) => ({ x, z })));
    expect(changed.roomSource).toBe(document.roomSource);
    expect(changed.original).toBe(document.original);
  }
});

it('builds new rooms/actors, retains source metadata and supports table undo after an intervening byte edit is undone', () => {
  const fixture = roomHeightFixture(), template = objectTemplates(fixture.document).find(item => item.type === 1)!;
  const y = floorMatchedObjectHeight(fixture.document, template, fixture.from.id);
  const proposal = createObject(fixture.document, { template: template.id, room: fixture.from.id, x: 20128.5, z: 20192.25, y });
  let history = execute({ document: fixture.document, past: [], future: [] }, proposal.command);
  const actor = history.document.objects.at(-1)!;
  history = execute(history, { label: 'Edit new height', patches: [numberPatch(history.document, 'data', actor.offset + 4, -15, 2)] });
  expect(history.document.objects.at(-1)!.y).toBe(-7.5);
  history = undo(history); history = undo(history);
  expect(history.document.files).toEqual(fixture.document.files);
  expect(history.document.sourceObjects).toBe(fixture.document.sourceObjects);
  history = redo(history);
  const saved = decodeProject(encodeProject(history.document)), built = buildPlayableLevel(saved, -1);
  expect(saved.objects.at(-1)).toMatchObject({ x: 20128.5, z: 20192.25, y, type: 1 });
  expect(saved.sourceObjects!.at(-1)).toMatchObject({ id: proposal.id, room: fixture.from.id });
  expect(built.document.objects.at(-1)).toMatchObject({ x: 20128.5, z: 20192.25, y, type: 1, zone: built.document.starts[0]!.zone });
  expect(encodeGameLevel(built.document).length).toBeGreaterThan(0);
  expect(() => applyCommand(history.document, proposal.command)).toThrow();
});

it('copies floor-relative height between storeys and enforces native half-unit object heights', () => {
  const fixture = roomHeightFixture(twoStoreyHeightTemplate()), template = objectTemplates(fixture.document).find(item => item.type === 1)!;
  const y = floorMatchedObjectHeight(fixture.document, template, fixture.from.id, true);
  const command = createObject(fixture.document, { template: template.id, room: fixture.from.id, x: 20128, z: 20192, y, upper: true }).command;
  expect(applyCommand(fixture.document, command).objects.at(-1)).toMatchObject({ upper: true, y });
  expect(() => createObject(fixture.document, { template: template.id, room: fixture.from.id, x: 20128, z: 20192, y: 0.25 })).toThrow(/half units/);
});

it('rejects avatars, projectiles, outside positions and missing upper floors without partial writes', () => {
  const { document, from } = roomHeightFixture(), files = structuredClone(document.files), template = objectTemplates(document).find(item => item.type === 1)!;
  const base = { template: template.id, room: from.id, x: 20128, z: 20192, y: -8 };
  const reader = new BinaryReader(document.files.data, 'data');
  for (const change of [{ template: -1 }, ...[34, 46].map(field => ({ template: document.objects.find(item => item.offset === reader.i32(field))!.id })), { x: -30000 }, { room: 'missing' }, { upper: true }])
    expect(() => createObject(document, { ...base, ...change })).toThrow();
  expect(document.files).toEqual(files);
});

it('exposes the actual native record capacity including reserved pools', () => {
  const header = readFileSync(new URL('../../src/game_state.h', import.meta.url), 'utf8');
  expect(objectSlotLimit).toBe(Number(/^#define MAX_OBJECTS\s+(\d+)/m.exec(header)![1]));
});

it('fills native capacity without aliasing actors with the trailing projectile padding point', () => {
  const fixture = roomHeightFixture(); let document = fixture.document;
  const template = objectTemplates(document).find(item => item.type === 1)!;
  const placement = { template: template.id, room: fixture.from.id, x: 20128, z: 20192, y: -8 };
  while (document.objects.length < objectSlotLimit) {
    document = applyCommand(document, createObject(document, placement).command);
    const reader = new BinaryReader(document.files.data, 'data');
    expect(document.objects.at(-1)!.cid).toBeLessThan(reader.i16(20));
  }
  expect(() => createObject(document, placement)).toThrow(/250/);
  const built = buildPlayableLevel(document, -1).document;
  expect(built.objects).toHaveLength(objectSlotLimit);
  expect(encodeGameLevel(built).length).toBeGreaterThan(0);
});
