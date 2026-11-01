import { authoredMovingSurfaceOwner } from '../src/model/authoredMovingRoom';
import { surfaceMaterialTarget } from '../src/model/wallMaterials';
import { expect, it } from 'vitest';
import { importedMovingFixture } from './importedMovingFixture';
import { decodeProject, encodeProject } from '../src/model/project';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { objectPlayer } from '../src/model/sourceObjects';
import { movingWallChoices, setSourceMovingRoom } from '../src/model/sourceMovingRoom';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { validateAnimationReferences } from '../src/model/animationBuild';

it.each(Array.from({ length: 16 }, (_, index) => index + 1))('retains imported identities and existing controller state while converting rooms in Level %i', level => {
  for (const kind of ['door', 'lift'] as const) {
    const { document, changed, built, room, mechanism } = importedMovingFixture(level, kind);
    expect(changed.roomSource.rooms.find(item => item.id === room.id)!.originZone).toBe(room.originZone);
    expect(changed.files).toEqual(document.files);
    const saved = decodeProject(encodeProject(changed));
    expect(saved.roomSource.rooms.find(item => item.id === room.id)!.moving).toEqual(changed.roomSource.rooms.find(item => item.id === room.id)!.moving);
    expect(buildPlayableLevel(saved, -1).document.mechanisms.at(-1)).toBeDefined();
    expect(built.topology!.cells.find(cell => cell.id === mechanism.zone)!.sourceId).toBe(room.id);
    for (const old of document.mechanisms) {
      const actual = built.document.mechanisms.find(item => item.kind === old.kind && item.id === old.id)!;
      expect(actual).toMatchObject({ top: old.top, bottom: old.bottom, current: old.current, direction: old.direction, conditions: old.conditions, openRule: old.openRule, closeRule: old.closeRule });
    }
    for (const start of document.starts) {
      const actual = built.document.starts.find(item => item.player === start.player)!;
      expect(actual).toMatchObject({ x: start.x, z: start.z });
      expect(built.topology!.cells.find(cell => cell.id === actual.zone)!.originZone).toBe(start.zone);
    }
    for (const object of document.objects.filter(item => item.zone >= 0 && objectPlayer(document, item) === undefined))
      expect(built.document.objects[object.id]).toMatchObject({ cid: object.cid, x: object.x, z: object.z, y: object.y, type: object.type });
    for (const wall of mechanism.walls) {
      const owner = built.document.zones.find(zone => zone.lower.some(record => record.offset === wall.graphic))!;
      const record = owner.lower.find(record => record.offset === wall.graphic)!;
      const pick = { offset: record.offset, upper: false, type: record.type };
      expect(authoredMovingSurfaceOwner(built, owner.id, pick)).toEqual({ room: room.id, kind });
      expect(surfaceMaterialTarget(built, owner.id, pick)).toBeUndefined();
    }
    validateAnimationReferences(built.document);
  }
}, 30000);


it('can use a native material from another room without transferring that room or its animation ownership', () => {
  const fixture = importedMovingFixture(1, 'door');
  const own = fixture.document.zones[fixture.room.recordZone]!.lower.map(record => record.offset);
  const material = movingWallChoices(fixture.document, fixture.room).find(record => !own.includes(record.offset))!;
  expect(material).toBeDefined();
  const changed = applyCommand(fixture.document, editRoomSource(setSourceMovingRoom(fixture.document, fixture.room.id, { ...fixture.value, wall: material.offset })));
  const built = buildPlayableLevel(changed, -1), controller = built.document.mechanisms.filter(item => item.kind === 'door').at(-1)!;
  for (const attachment of controller.walls) {
    const graphic = built.document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === attachment.graphic)!;
    expect(graphic.wall!.texture).toBe(material.wall!.texture);
  }
  expect(changed.files).toEqual(fixture.document.files);
});
