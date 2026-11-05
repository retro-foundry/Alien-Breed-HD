import { expect, it } from 'vitest';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { setUpperStorey, setSkyCeiling, storeyMaterials } from '../src/model/roomStoreys';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { roomSettings } from '../src/model/roomHeights';
import { createRoomOpening } from '../src/model/roomOpening';
import { addRoomPlane } from '../src/model/roomPlanes';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { setPolygonMaterial } from '../src/model/polygonMaterials';
import { nativePartialRoomCopyFixture } from './nativeRoomCopyFixture';
import { normalPlaneChoices } from '../src/model/roomPlanes';

it('adds an explicitly textured upper storey without replacing a partial lower floor', () => {
  const { document, copy } = nativePartialRoomCopyFixture(), materials = storeyMaterials(document, copy), settings = roomSettings(document, copy);
  const floor = normalPlaneChoices(document, 1)[0]!, ceiling = normalPlaneChoices(document, 2)[0]!;
  const added = applyCommand(document, editRoomSource(setUpperStorey(document, copy.id, {
    floor: settings.ceiling, ceiling: settings.ceiling - (settings.floor - settings.ceiling), wall: materials.walls[0]!.offset, lightingPoint: materials.walls[0]!.points[0]!, floorMaterial: floor.offset, ceilingMaterial: ceiling.offset,
  })));
  const original = buildPlayableLevel(document, -1), built = buildPlayableLevel(added, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === copy.id)!.id;
  expect(built.document.zones[cell]!.lower).toEqual(original.document.zones[cell]!.lower);
  expect(built.document.zones[cell]!.upper.some(record => record.type === 1 && record.texture === floor.texture)).toBe(true);
  expect(decodeProject(encodeProject(added)).roomSource).toEqual(added.roomSource);
});

it('removes an inherited upper storey and cuts an inherited two-storey opening without stale upper wall requirements', () => {
  const { document, from } = roomHeightFixture(twoStoreyHeightTemplate());
  const command = editRoomSource(setUpperStorey(document, from.id, null));
  const single = applyCommand(document, command), built = buildPlayableLevel(single, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === from.id)!;
  expect(built.document.zones[cell.id]!.upper).toHaveLength(0);
  expect(decodeProject(encodeProject(single)).roomSource).toEqual(single.roomSource);
  expect(applyCommand(single, command, true)).toEqual(document);
  const template = document.zones[from.recordZone]!;
  const proposal = createRoomOpening(document, from.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 },
    { lowerWall: template.lower.find(record => record.wall)!.offset, upperWall: template.upper.find(record => record.wall)!.offset, lightingPoint: template.lower.find(record => record.wall)!.points[0]! });
  expect(buildPlayableLevel(applyCommand(document, proposal.command), -1).document.zones).toHaveLength(document.zones.length + 6);
});

it('adds a native upper storey, paints it independently, saves, copies heights and undoes', () => {
  const { document, from } = roomHeightFixture();
  const materials = storeyMaterials(document, from), original = roomSettings(document, from);
  expect(materials.floor).toBeDefined(); expect(materials.ceiling).toBeDefined();
  const command = editRoomSource(setUpperStorey(document, from.id, { floor: original.ceiling, ceiling: original.ceiling - (original.floor - original.ceiling), wall: materials.walls[0]!.offset, lightingPoint: materials.walls[0]!.points[0]! }));
  const added = applyCommand(document, command), built = buildPlayableLevel(added, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === from.id)!;
  expect(built.document.zones[cell.id]!.upper.some(record => record.type === 1)).toBe(true);
  expect(built.document.zones[cell.id]!.upper.some(record => record.type === 2)).toBe(true);
  expect(built.document.zones[cell.id]!.upper.some(record => record.type === 4)).toBe(true);
  expect(built.document.zones[cell.id]!.upperFloor).toBe(original.ceiling);
  const upperTarget = { room: from.id, upper: true, record: materials.floor!.offset };
  const painted = applyCommand(added, editRoomSource(setPolygonMaterial(added, upperTarget, { texture: 257, scale: -2, mode: 'normal' })));
  const paintBuild = buildPlayableLevel(painted, -1);
  expect(paintBuild.document.zones[cell.id]!.upper.find(record => record.type === 1)!.texture).toBe(257);
  expect(paintBuild.document.zones[cell.id]!.lower.find(record => record.type === 1)!.texture).toBe(materials.floor!.texture);
  const copied = pasteRooms(painted, copyRooms(painted, [from.id]), 0, 1000);
  const copy = applyCommand(painted, copied.command);
  expect(buildPlayableLevel(copy, -1).document.zones.some(zone => zone.upper.some(record => record.type === 1 && record.texture === 257))).toBe(true);
  expect(decodeProject(encodeProject(copy)).roomSource).toEqual(copy.roomSource);
  expect(decodeProject(encodeProject(added)).roomSource).toEqual(added.roomSource);
  const opening = createRoomOpening(added, from.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 },
    { lowerWall: materials.walls[0]!.offset, upperWall: null, lightingPoint: materials.walls[0]!.points[0]! });
  const opened = applyCommand(added, opening.command), artifact = buildPlayableLevel(opened, -1);
  expect(opening.rooms).toHaveLength(5);
  const shaft = artifact.document.zones[artifact.topology!.cells.find(cell => cell.sourceId === from.id)!.id]!;
  expect(shaft.ceiling).toBe(original.ceiling - (original.floor - original.ceiling));
  expect(shaft.upper).toHaveLength(0);
  expect(shaft.boundaryIds.every(id => artifact.document.lines[id]!.connect >= 0)).toBe(true);
  expect(decodeProject(encodeProject(opened)).roomSource).toEqual(opened.roomSource);
  expect(applyCommand(opened, opening.command, true)).toEqual(added);
  expect(applyCommand(added, command, true)).toEqual(document);
});

it('creates a rectangular skylight and retains the ceiling around it', () => {
  const { document, from } = roomHeightFixture(), materials = storeyMaterials(document, from);
  const proposal = createRoomOpening(document, from.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 },
    { lowerWall: materials.walls[0]!.offset, upperWall: null, lightingPoint: materials.walls[0]!.points[0]! }, 'sky');
  const opened = applyCommand(document, proposal.command), built = buildPlayableLevel(opened, -1);
  for (const id of proposal.rooms) {
    const zone = built.document.zones[built.topology!.cells.find(cell => cell.sourceId === id)!.id]!;
    expect(zone.lower.some(record => record.type === 2)).toBe(id !== from.id);
    expect(zone.lower.some(record => record.type === 12)).toBe(id === from.id);
  }
  expect(decodeProject(encodeProject(opened)).roomSource).toEqual(opened.roomSource);
  expect(applyCommand(opened, proposal.command, true)).toEqual(document);
});

it('creates and paints a missing ceiling in an imported outdoor room', () => {
  const { document, from } = roomHeightFixture();
  const built = buildPlayableLevel(applyCommand(document, editRoomSource(setSkyCeiling(document, from.id, false, true))), -1);
  const zone = built.document.zones[built.topology!.cells.find(cell => cell.sourceId === from.id)!.id]!;
  const imported = built.document, room = imported.roomSource.rooms.find(room => room.originZone === zone.id)!;
  const material = imported.zones.flatMap(zone => zone.lower).find(record => record.type === 2)!;
  const command = editRoomSource(addRoomPlane(imported, room.id, false, material.offset));
  const added = applyCommand(imported, command);
  const target = { room: room.id, upper: false, record: material.offset };
  const painted = applyCommand(added, editRoomSource(setPolygonMaterial(added, target, { texture: 257, scale: -2, mode: 'normal' })));
  expect(buildPlayableLevel(painted, -1).document.zones[zone.id]!.lower.some(record => record.type === 2 && record.texture === 257)).toBe(true);
  expect(decodeProject(encodeProject(painted)).roomSource).toEqual(painted.roomSource);
  expect(applyCommand(added, command, true)).toEqual(imported);
});

it('authors a native sky ceiling and restores the material with Undo', () => {
  const { document, from } = roomHeightFixture();
  const command = editRoomSource(setSkyCeiling(document, from.id, false, true));
  const sky = applyCommand(document, command), built = buildPlayableLevel(sky, -1), cell = built.topology!.cells.find(cell => cell.sourceId === from.id)!;
  expect(built.document.zones[cell.id]!.lower.some(record => record.type === 2)).toBe(false);
  expect(built.document.zones[cell.id]!.lower.some(record => record.type === 12)).toBe(true);
  expect(built.document.zones[cell.id]!.backdrop).toBe(true);
  expect(decodeProject(encodeProject(sky)).roomSource).toEqual(sky.roomSource);
  expect(applyCommand(sky, command, true)).toEqual(document);
});
