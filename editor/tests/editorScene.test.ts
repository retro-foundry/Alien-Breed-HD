import { expect, it } from 'vitest';
import { originalFiles } from './helpers';
import { openLevel } from '../src/model/level';
import { editorScene, pickSceneFace, spatialEdit } from '../src/model/editorScene';
import { applyCommand, execute, undo, redo } from '../src/model/commands';
import { roomHeightFixture } from './roomHeightFixture';
import { originalObjectPalette, placePaletteObject } from '../src/model/objectPalette';
import { roomSettings } from '../src/model/roomHeights';
import { decodeProject, encodeProject } from '../src/model/project';
it('all original levels expose editable room planes and actors on their actual storey', () => {
  for (let i = 1; i <= 16; i++) {
    const document = openLevel('Original', originalFiles(i));
    for (const upper of [false, true]) {
      const scene = editorScene(document, upper);
      for (const face of scene.faces.filter(f => f.kind !== 'wall')) {
        const room = document.roomSource.rooms.find(r => r.id === face.room)!, settings = roomSettings(document, room);
        expect(face.points.every(p => p.y === settings[face.kind === 'floor' ? upper ? 'upperFloor' : 'floor' : upper ? 'upperCeiling' : 'ceiling'])).toBe(true);
      }
      for (const marker of scene.markers.filter(m => m.kind === 'object')) expect(document.objects[marker.id]!.upper).toBe(upper);
    }
  }
});
it('ray picking chooses the nearest actual room surface', () => {
  const { document, from } = roomHeightFixture(), scene = editorScene(document, false), floor = roomSettings(document, from).floor;
  const hit = pickSceneFace(scene.faces.filter(f => f.kind === 'floor'), { origin: { x: 20128, y: floor - 100, z: 20128 }, direction: { x: 0, y: 1, z: 0 } });
  expect(hit?.face.room).toBe(from.id); expect(hit?.position.y).toBe(floor);
});
it('a spatial object gesture saves and undoes horizontal and vertical edits together; rejected movement is atomic', () => {
  const fixture = roomHeightFixture(), original = originalFiles(1), choice = originalObjectPalette([original.data], [original.graphics]).find(c => c.name === 'Barrel')!;
  const proposal = placePaletteObject(fixture.document, choice, fixture.from.id, 20128, 20192, false), document = applyCommand(fixture.document, proposal.command);
  const marker = editorScene(document, false).markers.find(m => m.kind === 'object' && m.id === proposal.id)!;
  const verticalOnly = applyCommand(document, spatialEdit(document, { kind: 'marker', marker, position: { ...marker.position, y: marker.position.y - .5 } }));
  expect(editorScene(verticalOnly, false).markers.find(m => m.kind === 'object' && m.id === proposal.id)?.position.y).toBe(marker.position.y - .5);
  const command = spatialEdit(document, { kind: 'marker', marker, position: { x: 20144, y: marker.position.y - 7.5, z: 20184 } });
  let history = execute({ document, past: [], future: [] }, command);
  const moved = editorScene(history.document, false).markers.find(m => m.kind === 'object' && m.id === proposal.id)!;
  expect(moved.position).toEqual({ x: 20144, y: marker.position.y - 7.5, z: 20184 });
  const saved = decodeProject(encodeProject(history.document)); expect(editorScene(saved, false).markers.find(m => m.kind === 'object' && m.id === proposal.id)!.position).toEqual(moved.position);
  history = undo(history); expect(history.document.files.data).toEqual(document.files.data); expect(history.document.sourceObjects).toBe(document.sourceObjects);
  history = redo(history); expect(history.past).toHaveLength(1);
  expect(() => spatialEdit(document, { kind: 'marker', marker, position: { x: -30000, y: 0, z: -30000 } })).toThrow('inside a room');
});
it('height handles share existing quarter-unit validation and keep water independent', () => {
  const { document, from } = roomHeightFixture(), before = roomSettings(document, from);
  const command = spatialEdit(document, { kind: 'height', room: from.id, field: 'floor', value: before.floor - .25 });
  const changed = applyCommand(document, command), settings = roomSettings(changed, changed.roomSource.rooms.find(r => r.id === from.id)!);
  expect(settings.floor).toBe(before.floor - .25); expect(settings.water).toBe(before.water);
  expect(() => spatialEdit(document, { kind: 'height', room: from.id, field: 'floor', value: before.ceiling })).toThrow('ceiling');
});

it('spatial corner edits keep only horizontal source fields and reopen cleanly', () => {
  const { document, from } = roomHeightFixture(), corner = from.vertices[0]!;
  const command = spatialEdit(document, { kind: 'corner', room: from.id, index: 0, position: { x: corner.x + 8, z: corner.z, y: -123 } });
  const changed = applyCommand(document, command), saved = decodeProject(encodeProject(changed));
  const vertices = saved.roomSource.rooms.find(r => r.id === from.id)!.vertices;
  expect(vertices).toContainEqual({ x: corner.x + 8, z: corner.z });
  expect(vertices.every(p => Object.keys(p).sort().join(',') === 'x,z')).toBe(true);
});
