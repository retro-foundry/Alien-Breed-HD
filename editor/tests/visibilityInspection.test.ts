import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { visibilityInspection } from '../src/model/visibilityInspection';
import { roomHeightFixture } from './roomHeightFixture';

it('inspects the actual stored visibility and portal records across all original levels', () => {
  for (let level = 1; level <= 16; level++) {
    const document = openLevel(`Level ${level}`, originalFiles(level)), artifact = buildPlayableLevel(document, -1);
    const clipData = new DataView(document.files.clips.buffer, document.files.clips.byteOffset, document.files.clips.byteLength);
    let offset = 0;
    for (const zone of document.zones) {
      const result = visibilityInspection(artifact, undefined, zone.id);
      expect(result.rows.map(({ bounds: _bounds, ...row }) => row)).toEqual(zone.visibility);
      for (const row of result.rows) {
        if (row.clip < 0) { expect(row.bounds).toBeUndefined(); continue; }
        const start = offset, left: number[] = [], right: number[] = [];
        let rightSide = false;
        for (;;) {
          const point = clipData.getInt16(offset); offset += 2;
          if (point === -2) break;
          if (point === -1) rightSide = true; else (rightSide ? right : left).push(point);
        }
        expect(row.bounds).toEqual({ wordOffset: start / 2, left, right });
      }
      expect(result.portals.map(line => line.id)).toEqual(zone.boundaryIds.filter(id => document.lines[id]!.connect >= 0));
      expect(result.polygons.map(polygon => polygon.id)).toEqual(document.zones.filter(candidate => candidate.id === zone.id || zone.visibility.some(row => row.zone === candidate.id)).map(zone => zone.id));
    }
    expect(offset).toBe(document.connectTableOffset);
  }
});

it('rejects malformed clip boundaries rather than drawing misleading points', () => {
  const document = openLevel('Bad clipping', originalFiles(1)), clips = document.files.clips.slice();
  new DataView(clips.buffer).setInt16(0, 32767);
  const source = { ...document, files: { ...document.files, clips } };
  expect(() => visibilityInspection({ source, document: source }, undefined, 0)).toThrow(/invalid clipping boundary/);
});

it('maps authored rooms to current compiled cells and excludes disconnected rooms', () => {
  const fixture = roomHeightFixture(), artifact = buildPlayableLevel(fixture.document, -1);
  const result = visibilityInspection(artifact, fixture.from.id, undefined, -999);
  expect(result.cells).toEqual(artifact.topology!.sourceCells.get(fixture.from.id));
  expect(result.selected).toBe(result.cells[0]);
  expect(result.polygons.some(polygon => artifact.topology!.sourceCells.get(fixture.to.id)!.includes(polygon.id))).toBe(false);
  expect(result.portals).toHaveLength(0);
  expect(result.polygons.find(polygon => polygon.origin)!.vertices).toEqual(artifact.document.zones[result.selected!]!.boundaryIds.map(id => ({ x: artifact.document.lines[id]!.x, z: artifact.document.lines[id]!.z })));
});

it('does not substitute a zone selection for a missing source room cell', () => {
  const fixture = roomHeightFixture(), artifact = buildPlayableLevel(fixture.document, -1);
  expect(visibilityInspection(artifact, undefined, undefined).polygons).toHaveLength(0);
  expect(visibilityInspection(artifact, fixture.from.id, 0).selected).not.toBe(0);
  expect(visibilityInspection(artifact, 'missing-room', 0).polygons).toHaveLength(0);
});

it('inspects each generated convex cell of a concave source room independently', () => {
  const fixture = roomHeightFixture();
  const document = { ...fixture.document, roomSource: { ...fixture.document.roomSource, rooms: fixture.document.roomSource.rooms.map(room => room.id === fixture.from.id ? {
    ...room, wallSources: [null, null, null, null, null, null], vertices: [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20128 }, { x: 20128, z: 20128 }, { x: 20128, z: 20256 }, { x: 20000, z: 20256 }],
  } : room) } };
  const artifact = buildPlayableLevel(document, -1), cells = artifact.topology!.sourceCells.get(fixture.from.id)!;
  expect(cells.length).toBeGreaterThan(1);
  for (const id of cells) {
    const result = visibilityInspection(artifact, fixture.from.id, undefined, id);
    expect(result.selected).toBe(id);
    expect(result.rows.map(({ bounds: _bounds, ...row }) => row)).toEqual(artifact.document.zones.find(zone => zone.id === id)!.visibility);
  }
});
