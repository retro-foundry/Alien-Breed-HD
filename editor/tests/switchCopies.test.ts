import { expect, it } from 'vitest';
import { switchCopyFixture } from './switchCopyFixture';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { roomAppearanceBuildOptions } from '../src/model/roomAppearance';
import { remapSwitchConditions } from '../src/model/switchCopies';
import { decodeProject, encodeProject } from '../src/model/project';

it('allocates a copied switch in an unused native slot with independent points and graphics', () => {
  const { document, switchCopy, source, dx, dz } = switchCopyFixture();
  const result = buildSourceLevel(document, { resolvedExit: -1, ...roomAppearanceBuildOptions(document), switchCopies: [switchCopy] });
  const original = result.document.switches[source.id]!, copied = result.document.switches[switchCopy.slot]!;
  expect(result.document.switches).toHaveLength(8);
  expect(copied.zone).toBe(result.topology.cells.find(cell => cell.sourceId === switchCopy.room)!.id);
  expect(copied.zone).not.toBe(original.zone); expect(copied.graphic).not.toBe(original.graphic); expect(copied.point).not.toBe(original.point);
  expect([copied.point, copied.point + 1].map(id => result.document.points[id])).toEqual([source.point, source.point + 1]
    .map(id => expect.objectContaining({ x: document.points[id]!.x + dx, z: document.points[id]!.z + dz })));
  expect(copied).toMatchObject({ type: original.type, countdown: original.countdown });
  expect(decodeProject(encodeProject(result.document)).files).toEqual(result.document.files);
});

it('rejects reused slots, shared targets and missing room correspondence before changing source', () => {
  const { document, switchCopy, source } = switchCopyFixture(), files = document.files;
  const build = (copies: readonly typeof switchCopy[]) => buildSourceLevel(document, { resolvedExit: -1, ...roomAppearanceBuildOptions(document), switchCopies: copies });
  expect(() => build([{ ...switchCopy, slot: source.id }])).toThrow(/unused native slot/);
  expect(() => build([switchCopy, switchCopy])).toThrow(/independent graphics|unused native slot/);
  expect(() => build([{ ...switchCopy, graphicsRoom: 'missing' }])).toThrow(/explicit fresh rooms/);
  expect(() => build([{ ...switchCopy, room: document.roomSource.rooms[0]!.id }])).toThrow(/explicit fresh rooms/);
  expect(() => build([{ ...switchCopy, bytes: new Uint8Array(13) }])).toThrow(/complete 14-byte/);
  expect(document.files).toBe(files);
});

it('remaps internal switch requirements while preserving keys/external bits and rejecting aliases', () => {
  const keys = 3, internal = 1 << 4, external = 1 << 7;
  expect(remapSwitchConditions(keys | internal | external, new Map([[0, 6]]))).toBe(keys | (1 << 10) | external);
  expect(remapSwitchConditions(keys | internal, new Map([[0, 1], [1, 0]]))).toBe(keys | (1 << 5));
  expect(() => remapSwitchConditions(internal | external, new Map([[0, 3]]))).toThrow(/alias an external/);
  expect(() => remapSwitchConditions(internal, new Map([[0, 6], [1, 6]]))).toThrow(/distinct condition bits/);
});
