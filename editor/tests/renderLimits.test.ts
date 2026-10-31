import { expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { zoneGraphicsRecordLimit } from '../src/model/renderLimits';
import { graphicsBudgetFixture } from './graphicsBudgetFixture';
import { buildPlayableLevel, preparePlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeGameLevel, encodeProject } from '../src/model/project';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { buildGeneratedGraphics } from '../src/model/generatedGraphics';
import { importedGraphicsStreams } from '../src/model/graphicsBuild';
import { importedAnimationTables } from '../src/model/animationBuild';
import { BinaryReader } from '../src/model/binary';

it('uses the native header for all seven bounded renderer traversals', () => {
  const source = readFileSync(new URL('../../src/renderer.c', import.meta.url), 'utf8');
  expect(source.match(/(?:max_iter|scan_iter) = AB3D_ZONE_GRAPHICS_RECORD_LIMIT/g)).toHaveLength(7);
  expect(source).not.toMatch(/(?:max_iter|scan_iter) = 500/);
});

it.each([false, true])('accepts the exact native budget and preserves oversize editable source (upper=%s)', async upper => {
  const allowed = graphicsBudgetFixture(zoneGraphicsRecordLimit, upper), oversized = graphicsBudgetFixture(zoneGraphicsRecordLimit + 1, upper);
  expect(buildPlayableLevel(allowed, -1).document).toBe(allowed);
  expect((await preparePlayableLevel(allowed)).document).toBe(allowed);
  expect(encodeGameLevel(allowed).length).toBeGreaterThan(0);
  const saved = decodeProject(encodeProject(oversized));
  expect(saved.files).toEqual(oversized.files);
  expect(upper ? saved.zones[0]!.upper : saved.zones[0]!.lower).toHaveLength(zoneGraphicsRecordLimit + 1);
  const message = new RegExp(`Room 0 ${upper ? 'upper' : 'lower'} floor has ${zoneGraphicsRecordLimit + 1} graphics commands`);
  expect(() => buildPlayableLevel(saved, -1)).toThrow(message);
  expect(() => encodeGameLevel(saved)).toThrow(message);
  await expect(preparePlayableLevel(saved)).rejects.toThrow(message);
  await expect(preparePlayableLevel(saved)).rejects.toThrow(message);
  expect(() => buildSourceLevel(saved, { resolvedExit: -1 })).toThrow(message);
  const reader = new BinaryReader(saved.files.graphics, 'budget'), streams = importedGraphicsStreams(saved);
  expect(() => buildGeneratedGraphics(saved, saved.zones.map(zone => ({ id: zone.id,
    dataOffset: reader.i32(16 + zone.id * 4), ...streams.get(zone.id)! })), importedAnimationTables(saved).tables)).toThrow(message);
});
