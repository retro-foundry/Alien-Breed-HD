import { expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { buildSourceLevel } from '../src/model/sourceBuild';
import { originalFiles } from './helpers';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('rebuilds complete source level %i with fresh runtime tables', level => {
  const original = openLevel('original', originalFiles(level), level - 1), before = structuredClone(original.files), sourceBefore = JSON.stringify(original.roomSource);
  const built = buildSourceLevel(original, { resolvedExit: -1 });
  expect(built.document.zones.length).toBe(built.topology.cells.length);
  expect(built.document.playtestSlot).toBe(level - 1); expect(built.document.exitZone).toBe(-1);
  expect(built.document.mechanisms).toHaveLength(original.mechanisms.length);
  expect(built.document.switches.filter(slot => slot.zone >= 0)).toHaveLength(original.switches.filter(slot => slot.zone >= 0).length);
  for (const start of original.starts) {
    const generated = built.document.starts.find(item => item.player === start.player)!;
    expect({ ...generated, zone: start.zone }).toEqual(start);
    expect(built.topology.cells[generated.zone]!.originZone).toBe(start.zone);
  }
  expect(original.files).toEqual(before); expect(JSON.stringify(original.roomSource)).toBe(sourceBefore);
  expect(built.document.drawGroups ?? built.topology.cells.map(cell => cell.id))
    .toEqual(built.topology.cells.map(cell => built.topology.sourceCells.get(cell.sourceId)![0]!));
  const second = buildSourceLevel(original, { resolvedExit: -1 }); expect(second.document.files).toEqual(built.document.files);
});
