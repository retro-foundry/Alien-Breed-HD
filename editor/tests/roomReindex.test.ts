import { describe, expect, it } from 'vitest';
import { openLevel } from '../src/model/level';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { importedAnimationTables, remapAnimationRooms } from '../src/model/animationBuild';
import { originalFiles } from './helpers';
import { reindexRooms } from './reindexedRooms';

describe('complete room identity remapping', () => {
  it.each(Array.from({ length: 16 }, (_, i) => i + 1))('rewrites level %i room relationships without changing coordinates or material records', level => {
    const original = openLevel(`Level ${level}`, originalFiles(level), level - 1);
    const ids = new Map(original.zones.map(zone => [zone.id, original.zones.length - 1 - zone.id]));
    expect(buildGraphicsStreams(original, undefined, undefined, { roomIds: new Map(original.zones.map(zone => [zone.id, zone.id])) })).toEqual(original.files.graphics);
    const automaticallyMapped = buildGraphicsStreams(original, undefined, undefined, { roomIds: ids });
    expect(buildGraphicsStreams(original, undefined, undefined, { roomIds: ids,
      animations: remapAnimationRooms(original, importedAnimationTables(original).tables, ids) })).toEqual(automaticallyMapped);
    const rebuilt = reindexRooms(original, ids, original.zones.at(-1)!.id);
    expect(rebuilt.exitZone).toBe(0); expect(rebuilt.points).toEqual(original.points);
    expect(rebuilt.starts).toEqual(original.starts.map(start => ({ ...start, zone: ids.get(start.zone) })));
    for (const zone of original.zones) {
      const changed = rebuilt.zones[ids.get(zone.id)!]!;
      expect(changed.fileId).toBe(changed.id); expect(changed.boundaryIds).toEqual(zone.boundaryIds);
      expect(changed.teleport).toEqual({ ...zone.teleport, zone: zone.teleport.zone < 0 ? zone.teleport.zone : ids.get(zone.teleport.zone) });
      expect(changed.visibleZones).toEqual(zone.visibleZones.map(id => ids.get(id)));
      for (const [before, after] of [[zone.lower, changed.lower], [zone.upper, changed.upper]] as const)
        expect(after.map(({ offset: _, ...record }) => record)).toEqual(before.map(({ offset: _, ...record }) => record));
    }
    original.objects.forEach((object, i) => expect(rebuilt.objects[i]).toEqual({ ...object, zone: object.zone < 0 ? object.zone : ids.get(object.zone) }));
    original.lines.forEach((line, i) => expect(rebuilt.lines[i]).toEqual({ ...line, connect: line.connect < 0 ? line.connect : ids.get(line.connect) }));
    original.mechanisms.forEach((item, i) => expect(rebuilt.mechanisms[i]!.zone).toBe(ids.get(item.zone)));
    original.switches.forEach((item, i) => expect(rebuilt.switches[i]!.zone).toBe(item.zone < 0 ? item.zone : ids.get(item.zone)));
    expect(reindexRooms(rebuilt, ids, rebuilt.exitZone!).starts).toEqual(original.starts);
  });

  it('rejects incomplete, duplicate and out-of-range target identities before altering source bytes', () => {
    const original = openLevel('original', originalFiles(1)), before = original.files.graphics.slice();
    expect(() => buildGraphicsStreams(original, undefined, undefined, { roomIds: new Map([[0, 0]]) })).toThrow(/each source room/);
    expect(() => buildGraphicsStreams(original, undefined, undefined, { roomIds: new Map(original.zones.map(zone => [zone.id, 0])) })).toThrow(/unique target/);
    expect(() => buildGraphicsStreams(original, undefined, undefined, { roomIds: new Map(original.zones.map(zone => [zone.id, zone.id + 1])) })).toThrow(/Generated graphics room/);
    expect(original.files.graphics).toEqual(before);
  });
});
