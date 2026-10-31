import { describe, expect, it } from 'vitest';
import { openLevel, type LevelDocument } from '../src/model/level';
import { importedAnimationTables, validateAnimationReferences } from '../src/model/animationBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { originalFiles } from './helpers';

interface BadLink { name: string; change: (view: DataView, document: LevelDocument) => void; error: RegExp }
const badLinks: BadLink[] = [
  { name: 'out-of-file moving height', change: (v, d) => v.setInt32(d.mechanisms[0]!.offset + 8, d.files.graphics.length + 2), error: /Door 0.*graphics target/ },
  { name: 'moving height inside a polygon payload', change: (v, d) => v.setInt32(d.mechanisms[0]!.offset + 8, v.getInt32(d.mechanisms[0]!.offset + 8) + 2), error: /graphics target/ },
  { name: 'missing controlled room', change: (v, d) => v.setInt16(d.mechanisms[0]!.offset + 12, 32767), error: /Door 0.*missing room/ },
  { name: 'missing collision line', change: (v, d) => v.setInt16(d.mechanisms.find(item => item.walls.length)!.offset + 18, 32767), error: /missing collision line/ },
  { name: 'wall target pointing at a floor', change: (v, d) => v.setInt32(d.mechanisms.find(item => item.walls.length)!.offset + 20, d.zones[0]!.lower.find(record => record.type === 1)!.offset), error: /wall.*graphics target/ },
  { name: 'switch target at a wall start instead of its phase field', change: (v, d) => { const item = d.switches.find(item => item.zone >= 0)!; v.setInt32(item.offset + 6, item.graphic - 10); }, error: /Switch.*graphics target/ },
  { name: 'switch point pair past the last point', change: (v, d) => v.setInt16(d.switches.find(item => item.zone >= 0)!.offset + 4, d.points.length - 1), error: /activation point pair/ },
  { name: 'missing switch room', change: (v, d) => v.setInt16(d.switches.find(item => item.zone >= 0)!.offset, 32767), error: /Switch.*missing room/ },
];

describe('native animation import boundary', () => {
  it('resolves lift file aliases but requires direct indices for doors and switches', () => {
    const original = openLevel('original', originalFiles(1)), data = original.files.data.slice(), graphics = original.files.graphics.slice();
    const last = original.zones.at(-1)!;
    new DataView(data.buffer).setInt16(last.offset, 300);
    const lift = original.mechanisms.find(item => item.kind === 'lift')!, door = original.mechanisms.find(item => item.kind === 'door')!;
    const view = new DataView(graphics.buffer); view.setInt16(lift.offset + 12, 300);
    expect(() => openLevel('lift alias', { ...original.files, data, graphics })).not.toThrow();
    view.setInt16(door.offset + 12, 300);
    expect(() => openLevel('door alias', { ...original.files, data, graphics })).toThrow(/Door.*missing room/);
    view.setInt16(door.offset + 12, door.zone); view.setInt16(original.switches.find(item => item.zone >= 0)!.offset, 300);
    expect(() => openLevel('switch alias', { ...original.files, data, graphics })).toThrow(/Switch.*missing room/);
  });
  it.each(badLinks)('rejects $name before loading the engine and preserves original bytes', ({ change, error }) => {
    const document = openLevel('original', originalFiles(1)), graphics = document.files.graphics.slice(), before = document.files.graphics.slice();
    change(new DataView(graphics.buffer), document);
    expect(() => openLevel('bad animation', { ...document.files, graphics })).toThrow(error);
    expect(document.files.graphics).toEqual(before);
  });

  it('accepts unused stored switch fields without dereferencing them', () => {
    const document = openLevel('original', originalFiles(1)), graphics = document.files.graphics.slice(), inactive = document.switches.find(item => item.zone < 0)!;
    const view = new DataView(graphics.buffer); view.setInt32(inactive.offset + 6, 2147483647); view.setInt16(inactive.offset + 4, 32767);
    expect(() => openLevel('inactive stored fields', { ...document.files, graphics })).not.toThrow();
  });

  it('checks water record roles and group structure, including generated live water links', () => {
    const document = openLevel('original', originalFiles(1)), { tables } = importedAnimationTables(document);
    const water = document.zones.flatMap(zone => [...zone.lower, ...zone.upper].filter(record => record.type === 7).map(record => ({ zone: zone.id, graphic: record.offset })))[0]!;
    const graphics = buildGraphicsStreams(document, undefined, undefined, { animations: { ...tables,
      water: tables.water.map((group, i) => i ? group : { ...group, entries: [water] }) } });
    const rebuilt = openLevel('live water', { ...document.files, graphics });
    expect(() => validateAnimationReferences(rebuilt)).not.toThrow();
    const lastLift = rebuilt.mechanisms.filter(item => item.kind === 'lift').at(-1)!;
    const group = lastLift.offset + 18 + lastLift.walls.length * 10 + 4;
    const wrong = graphics.slice(), view = new DataView(wrong.buffer);
    view.setInt32(group + 16, rebuilt.zones[0]!.lower.find(record => record.type === 1)!.offset);
    expect(() => openLevel('wrong water target', { ...rebuilt.files, graphics: wrong })).toThrow(/Water animation 0.*graphics target/);
    view.setInt16(group + 14, 32767);
    expect(() => openLevel('missing water room', { ...rebuilt.files, graphics: wrong })).toThrow(/invalid room list/);
  });
});
