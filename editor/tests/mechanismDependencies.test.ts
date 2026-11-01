import { expect, it } from 'vitest';
import { mechanismDependencies, mechanismsForRoom } from '../src/model/mechanismDependencies';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { importedAnimationTables } from '../src/model/animationBuild';

it.each(Array.from({ length: 16 }, (_, index) => index + 1))('resolves every native mechanism write and trigger incidence in level %i', level => {
  const document = openLevel('Dependencies', originalFiles(level)), tables = importedAnimationTables(document).tables;
  for (const mechanism of document.mechanisms) {
    const dependencies = mechanismDependencies(document, mechanism.kind, mechanism.id);
    const record = tables[mechanism.kind === 'door' ? 'doors' : 'lifts']![mechanism.id]!;
    expect(dependencies.surface.graphic).toBe(record.graphic);
    expect(dependencies.walls).toHaveLength(record.walls.length);
    const writes = [dependencies.surface, ...dependencies.walls];
    for (const write of writes) expect(document.zones.find(zone => zone.id === write.room)!.lower
      .concat(document.zones.find(zone => zone.id === write.room)!.upper).some(graphic => graphic.offset === write.graphic)).toBe(true);
    dependencies.walls.forEach((wall, index) => {
      expect(wall.line).toBe(record.walls[index]!.line);
      expect(wall.collisionRooms.length).toBeGreaterThan(0);
      for (const room of wall.collisionRooms) expect(document.zones.find(zone => zone.id === room)!.boundaryIds).toContain(wall.line);
    });
    expect(dependencies.graphicsRooms).toEqual([...new Set([dependencies.room, ...writes.map(write => write.room)])].sort((a, b) => a - b));
    const mask = dependencies.keys.reduce((mask, number) => mask | (1 << (number - 1)), 0)
      | dependencies.switches.reduce((mask, item) => mask | (1 << (item.number + 3)), 0);
    expect(mask).toBe(mechanism.conditions & 4095);
  }
});

it('selects a lift by its native slot before file aliases and resolves an out-of-range alias', () => {
  const original = openLevel('Aliases', originalFiles(1)), lift = original.mechanisms.find(item => item.kind === 'lift')!;
  const other = original.zones.find(zone => zone.id !== lift.zone)!, data = original.files.data.slice();
  new DataView(data.buffer).setInt16(other.offset, lift.zone);
  const conflict = openLevel(original.name, { ...original.files, data });
  expect(mechanismsForRoom(conflict, lift.zone).some(item => item.kind === 'lift' && item.id === lift.id)).toBe(true);
  expect(mechanismsForRoom(conflict, other.id).some(item => item.kind === 'lift' && item.id === lift.id)).toBe(false);
  new DataView(data.buffer).setInt16(original.zones[lift.zone]!.offset, 300);
  const graphics = original.files.graphics.slice(); new DataView(graphics.buffer).setInt16(lift.offset + 12, 300);
  const alias = openLevel(original.name, { ...original.files, data, graphics });
  expect(mechanismsForRoom(alias, lift.zone).some(item => item.kind === 'lift' && item.id === lift.id)).toBe(true);
  expect(mechanismDependencies(alias, 'lift', lift.id).room).toBe(lift.zone);
});
