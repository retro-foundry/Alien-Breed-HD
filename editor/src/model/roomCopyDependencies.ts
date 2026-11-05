import { activeNativeMechanisms } from './nativeMechanisms';
import { activeNativeSwitches } from './nativeSwitches';
import type { LevelDocument } from './level';
import { mechanismDependencies } from './mechanismDependencies';
import { compileRoomTopology } from './roomTopology';
import { roomAppearanceChoices } from './roomAppearance';
import { hasNativeWallMaterials } from './roomMaterials';

/** Capture the connected authoring group, not just the controlled volume:
 * native moving graphics and switch walls can belong to adjacent rooms. */
export function roomCopyDependencies(document: LevelDocument, ids: readonly string[]): readonly string[] {
  const selected = new Set(ids), topology = compileRoomTopology(document, document.roomSource);
  const sourceRooms = new Map(document.roomSource.rooms.map(room => [room.id, room]));
  const original = (zone: number) => document.roomSource.rooms.find(room => room.originZone === zone)?.id;
  type Binding = { control: string | undefined; rooms: (string | undefined)[]; portals: (string | undefined)[]; label: string };
  const bindings: Binding[] = activeNativeMechanisms(document).map(item => {
    const dependencies = mechanismDependencies(document, item.kind, item.id);
    return { control: original(dependencies.room), rooms: [...new Set([...dependencies.graphicsRooms, ...dependencies.walls.flatMap(wall => wall.collisionRooms)])].map(original),
      portals: [original(dependencies.room)], label: `${item.kind} ${item.id}` };
  });
  for (const item of activeNativeSwitches(document)) {
    const owner = document.zones.find(zone => [...zone.lower, ...zone.upper].some(record => record.offset === item.graphic - 10));
    const rooms = [original(item.zone), owner ? original(owner.id) : undefined];
    bindings.push({ control: original(item.zone), rooms, portals: rooms, label: `Switch ${item.id + 1}` });
  }
  for (const room of document.roomSource.rooms) if (room.moving) bindings.push({ control: room.id, rooms: [room.id], portals: [room.id], label: `Authored ${room.moving.kind}` });
  for (const item of document.roomSource.controllerCopies ?? []) bindings.push({ control: item.room, rooms: [item.room, ...item.graphicsRooms.map(pair => pair[1]), ...item.collisionRooms.map(pair => pair[1])], portals: [item.room], label: `Copied ${item.kind}` });
  for (const item of document.roomSource.switchCopies ?? []) bindings.push({ control: item.room, rooms: [item.room, item.graphicsRoom], portals: [item.room, item.graphicsRoom], label: `Copied switch ${item.slot + 1}` });
  const add = (id: string | undefined, label: string) => {
    if (!id || !sourceRooms.has(id)) throw new Error(`${label} has a missing attached room. Repair its attachment before copying.`);
    selected.add(id);
  };
  const neighbours = (id: string) => topology.cells.filter(cell => cell.sourceId === id).flatMap(cell => cell.neighbours
    .filter(target => target >= 0).map(target => topology.cells[target]!.sourceId)).filter(target => target !== id);
  let changed = true;
  while (changed) {
    const count = selected.size;
    for (const binding of bindings) if (binding.control && selected.has(binding.control)) {
      binding.rooms.forEach(id => add(id, binding.label));
      binding.portals.forEach(id => { if (id) neighbours(id).forEach(target => add(target, binding.label)); });
    }
    for (const id of selected) {
      const room = sourceRooms.get(id);
      if (!room) throw new Error('Select existing rooms before copying.');
      if (!hasNativeWallMaterials(room.originZone, room.wallSources)) continue;
      const choices = roomAppearanceChoices(document, room);
      if (!choices.lower.length || choices.template.upper.length > 0 && !choices.upper.length)
        neighbours(id).forEach(target => add(target, 'Native room materials'));
    }
    changed = selected.size !== count;
  }
  return Object.freeze([...selected]);
}
