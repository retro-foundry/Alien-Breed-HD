import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { movingRoomConflict } from '../src/model/movingRoomEligibility';
import { compileRoomTopology } from '../src/model/roomTopology';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { movingRoomFromDraft, setSourceMovingRoom } from '../src/model/sourceMovingRoom';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';

export function importedMovingFixture(level: number, kind: 'door' | 'lift') {
  const document = openLevel(`Imported movement ${level}`, originalFiles(level), level - 1, -1);
  const topology = compileRoomTopology(document, document.roomSource), errors: string[] = [];
  for (const room of document.roomSource.rooms) {
    if (movingRoomConflict(document, room)) continue;
    const cells = topology.cells.filter(cell => cell.sourceId === room.id);
    if (cells.length !== 1) continue;
    const cell = cells[0]!, settings = document.zones[room.recordZone]!;
    if (settings.lower.filter(record => record.type === (kind === 'door' ? 2 : 1)).length !== 1 || [...settings.lower, ...settings.upper].some(record => record.type === 5)) continue;
    const neighbours = cell.neighbours.filter(id => id >= 0).map(id => document.zones[topology.cells.find(cell => cell.id === id)!.recordZone]!);
    if (!neighbours.length) continue;
    const top = Math.max(settings.ceiling, ...neighbours.map(room => room.ceiling)), bottom = kind === 'door' ? Math.min(settings.floor, ...neighbours.map(room => room.floor)) : settings.floor;
    if (top >= bottom || bottom * 4 === 999) continue;
    const wall = roomAppearanceChoices(document, room).lower[0]; if (!wall) continue;
    try {
      const value = movingRoomFromDraft({ kind, top, bottom, current: bottom, wall: wall.offset, conditions: 0, openRule: kind === 'door' ? 4 : 2, closeRule: kind === 'door' ? 0 : 2 });
      const changed = applyCommand(document, editRoomSource(setSourceMovingRoom(document, room.id, value))), built = buildPlayableLevel(changed, -1);
      return { document, changed, built, room, value, mechanism: built.document.mechanisms.filter(item => item.kind === kind).at(-1)! };
    } catch (error) { errors.push(`${room.id}: ${String(error)}`); }
  }
  throw new Error(`No convertible ${kind} in Level ${level}: ${errors.slice(0, 5).join('; ')}`);
}
