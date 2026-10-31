import { checkedInteger } from './binary';
import { pointInRoomOutline } from './geometry';
import type { LevelDocument, Start } from './level';

/** Authored starts reference editable rooms, never generated convex cell IDs.
 * The binary start and multiplayer avatar remain the import baseline until build. */
export interface SourceStart { readonly player: 1 | 2; readonly room: string; readonly x: number; readonly z: number }

/** A source-only room has no binary zone yet. Expose -1 rather than showing
 * the template's unrelated zone as this start's location. */
export function editableStarts(document: LevelDocument): (Start & { room?: string })[] {
  return document.starts.map(start => {
    const authored = document.sourceStarts?.find(item => item.player === start.player);
    return authored ? { ...authored, zone: document.roomSource.rooms.find(room => room.id === authored.room)?.originZone ?? -1 } : start;
  });
}

export function restoreSourceStarts(value: unknown): readonly SourceStart[] {
  if (!Array.isArray(value) || value.length > 2) throw new Error('The source player starts must contain at most two players.');
  const players = new Set<number>();
  return Object.freeze(value.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('player' in item) || (item.player !== 1 && item.player !== 2)
      || players.has(item.player) || !('room' in item) || typeof item.room !== 'string' || !item.room || item.room.length > 128
      || !('x' in item) || typeof item.x !== 'number' || !('z' in item) || typeof item.z !== 'number'
      || Object.keys(item).some(key => !['player', 'room', 'x', 'z'].includes(key))) throw new Error('Invalid or duplicate source player start.');
    players.add(item.player);
    // Header start WORDs and 16.16 avatar coordinates share this whole-unit range.
    return Object.freeze({ player: item.player, room: item.room,
      x: checkedInteger(item.x, -32768, 32767, 'Player start X'), z: checkedInteger(item.z, -32768, 32767, 'Player start Z') });
  }));
}

export function placeSourceStart(document: LevelDocument, player: 1 | 2, room: string, x: number, z: number): readonly SourceStart[] {
  const starts = restoreSourceStarts([...(document.sourceStarts ?? []).filter(start => start.player !== player), { player, room, x, z }]);
  const target = document.roomSource.rooms.find(item => item.id === room);
  if (!target || !pointInRoomOutline(target.vertices, { x, z })) throw new Error('Place the player start inside its selected editable room.');
  if (document.sourceStarts?.some(start => start.player === player && start.room === room && start.x === x && start.z === z)) return document.sourceStarts;
  return starts;
}
