import { normaliseRoomOutline, type Position } from './geometry';
import type { LevelDocument } from './level';
import type { RoomSource } from './roomSource';
import { compileRoomTopology, type CompiledRoomCell, type RoomTopology } from './roomTopology';
import { BinaryReader } from './binary';
import { restoreSourceStarts } from './sourceStarts';

export interface ReferenceDiagnostic { severity: 'error' | 'warning'; subject: string; message: string }
export interface RoomReferencePlan {
  starts: { player: 1 | 2; cell: number | null }[];
  objects: Map<number, number | null>;
  teleports: Map<number, number | null>;
  cellTeleports: Map<number, number | null>;
  switches: Map<number, number | null>;
  mechanisms: { kind: 'door' | 'lift'; id: number; cells: number[] }[];
  diagnostics: ReferenceDiagnostic[];
}

/** Primary convex boundary test; collision's secondary corner fences must
 * not reject a point within its authored room (the original level 3 start).
 * Boundaries are included, with the lowest compiler cell ID resolving an
 * internal decomposition seam deterministically. No coordinate rounding. */
export function cellContainsPoint(topology: RoomTopology, cell: CompiledRoomCell, point: Position): boolean {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.z)) return false;
  return cell.vertices.every((id, index) => {
    const a = topology.points[id]!, b = topology.points[cell.vertices[(index + 1) % cell.vertices.length]!]!;
    return (b.x - a.x) * (point.z - a.z) - (b.z - a.z) * (point.x - a.x) <= 0;
  });
}

/** Source-to-runtime relationships for objects.c's OBJ_ZONE, the two header
 * starts, zone teleport fields, Anims.s mechanisms and SwitchRoutine. Keep
 * unresolved references null and diagnostic-bearing, never silently disabled.
 * GraphicRoom at object +26 is not the authored zone at +12: retain that
 * opaque/runtime field for its own compiler stage rather than guessing.
 */
export function planRoomReferences(document: LevelDocument, source: RoomSource, topology = compileRoomTopology(document, source)): RoomReferencePlan {
  const diagnostics: ReferenceDiagnostic[] = [];
  const byOrigin = new Map<number, CompiledRoomCell[]>();
  for (const cell of topology.cells) if (cell.originZone !== null)
    byOrigin.set(cell.originZone, [...(byOrigin.get(cell.originZone) ?? []), cell]);
  const unchanged = new Set<number>();
  for (const room of source.rooms) if (room.originZone !== null) {
    const zone = document.zones.find(zone => zone.id === room.originZone)!;
    const original = normaliseRoomOutline(zone.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })));
    if (JSON.stringify(original) === JSON.stringify(room.vertices)) unchanged.add(room.originZone);
  }
  const error = (subject: string, message: string) => diagnostics.push({ severity: 'error', subject, message });
  const locate = (origin: number, point: Position, subject: string, preserveLegacyOutside = false): number | null => {
    const cells = byOrigin.get(origin);
    if (!cells?.length) { error(subject, `Its destination room ${origin} was deleted. Choose another room or remove this dependency.`); return null; }
    const inside = cells.filter(cell => cellContainsPoint(topology, cell, point)).toSorted((a, b) => a.id - b.id);
    if (inside.length) return inside[0]!.id;
    if (preserveLegacyOutside && unchanged.has(origin) && cells.length === 1) {
      diagnostics.push({ severity: 'warning', subject, message: `The original object lies outside room ${origin}; its unchanged room relationship has been retained.` });
      return cells[0]!.id;
    }
    error(subject, `Its position (${point.x}, ${point.z}) is outside the remaining geometry of room ${origin}. Move it inside before building.`);
    return null;
  };
  const authoredStarts = restoreSourceStarts(document.sourceStarts ?? []);
  const starts = document.starts.map(start => {
    const authored = authoredStarts.find(item => item.player === start.player);
    if (!authored) return { player: start.player, cell: locate(start.zone, start, `Player ${start.player} start`) };
    const cells = topology.cells.filter(cell => cell.sourceId === authored.room);
    const cell = cells.filter(cell => cellContainsPoint(topology, cell, authored)).toSorted((a, b) => a.id - b.id)[0];
    if (!cell) error(`Player ${start.player} start`, cells.length
      ? `Its position (${authored.x}, ${authored.z}) is outside editable room ${authored.room}. Move it inside before building.`
      : `Its editable room ${authored.room} was deleted. Choose another room before building.`);
    return { player: start.player, cell: cell?.id ?? null };
  });
  const reader = new BinaryReader(document.files.data, 'source avatar references');
  const avatarStarts = new Map(authoredStarts.map(start => [reader.i32(start.player === 1 ? 46 : 50), starts.find(item => item.player === start.player)!.cell]));
  const objects = new Map(document.objects.map(object => [object.id, avatarStarts.has(object.offset) ? avatarStarts.get(object.offset)!
    : object.zone < 0 ? object.zone : locate(object.zone, object, `Object ${object.id}`, true)]));
  const teleports = new Map<number, number | null>();
  for (const room of source.rooms) if (room.originZone !== null) {
    const template = document.zones.find(zone => zone.id === room.recordZone)!;
    teleports.set(room.originZone, template.teleport.zone < 0 ? template.teleport.zone
      : locate(template.teleport.zone, template.teleport, `Room ${room.originZone} teleport`));
  }
  const cellTeleports = new Map<number, number | null>();
  for (const cell of topology.cells) {
    if (cell.originZone !== null) {
      cellTeleports.set(cell.id, teleports.get(cell.originZone)!); continue;
    }
    const template = document.zones.find(zone => zone.id === cell.recordZone)!;
    cellTeleports.set(cell.id, template.teleport.zone < 0 ? template.teleport.zone
      : locate(template.teleport.zone, template.teleport, `Room ${cell.sourceId} template teleport`));
  }
  const switches = new Map<number, number | null>();
  for (const item of document.switches) {
    if (item.zone < 0) { switches.set(item.id, item.zone); continue; }
    const a = document.points[item.point], b = document.points[item.point + 1];
    if (!a || !b) { error(`Switch ${item.id}`, 'Its activation point pair is missing. Repair the switch before building.'); switches.set(item.id, null); continue; }
    switches.set(item.id, locate(item.zone, { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 }, `Switch ${item.id}`));
  }
  const mechanisms = document.mechanisms.map(item => {
    const cells = (byOrigin.get(item.zone) ?? []).map(cell => cell.id);
    if (!cells.length) error(`${item.kind} ${item.id}`, `Its controlled room ${item.zone} was deleted. Remove or reassign this mechanism before building.`);
    return { kind: item.kind, id: item.id, cells };
  });
  return { starts, objects, teleports, cellTeleports, switches, mechanisms, diagnostics };
}

export function assertRoomReferences(plan: RoomReferencePlan): void {
  const errors = plan.diagnostics.filter(item => item.severity === 'error');
  if (errors.length) throw new Error(`The level has unresolved room references:\n${errors.map(item => `${item.subject}: ${item.message}`).join('\n')}`);
}
