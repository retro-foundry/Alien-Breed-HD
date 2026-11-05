import { activeNativeSwitches } from './nativeSwitches';
import { BinaryReader, checkedInteger } from './binary';
import { readGraphicRecord } from './graphics';
import { encodeWallSurface } from './surfaces';
import type { CellGraphics } from './cellGraphics';
import type { LevelDocument } from './level';
import type { PointLightingBinding } from './pointLighting';
import type { AnimationTables, SwitchAnimation } from './animationBuild';
import { importedAnimationTables } from './animationBuild';
import { roomSettings } from './roomHeights';
import type { Position } from './geometry';

const on = (p: Position, a: Position, b: Position) => Math.abs((p.x - a.x) * (b.z - a.z) - (p.z - a.z) * (b.x - a.x)) < 1e-6
  && p.x >= Math.min(a.x, b.x) - 1e-6 && p.x <= Math.max(a.x, b.x) + 1e-6 && p.z >= Math.min(a.z, b.z) - 1e-6 && p.z <= Math.max(a.z, b.z) + 1e-6;

/** leveld.asc LEVELSAVE (points at 901, panel at 1413/1445, slots at
 * 1962): a 64-unit panel centred on a lower wall, floor-64 to floor-32,
 * texture 11, 32 strips and a consecutive activation/draw point pair.
 * Native SwitchRoutine owns phase, texture and condition bit 4+slot. */
export function buildAuthoredSwitches(document: LevelDocument, input: CellGraphics, lighting: ReadonlyMap<number, PointLightingBinding>) {
  const points = [...input.topology.points], streams = new Map(input.streams), pointReferences = new Map(input.pointReferences);
  const addedLighting = new Map<number, PointLightingBinding>(), records = new Map<number, SwitchAnimation>();
  const occupied = new Set([...activeNativeSwitches(document).map(item => item.id), ...(document.roomSource.switchCopies ?? []).map(item => item.slot)]);
  for (const room of document.roomSource.rooms) for (const item of room.switches ?? []) {
    if (occupied.has(item.slot)) throw new Error(`Switch ${item.slot + 1} is already used. Choose an unused switch number.`);
    occupied.add(item.slot);
    const template = item.native === undefined ? undefined : importedAnimationTables(document).tables.switches?.[item.native];
    if (item.native !== undefined && (!template || template.zone < 0)) throw new Error(`Switch ${item.slot + 1} lost its native settings template.`);
    const bytes = template?.bytes.slice() ?? new Uint8Array(14);
    bytes[2] = item.automatic ? bytes[2] || 1 : 0;
    const originalPanel = template && document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === template.graphic - 10)?.wall;
    if (template && !originalPanel) throw new Error(`Switch ${item.slot + 1} lost its native panel template.`);
    const edges = room.boundaryMaterials?.edges, first = edges?.findIndex((edge, index) => edge === item.boundary && edges[(index + edges.length - 1) % edges.length] !== edge) ?? -1;
    if (first < 0) throw new Error(`Switch ${item.slot + 1} lost its wall. Reassign it to an existing wall or remove it before building.`);
    const indices = Array.from({ length: room.vertices.length }, (_, step) => (first + step) % room.vertices.length).filter(index => edges![index] === item.boundary);
    const a = room.vertices[first]!, b = room.vertices[(indices.at(-1)! + 1) % room.vertices.length]!;
    const length = Math.hypot(b.x - a.x, b.z - a.z);
    if (length < 64 || indices.some(index => !on(room.vertices[index]!, a, b))) throw new Error(`Switch ${item.slot + 1} needs a straight wall at least 64 units long. Reassign its wall.`);
    const centre = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 };
    const pair = [-32, 32].map(offset => ({ x: centre.x + offset * (b.x - a.x) / length, z: centre.z + offset * (b.z - a.z) / length }));
    const settings = roomSettings(document, room), topFixed = (settings.floor - 64) * 256, bottomFixed = (settings.floor - 32) * 256;
    const candidates = input.topology.cells.filter(cell => cell.sourceId === room.id).flatMap(cell => {
      const stream = streams.get(cell.id)!;
      const backing = stream.lower.flatMap((record, index) => {
        const graphic = readGraphicRecord(new BinaryReader(record.bytes, 'switch wall'), 0, points.length);
        if (graphic.type !== 0 || !graphic.wall || graphic.wall.topFixed > topFixed || graphic.wall.bottomFixed < bottomFixed) return [];
        const [p, q] = graphic.points.map(id => points[id]!);
        if (!on(p!, a, b) || !on(q!, a, b)) return [];
        const along = (point: Position) => ((point.x - pair[0]!.x) * (b.x - a.x) + (point.z - pair[0]!.z) * (b.z - a.z)) / length;
        return [{ cell, stream, index, wall: graphic.wall, p: graphic.points[0]!, from: Math.min(along(p!), along(q!)), to: Math.max(along(p!), along(q!)) }];
      });
      // A harmless inserted corner can split a solid wall beneath the panel.
      // Require complete backing in one cell, never bridge a doorway or gap.
      let covered = 0;
      for (const span of backing.toSorted((a, b) => a.from - b.from)) {
        if (span.from > covered + 1e-6) break;
        covered = Math.max(covered, span.to);
      }
      return covered >= 64 - 1e-6 ? backing.filter(span => span.from <= 32 && span.to >= 32).slice(0, 1) : [];
    });
    if (candidates.length !== 1) throw new Error(`Switch ${item.slot + 1} needs one solid lower wall behind its entire panel, 32 to 64 units above the floor. Close the passage, choose a longer wall, or reassign the switch.`);
    const target = candidates[0]!;
    // Keep winding identical to the supporting native wall.
    const [p, q] = target.wall.points.map(id => points[id]!);
    if ((q!.x - p!.x) * (b.x - a.x) + (q!.z - p!.z) * (b.z - a.z) < 0) pair.reverse();
    if (points.length + 2 > 2048) throw new Error('Switch panels exceed the engine limit of 2,048 points.');
    const point = points.length;
    const binding = lighting.get(target.p) ?? (target.p < document.points.length ? { kind: 'copy' as const, source: target.p } : undefined);
    if (!binding) throw new Error(`Switch ${item.slot + 1} has no supporting wall lighting. Assign the room's appearance before building.`);
    pair.forEach((position, index) => {
      points.push({ x: checkedInteger(Math.trunc(position.x), -32768, 32767, 'Switch panel X'), z: checkedInteger(Math.trunc(position.z), -32768, 32767, 'Switch panel Z') });
      addedLighting.set(point + index, binding);
    });
    const panel = { bytes: encodeWallSurface({ type: 0, points: [point, point + 1], stripStart: 0, stripEnd: 31,
      horizontalOffset: ((originalPanel?.horizontalOffset ?? 0) & 0x7c) | (bytes[10] ? 2 : 0), verticalOffset: (-settings.floor) & 31, texture: 11, verticalMask: 31, verticalShift: 5, horizontalMask: 31,
      topFixed, bottomFixed, brightness: target.wall.brightness }) };
    // Insert just after the supporting wall, before the original plane/object
    // commands. Recompute earlier authored indices in the same stream.
    const index = target.index + 1;
    for (const record of records.values()) if (record.target?.room === target.cell.id && record.target.index >= index) record.target = { ...record.target, index: record.target.index + 1 };
    streams.set(target.cell.id, { ...target.stream, lower: [...target.stream.lower.slice(0, index), panel, ...target.stream.lower.slice(index)] });
    pointReferences.set(target.cell.id, [...(pointReferences.get(target.cell.id) ?? []), point, point + 1]);
    records.set(item.slot, { bytes, zone: target.cell.id, point, graphic: 0, target: { room: target.cell.id, upper: false, index } });
  }
  return { graphics: { ...input, streams, pointReferences, topology: { ...input.topology, points } } satisfies CellGraphics, records, addedLighting };
}

export function appendAuthoredSwitches(tables: AnimationTables, records: ReadonlyMap<number, SwitchAnimation>): AnimationTables {
  if (!records.size) return tables;
  const switches = tables.switches ? [...tables.switches] : Array.from({ length: 8 }, (): SwitchAnimation => ({ bytes: new Uint8Array(14), zone: -1, point: 0, graphic: 0 }));
  for (const [slot, record] of records) {
    if (switches[slot]!.zone >= 0) throw new Error(`Switch ${slot + 1} is already active.`);
    switches[slot] = record;
  }
  return { ...tables, switches };
}
