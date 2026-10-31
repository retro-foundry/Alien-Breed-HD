import { normaliseRoomOutline, type Position } from './geometry';
import type { Graphic, LevelDocument } from './level';
import type { RoomTopology } from './roomTopology';
import { encodePolygonSurface, polygonForCell } from './surfaces';
import type { WallSurface } from './surfaces';
import { checkedInteger } from './binary';

/** Bind an explicit interval along a generated wall chain using the selected
 * record's authored texels/world-unit ratio. The caller supplies distances
 * from that chain's origin, so split endpoints share exactly one texture
 * coordinate. This does not infer the AMOS repeat/fit setting from tile sizes
 * or replace independent horizontal/vertical phase and sampling fields.
 */
export function wallForSpan(template: WallSurface, templateLength: number, points: [number, number],
  startDistance: number, endDistance: number): WallSurface {
  if (!Number.isFinite(templateLength) || templateLength <= 0 || !Number.isFinite(startDistance) || !Number.isFinite(endDistance)
    || startDistance < 0 || endDistance <= startDistance) throw new Error('Wall binding requires a nonzero material reference length and an increasing physical span.');
  points.forEach(point => checkedInteger(point, 0, 2047, 'Wall point index'));
  if (points.length !== 2 || points[0] === points[1]) throw new Error('A generated wall needs two different endpoints.');
  const column = (distance: number) => checkedInteger(template.stripStart + Math.trunc((distance / templateLength) * (template.stripEnd - template.stripStart)),
    -32768, 32767, 'Generated wall texture column');
  return { ...template, points: [...points], stripStart: column(startDistance), stripEnd: column(endDistance) };
}

export interface GeneratedPolygon {
  /** Original record identity retained for animation dependency remapping. */
  sourceOffset: number;
  bytes: Uint8Array;
}
export interface CellPolygons { lower: GeneratedPolygon[]; upper: GeneratedPolygon[] }

function outlineKey(points: readonly Position[]): string {
  return JSON.stringify(normaliseRoomOutline(points.map(({ x, z }) => ({ x, z }))));
}

/** Bind complete room floor/roof/water material records to generated convex
 * cells. Heights, sampling scale, tile and brightness remain authored values.
 * Unchanged footprints retain the exact original corner order and tail bytes.
 * Partial decorative polygons need their own geometry mapping; do not expand
 * them silently across an edited room. Stream ordering and wall/animation
 * dependencies are assembled separately from this polygon stage.
 */
export function buildCellPolygons(document: LevelDocument, topology: RoomTopology): Map<number, CellPolygons> {
  const result = new Map<number, CellPolygons>();
  for (const cell of topology.cells) {
    if (result.has(cell.id)) throw new Error('Surface geometry contains duplicate runtime room identities.');
    const template = document.zones.find(zone => zone.id === cell.recordZone);
    if (!template) throw new Error(`Room ${cell.sourceId} has no surface settings template.`);
    const outline = cell.vertices.map(id => {
      const point = topology.points[id];
      if (!point) throw new Error(`Room ${cell.sourceId} has a missing surface corner.`);
      return point;
    });
    const originalKey = outlineKey(template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })));
    const unchanged = outlineKey(outline) === originalKey;
    const layer = (records: readonly Graphic[]): GeneratedPolygon[] => records.filter(record => record.polygon).map(record => {
      const polygon = record.polygon!;
      const retainedPoints = polygon.points.every(id => {
        const original = document.points[id]!, generated = topology.points[id];
        return generated?.x === original.x && generated.z === original.z;
      });
      if (unchanged && retainedPoints) return { sourceOffset: record.offset, bytes: document.files.graphics.slice(record.offset, record.offset + record.size) };
      if (polygon.points.length < 3 || outlineKey(polygon.points.map(id => document.points[id]!)) !== originalKey)
        throw new Error(`Room ${cell.sourceId} surface at byte ${record.offset} has a partial or degenerate footprint. Assign its geometry before compiling this room.`);
      return { sourceOffset: record.offset, bytes: encodePolygonSurface(polygonForCell(polygon, cell.vertices, polygon.heightQuarter / 4)) };
    });
    result.set(cell.id, { lower: layer(template.lower), upper: layer(template.upper) });
  }
  return result;
}
