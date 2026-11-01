import { normaliseRoomOutline, type Position } from './geometry';
import type { Graphic, LevelDocument } from './level';
import type { RoomTopology } from './roomTopology';
import { encodePolygonSurface, polygonForCell } from './surfaces';
import type { WallSurface } from './surfaces';
import { checkedInteger } from './binary';
import { roomSettings } from './roomHeights';

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
  const column = (distance: number) => {
    const texels = (distance / templateLength) * (template.stripEnd - template.stripStart), nearest = Math.round(texels);
    // Subtracting arc distances on a diagonal can put an exact texel boundary
    // a few floating-point ULPs below its integer. Preserve that boundary;
    // ordinary fractional columns still use the original truncation rule.
    const integral = Math.abs(texels - nearest) <= Number.EPSILON * 8 * Math.max(1, Math.abs(texels)) ? nearest : Math.trunc(texels);
    return checkedInteger(template.stripStart + integral, -32768, 32767, 'Generated wall texture column');
  };
  return { ...template, points: [...points], stripStart: column(startDistance), stripEnd: column(endDistance) };
}

export interface GeneratedPolygon {
  /** Original record identity retained for animation dependency remapping. */
  sourceOffset: number;
  bytes: Uint8Array;
}
export interface CellPolygons { lower: GeneratedPolygon[]; upper: GeneratedPolygon[] }
export interface SurfaceTranslations {
  rooms: ReadonlyMap<string, Position>;
  pointId: (x: number, z: number) => number;
}

function outlineKey(points: readonly Position[]): string {
  return JSON.stringify(normaliseRoomOutline(points.map(({ x, z }) => ({ x, z }))));
}

/** Bind complete room floor/roof/water material records to generated convex
 * cells. Heights, sampling scale, tile and brightness remain authored values.
 * Unchanged footprints retain the exact original corner order and tail bytes.
 * Proven whole draw-volume translations retain partial polygon footprints
 * and point order. Other reshaping needs explicit geometry mapping; never
 * expand a partial decoration across an edited room. Stream ordering and wall/animation
 * dependencies are assembled separately from this polygon stage.
 */
export function buildCellPolygons(document: LevelDocument, topology: RoomTopology, translations?: SurfaceTranslations): Map<number, CellPolygons> {
  const result = new Map<number, CellPolygons>();
  for (const cell of topology.cells) {
    if (result.has(cell.id)) throw new Error('Surface geometry contains duplicate runtime room identities.');
    const template = document.zones.find(zone => zone.id === cell.recordZone);
    if (!template) throw new Error(`Room ${cell.sourceId} has no surface settings template.`);
    const settings = roomSettings(document, cell);
    const outline = cell.vertices.map(id => {
      const point = topology.points[id];
      if (!point) throw new Error(`Room ${cell.sourceId} has a missing surface corner.`);
      return point;
    });
    const originalKey = outlineKey(template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })));
    const unchanged = outlineKey(outline) === originalKey;
    const layer = (records: readonly Graphic[], upper: boolean): GeneratedPolygon[] => records.filter(record => record.polygon).map(record => {
      const polygon = record.polygon!;
      const floor = upper ? 'upperFloor' : 'floor', roof = upper ? 'upperCeiling' : 'ceiling';
      const field = polygon.type === 2 ? roof : polygon.type === 7 ? undefined : floor;
      // Only a plane coinciding with its authored room boundary is anchored.
      // Decorative offsets and water retain their explicit source heights.
      const height = field && polygon.heightQuarter / 4 === template[field] ? settings[field] : polygon.heightQuarter / 4;
      const retainedPoints = polygon.points.every(id => {
        const original = document.points[id]!, generated = topology.points[id];
        return generated?.x === original.x && generated.z === original.z;
      });
      if (unchanged && retainedPoints) return { sourceOffset: record.offset, bytes: height === polygon.heightQuarter / 4
        ? document.files.graphics.slice(record.offset, record.offset + record.size) : encodePolygonSurface({ ...polygon, heightQuarter: height * 4 }) };
      const translation = translations?.rooms.get(cell.sourceId);
      if (translation) {
        const translatedPoint = (id: number) => {
          const point = document.points[id];
          if (!point) throw new Error(`Room ${cell.sourceId} surface has a missing translated source point ${id}.`);
          return translations!.pointId(point.x + translation.x, point.z + translation.z);
        };
        return { sourceOffset: record.offset, bytes: encodePolygonSurface({ ...polygon,
          points: polygon.points.map(translatedPoint), closingPoint: polygon.closingPoint < 0 ? polygon.closingPoint : translatedPoint(polygon.closingPoint), heightQuarter: height * 4 }) };
      }
      if (polygon.points.length < 3 || outlineKey(polygon.points.map(id => document.points[id]!)) !== originalKey)
        throw new Error(`Room ${cell.sourceId} surface at byte ${record.offset} has a partial or degenerate footprint. Assign its geometry before compiling this room.`);
      return { sourceOffset: record.offset, bytes: encodePolygonSurface(polygonForCell(polygon, cell.vertices, height)) };
    });
    result.set(cell.id, { lower: layer(template.lower, false), upper: layer(template.upper, true) });
  }
  return result;
}
