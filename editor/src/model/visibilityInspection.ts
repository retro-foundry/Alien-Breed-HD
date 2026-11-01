import type { PlayableLevel } from './playableBuild';
import { BinaryReader } from './binary';
import type { LevelDocument } from './level';

const clipTables = new WeakMap<LevelDocument, Map<number, { wordOffset: number; left: number[]; right: number[] }>>();
/** src/level.c level_assign_clips: saved nonnegative references are flags;
 * real offsets follow stream order across every zone, not the flag value. */
function assignedClips(document: LevelDocument) {
  const cached = clipTables.get(document); if (cached) return cached;
  const reader = new BinaryReader(document.files.clips, 'visibility clip inspection');
  const result = new Map<number, { wordOffset: number; left: number[]; right: number[] }>();
  let offset = 0;
  for (const zone of document.zones) for (const row of zone.visibility) if (row.clip >= 0) {
    const words = reader.wordsUntil(offset, -2, 8192), split = words.indexOf(-1);
    if (split < 0 || words.some((id, index) => index !== split && (!Number.isInteger(id) || id < 0 || !document.points[id])))
      throw new Error(`Zone ${zone.id} to ${row.zone} has invalid clipping boundary points. Repair its clip data before inspection.`);
    result.set(row.offset, { wordOffset: offset / 2, left: words.slice(0, split), right: words.slice(split + 1) });
    offset += (words.length + 1) * 2;
  }
  if (offset !== document.connectTableOffset) throw new Error('Visibility clipping stream does not match its connect-table boundary.');
  clipTables.set(document, result); return result;
}

/** Inspect the exact preview/export artifact, never old imported visibility
 * over newly edited outlines. Multiple convex cells remain individually selectable. */
export function visibilityInspection(artifact: PlayableLevel, room: string | undefined, zone: number | undefined, requested?: number, target?: number) {
  const source = room ? artifact.source.roomSource.rooms.find(item => item.id === room) : undefined;
  const cells = room ? source ? artifact.topology?.sourceCells.get(source.id)
    ?? (source.originZone === null ? [] : [source.originZone]) : [] : zone === undefined ? [] : [zone];
  const selected = cells.includes(requested ?? -1) ? requested! : cells[0];
  const origin = artifact.document.zones.find(item => item.id === selected);
  const visible = new Set(origin?.visibility.map(item => item.zone) ?? []);
  const polygons = artifact.document.zones.filter(item => item.id === selected || visible.has(item.id)).map(item => ({
    id: item.id, origin: item.id === selected,
    vertices: item.boundaryIds.map(id => artifact.document.lines[id]!).map(line => ({ x: line.x, z: line.z })),
  }));
  const portals = origin?.boundaryIds.map(id => artifact.document.lines[id]!).filter(line => line.connect >= 0) ?? [];
  const clips = assignedClips(artifact.document);
  const rows = (origin?.visibility ?? []).map(row => ({ ...row, bounds: clips.get(row.offset) }));
  const selectedTarget = rows.some(row => row.zone === target) ? target : rows.find(row => row.bounds)?.zone;
  const bounds = rows.find(row => row.zone === selectedTarget)?.bounds;
  const clipPoints = (['left', 'right'] as const).flatMap(side => (bounds?.[side] ?? []).map(id => ({ ...artifact.document.points[id]!, side })));
  return { cells, selected, rows, polygons, portals, target: selectedTarget, clipPoints };
}
