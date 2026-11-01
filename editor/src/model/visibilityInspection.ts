import type { PlayableLevel } from './playableBuild';

/** Inspect the exact preview/export artifact, never old imported visibility
 * over newly edited outlines. Multiple convex cells remain individually selectable. */
export function visibilityInspection(artifact: PlayableLevel, room: string | undefined, zone: number | undefined, requested?: number) {
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
  return { cells, selected, rows: origin?.visibility ?? [], polygons, portals };
}
