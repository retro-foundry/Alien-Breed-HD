import { BinaryReader, checkedInteger } from './binary';
import type { LevelDocument } from './level';
import { setCornerLight } from './cornerLights';
import { buildPlayableLevel } from './playableBuild';
import type { RoomChange } from './roomSource';
import { roomSettings } from './roomHeights';

/** leveld.asc PRESETS / UPPERPRESETS keys 1–0. Keep the original
 * unsigned low-byte words (255/241), rather than changing retained bits. */
export const roomLightPresets = [
  { label: 'Bright, lit from above', floor: 255, ceiling: 1 },
  { label: 'Bright, lit from below', floor: 1, ceiling: 255 },
  { label: 'Bright bottom, dark top', floor: 1, ceiling: 241 },
  { label: 'Bright top, dark bottom', floor: 241, ceiling: 1 },
  { label: 'Dark, lit from above', floor: 241, ceiling: 15 },
  { label: 'Dark, lit from below', floor: 15, ceiling: 241 },
  { label: 'Ceiling glowing', floor: 241, ceiling: 'glow' },
  { label: 'Floor glowing', floor: 'glow', ceiling: 241 },
  { label: 'Glowing in sync', floor: 'glow', ceiling: 'glow' },
  { label: 'Glowing out of sync', floor: 'complement', ceiling: 'glow' },
] as const;

export function applyRoomLightPreset(document: LevelDocument, id: string, upper: boolean, preset: number): RoomChange {
  checkedInteger(preset, 0, roomLightPresets.length - 1, 'Room lighting preset');
  const room = document.roomSource.rooms.find(item => item.id === id);
  if (!room) throw new Error('Select a room before applying a lighting preset.');
  const settings = roomSettings(document, room);
  if (upper && !(settings.upper.length && settings.upperFloor > settings.upperCeiling)) throw new Error('Add an upper storey before applying its lighting preset.');
  const built = buildPlayableLevel(document, -1), reader = new BinaryReader(built.document.files.data, 'room preset lighting');
  const cells = built.topology ? built.topology.cells.filter(cell => cell.sourceId === id).map(cell => cell.id) : [room.originZone!];
  // ZO(CZ,A) is a geometric point number in the original editor. Resolve
  // that number before per-surface lighting clones, and persist the resulting
  // word so later moves/copies/rebuilds do not choose a different pattern.
  const geometricIds = new Map<number, number>();
  const geometricPoints = built.topology ? built.topology.cells.filter(cell => cell.sourceId === id).flatMap(cell => cell.vertices) : document.points.map(point => point.id);
  for (const corner of room.vertices.keys()) {
    const point = room.vertices[corner]!;
    const pointId = geometricPoints.find(pointId => built.document.points[pointId]!.x === point.x && built.document.points[pointId]!.z === point.z);
    if (pointId !== undefined) geometricIds.set(corner, pointId);
  }
  let source = document;
  const targets = new Map<string, { corner: number; record: number; lowerBrightness: number; upperBrightness: number }>();
  for (const cell of cells) {
    const zone = built.document.zones[cell]!;
    for (const face of upper ? zone.upper : zone.lower) {
      if (![1, 2].includes(face.type)) continue;
      const record = built.surfaceSources ? built.surfaceSources.find(item => item.room === cell && item.upper === upper && item.offset === face.offset)?.sourceOffset : face.offset;
      if (record === undefined) throw new Error('This surface has no source lighting identity. Restore its material before applying a room lighting preset.');
      for (const pointId of face.points) {
        const point = built.document.points[pointId]!, corner = room.vertices.findIndex(vertex => vertex.x === point.x && vertex.z === point.z);
        // Decomposition introduces interior vertices whose values are derived
        // from the authored outline. Decorative vertices need their own editor.
        if (corner < 0) continue;
        const offset = built.document.pointBrightnessOffset + pointId * 4;
        const value = { corner, record, lowerBrightness: reader.i16(offset), upperBrightness: reader.i16(offset + 2) };
        const setting = face.type === 1 ? roomLightPresets[preset]!.floor : roomLightPresets[preset]!.ceiling;
        let brightness: number;
        if (typeof setting === 'number') brightness = setting;
        else {
          const pointNumber = geometricIds.get(corner);
          if (pointNumber === undefined) throw new Error('This corner has no geometric point number for the original glowing preset. Restore its outline before applying.');
          if (setting === 'complement' && pointNumber > 1000) throw new Error('The original out-of-sync preset needs point numbers at most 1,000. Use another preset or assign corner animations individually.');
          const selector = ((setting === 'complement' ? 1000 - pointNumber : pointNumber) % 5) + 1;
          brightness = (0xf000 + selector * 0x100) << 16 >> 16;
        }
        if (upper) value.upperBrightness = brightness; else value.lowerBrightness = brightness;
        const key = `${corner}:${record}`, previous = targets.get(key);
        if (previous && (previous.lowerBrightness !== value.lowerBrightness || previous.upperBrightness !== value.upperBrightness)) throw new Error('This corner has separate lighting identities on one surface. Edit its corner lighting individually.');
        targets.set(key, value);
      }
    }
  }
  if (!targets.size) throw new Error('Room lighting presets need a normal floor or ceiling. Water and retained bump surfaces use their own shading.');
  for (const { corner, record, lowerBrightness, upperBrightness } of targets.values()) {
    source = { ...source, roomSource: setCornerLight(source, id, corner, record, { lowerBrightness, upperBrightness }).after };
  }
  // Keep this a single validated transaction, including rooms split into cells.
  buildPlayableLevel(source, -1);
  return { label: `Apply ${upper ? 'upper' : 'lower'} room lighting preset`, before: document.roomSource, after: source.roomSource };
}
