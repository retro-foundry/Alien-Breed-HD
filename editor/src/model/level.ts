/* Current runtime contract: src/level.c, src/game_types.h and the shipped maps.
 * The older AMOS LEVELSAVE uses different header/graphics encodings. */
import { BinaryReader, LevelFormatError } from './binary';
import { unpackSb } from './sb';
import { readWallSurface, readPolygonSurface, type WallSurface, type PolygonSurface } from './surfaces';

export interface Point { id: number; x: number; z: number; offset: number }
export interface FloorLine {
  id: number; offset: number; x: number; z: number; dx: number; dz: number;
  connect: number; length: number;
}
export interface Graphic {
  type: number; offset: number; size: number; points: number[];
  height?: number; top?: number; bottom?: number; texture?: number;
  wall?: WallSurface; polygon?: PolygonSurface;
}
export interface Zone {
  id: number; fileId: number; offset: number;
  floor: number; ceiling: number; upperFloor: number; upperCeiling: number; water: number;
  brightness: number; upperBrightness: number; control: number; upperControl: number;
  backdrop: boolean; teleport: { zone: number; x: number; z: number };
  floorSound: number; upperFloorSound: number;
  boundaryIds: number[]; wallIds: number[]; visibleZones: number[];
  rotationPointIds: number[]; rotationOffset: number; exitListOffset: number; dataEnd: number;
  visibility: { zone: number; clip: number; mask: number; offset: number }[];
  lower: Graphic[]; upper: Graphic[];
}
export interface LevelObject {
  id: number; cid: number; offset: number; pointOffset: number;
  type: number; zone: number; x: number; z: number; y: number; health: number; upper: boolean;
}
export interface Mechanism {
  id: number; kind: 'door' | 'lift'; offset: number; zone: number;
  bottom: number; top: number; current: number; direction: number; conditions: number;
  openRule: number; closeRule: number; walls: { line: number; graphic: number; base: number }[];
}
export interface Switch {
  id: number; offset: number; zone: number; type: number; countdown: number;
  point: number; graphic: number; state: Uint8Array;
}
export interface Start { player: 1 | 2; x: number; z: number; zone: number }
export interface LevelFiles { data: Uint8Array; graphics: Uint8Array; clips: Uint8Array }
export interface LevelDocument {
  name: string;
  files: LevelFiles;
  encoded: LevelFiles;
  original: LevelFiles;
  points: Point[];
  lines: FloorLine[];
  zones: Zone[];
  objects: LevelObject[];
  mechanisms: Mechanism[];
  switches: Switch[];
  starts: Start[];
  controlPoints: { id: number; offset: number; x: number; z: number; y: number; upper: number }[];
  pointBrightnessOffset: number;
  objectPointsOffset: number;
  connectTableOffset: number;
}

export const OBJECT_NAMES = [
  'Alien', 'Health', 'Projectile', 'Weapon', 'Key', 'Player 1', 'Robot', 'Large enemy',
  'Floater', 'Ammunition', 'Barrel', 'Player 2', 'Rifle marine', 'Worm', 'Large red alien',
  'Small red alien', 'Tree', 'Eyeball', 'Tough marine', 'Flame marine', 'Flame spout',
];

export function objectName(type: number): string {
  return type === -1 ? 'Remains / decoration' : (OBJECT_NAMES[type] ?? `Object type ${type}`);
}

function graphicsStream(reader: BinaryReader, offset: number, points: number): Graphic[] {
  if (offset === 0) return [];
  reader.require(offset, 2);
  const result: Graphic[] = [];
  let cursor = offset + 2;
  for (let count = 0; count < 4096; count++) {
    const type = reader.i16(cursor);
    if (type < 0) return result;
    const entry: Graphic = { type, offset: cursor, size: 0, points: [] };
    const payload = cursor + 2;
    let size: number;
    if (type === 0 || type === 13 || type === 5) {
      size = 30;
      entry.points = [reader.i16(payload), reader.i16(payload + 2)];
      if (type !== 5) {
        entry.wall = readWallSurface(reader, cursor);
        entry.texture = entry.wall.texture;
        entry.top = entry.wall.topFixed / 256;
        entry.bottom = entry.wall.bottomFixed / 256;
      }
    } else if (type === 3 || type === 12) size = 2;
    else if (type === 4) size = 4;
    else if (type === 6) size = 6;
    else if ([1, 2, 7, 8, 9, 10, 11].includes(type)) {
      const sides = reader.i16(payload + 2) + 1;
      if (sides < 0 || sides > 100) throw new LevelFormatError('polygon has an invalid vertex count', reader.name, payload + 2);
      size = 14 + sides * 2;
      entry.polygon = readPolygonSurface(reader, cursor);
      entry.height = entry.polygon.heightQuarter / 4;
      entry.points = entry.polygon.points;
      entry.texture = entry.polygon.texture;
    } else throw new LevelFormatError(`unrecognised graphics opcode ${type}`, reader.name, cursor);
    reader.require(cursor, size);
    if (entry.points.some(point => point < 0 || point >= points))
      throw new LevelFormatError('graphic references a point outside the point table', reader.name, cursor);
    entry.size = size;
    result.push(entry);
    cursor += size;
  }
  throw new LevelFormatError('graphics stream has no terminator', reader.name, offset);
}

function mechanismTable(reader: BinaryReader, offset: number, kind: 'door' | 'lift'): Mechanism[] {
  if (offset <= 16) return [];
  const result: Mechanism[] = [];
  let cursor = offset;
  for (let i = 0; i < 256; i++) {
    if (reader.i16(cursor) === 999) return result;
    reader.require(cursor, 18);
    const mechanism: Mechanism = {
      id: i, kind, offset: cursor, bottom: reader.i16(cursor) / 4, top: reader.i16(cursor + 2) / 4,
      current: reader.i16(cursor + 4) / 4, direction: reader.i16(cursor + 6),
      zone: reader.i16(cursor + 12), conditions: reader.u16(cursor + 14),
      openRule: reader.u8(cursor + 16), closeRule: reader.u8(cursor + 17), walls: [],
    };
    cursor += 18;
    let terminated = false;
    for (let wall = 0; wall < 64; wall++) {
      const line = reader.i16(cursor);
      if (line < 0) { cursor += 2; terminated = true; break; }
      mechanism.walls.push({ line, graphic: reader.i32(cursor + 2), base: reader.i32(cursor + 6) });
      cursor += 10;
    }
    if (!terminated) throw new LevelFormatError(`${kind} wall list has no terminator`, reader.name, cursor);
    result.push(mechanism);
  }
  throw new LevelFormatError(`${kind} table has no 999 terminator`, reader.name, offset);
}

export function parseLevel(name: string, files: LevelFiles, encoded: LevelFiles = files, original: LevelFiles = files): LevelDocument {
  const data = new BinaryReader(files.data, 'twolev.bin');
  const graphics = new BinaryReader(files.graphics, 'twolev.graph.bin');
  data.require(0, 54); graphics.require(0, 16);
  const pointCount = data.i16(14) + 1;
  const graphTable = graphics.i32(12);
  const zoneSlots = (graphTable - 16) / 4;
  if (pointCount < 1 || pointCount > 2048)
    throw new LevelFormatError('point count is outside the engine limit of 2,048', data.name, 14);
  if (!Number.isInteger(zoneSlots) || zoneSlots < 1 || zoneSlots > 256)
    throw new LevelFormatError('invalid zone offset table (maximum 256 zones)', graphics.name, 12);
  graphics.require(16, zoneSlots * 4); graphics.require(graphTable, zoneSlots * 8);
  const pointOffset = data.i32(22), floorOffset = data.i32(26), objectOffset = data.i32(30);
  const objectPointsOffset = data.i32(42), objectPointCount = data.i16(20) + 1;
  data.require(pointOffset, pointCount * 8);
  if (objectPointCount < 1 || objectPointCount > 2048)
    throw new LevelFormatError('invalid object point count', data.name, 20);
  data.require(objectPointsOffset, objectPointCount * 8);
  const points = Array.from({ length: pointCount }, (_, id) => ({
    id, offset: pointOffset + id * 4, x: data.i16(pointOffset + id * 4), z: data.i16(pointOffset + id * 4 + 2),
  }));
  const zones: Zone[] = [];
  let maxLine = -1;
  for (let id = 0; id < zoneSlots; id++) {
    const offset = graphics.i32(16 + id * 4);
    if (offset < 0) continue;
    data.require(offset, 48);
    const exitListOffset = offset + data.i16(offset + 32);
    const lists = data.wordsUntil(exitListOffset, -2, 256);
    const separator = lists.indexOf(-1);
    if (separator === -1) throw new LevelFormatError('zone has no wall-list separator', data.name, offset + 32);
    const boundaryIds = lists.slice(0, separator), wallIds = lists.slice(separator + 1);
    if (lists.some(line => line < -1)) throw new LevelFormatError('invalid collision line index', data.name, offset + 32);
    for (const line of lists) if (line > maxLine) maxLine = line;
    const visibleZones: number[] = [];
    const visibility: Zone['visibility'] = [];
    let cursor = offset + 48, terminated = false;
    for (let i = 0; i < 512; i++) {
      const zone = data.i16(cursor);
      if (zone < 0) { terminated = true; break; }
      data.require(cursor, 8); visibleZones.push(zone);
      visibility.push({ zone, clip: data.i16(cursor + 2), mask: data.u32(cursor + 4), offset: cursor });
      cursor += 8;
    }
    if (!terminated) throw new LevelFormatError('zone visibility list has no terminator', data.name, offset + 48);
    const rotationOffset = offset + data.i16(offset + 34);
    const rotationPointIds = data.wordsUntil(rotationOffset, -1, 2049);
    if (rotationPointIds.some(id => id < 0 || id >= pointCount)) throw new LevelFormatError('rotation list references a missing point', data.name, rotationOffset);
    // LEVELSAVE terminates visibility with a long -1, and the other two
    // lists with words. Account for every byte when rebuilding the zone tail.
    const dataEnd = Math.max(cursor + 4, exitListOffset + (lists.length + 1) * 2, rotationOffset + (rotationPointIds.length + 1) * 2);
    zones.push({
      id, fileId: data.i16(offset), offset,
      floor: data.i32(offset + 2) / 256, ceiling: data.i32(offset + 6) / 256,
      upperFloor: data.i32(offset + 10) / 256, upperCeiling: data.i32(offset + 14) / 256,
      water: data.i32(offset + 18) / 256, brightness: data.i16(offset + 22), upperBrightness: data.i16(offset + 24),
      control: data.i8(offset + 26), upperControl: data.i8(offset + 27),
      backdrop: data.i16(offset + 36) !== 0,
      teleport: { zone: data.i16(offset + 38), x: data.i16(offset + 40), z: data.i16(offset + 42) },
      floorSound: data.i16(offset + 44), upperFloorSound: data.i16(offset + 46),
      boundaryIds, wallIds, visibleZones, visibility, rotationPointIds, rotationOffset, exitListOffset, dataEnd,
      lower: graphicsStream(graphics, graphics.i32(graphTable + id * 8), pointCount),
      upper: graphicsStream(graphics, graphics.i32(graphTable + id * 8 + 4), pointCount),
    });
  }
  data.require(floorOffset, (maxLine + 1) * 16);
  const lines = Array.from({ length: maxLine + 1 }, (_, id): FloorLine => {
    const offset = floorOffset + id * 16;
    return { id, offset, x: data.i16(offset), z: data.i16(offset + 2), dx: data.i16(offset + 4),
      dz: data.i16(offset + 6), connect: data.i16(offset + 8), length: data.i16(offset + 10) };
  });
  const objects: LevelObject[] = [];
  let objectTerminated = false;
  for (let id = 0; id < 256; id++) {
    const offset = objectOffset + id * 64, cid = data.i16(offset);
    if (cid < 0) { objectTerminated = true; break; }
    data.require(offset, 64);
    if (cid >= objectPointCount) throw new LevelFormatError('object point index is outside the point table', data.name, offset);
    const pointOffset = objectPointsOffset + cid * 8;
    objects.push({ id, cid, offset, pointOffset, type: data.i8(offset + 16), zone: data.i16(offset + 12),
      x: data.i32(pointOffset) / 65536, z: data.i32(pointOffset + 4) / 65536,
      y: data.i16(offset + 4) / 4, health: data.u8(offset + 18), upper: data.u8(offset + 63) !== 0 });
  }
  if (!objectTerminated) throw new LevelFormatError('object table has no terminator', data.name, objectOffset);
  const mechanisms = [...mechanismTable(graphics, graphics.i32(0), 'door'), ...mechanismTable(graphics, graphics.i32(4), 'lift')];
  const switches: Switch[] = [];
  const switchesOffset = graphics.i32(8);
  if (switchesOffset > 16) for (let id = 0; id < 8; id++) {
    const offset = switchesOffset + id * 14;
    graphics.require(offset, 14);
    // Anims.s SwitchRoutine / objects.c switches: +2/+3 are type/countdown,
    // +4 is a point pair, +6 a graphics pointer, +10..13 runtime state.
    switches.push({ id, offset, zone: graphics.i16(offset), type: graphics.u8(offset + 2), countdown: graphics.i8(offset + 3),
      point: graphics.i16(offset + 4), graphic: graphics.i32(offset + 6), state: graphics.bytes.slice(offset + 10, offset + 14) });
  }
  const controlCount = data.i16(12);
  if (controlCount < 0 || controlCount > 100) throw new LevelFormatError('invalid control-point count', data.name, 12);
  const controlPoints = Array.from({ length: controlCount }, (_, id) => {
    const offset = 54 + id * 8;
    return { id, offset, x: data.i16(offset), z: data.i16(offset + 2), y: data.i16(offset + 4), upper: data.i16(offset + 6) };
  });
  const starts: Start[] = [
    { player: 1, x: data.i16(0), z: data.i16(2), zone: data.i16(4) },
    { player: 2, x: data.i16(6), z: data.i16(8), zone: data.i16(10) },
  ];
  const clips = new BinaryReader(files.clips, 'twolev.clips');
  let connectTableOffset = 0;
  for (const zone of zones) for (const entry of zone.visibility) if (entry.clip >= 0) {
    const words = clips.wordsUntil(connectTableOffset, -2, 8192);
    connectTableOffset += (words.length + 1) * 2;
  }
  clips.require(connectTableOffset, 2);
  return { name, files, encoded, original, points, lines, zones, objects, mechanisms, switches, starts, controlPoints,
    pointBrightnessOffset: pointOffset + pointCount * 4, objectPointsOffset, connectTableOffset };
}

export function openLevel(name: string, encoded: LevelFiles): LevelDocument {
  const files = { data: unpackSb(encoded.data, 'twolev.bin'), graphics: unpackSb(encoded.graphics, 'twolev.graph.bin'),
    clips: unpackSb(encoded.clips, 'twolev.clips') };
  return parseLevel(name, files, encoded);
}

export function zoneContainsPoint(document: LevelDocument, zone: Zone, x: number, z: number): boolean {
  let inside = false;
  for (const id of [...zone.boundaryIds, ...zone.wallIds]) {
    const line = document.lines[id];
    if (!line) continue;
    const x2 = line.x + line.dx, z2 = line.z + line.dz;
    const cross = (x - line.x) * line.dz - (z - line.z) * line.dx;
    if (cross === 0 && x >= Math.min(line.x, x2) && x <= Math.max(line.x, x2) && z >= Math.min(line.z, z2) && z <= Math.max(line.z, z2)) return true;
    if ((line.z > z) !== (z2 > z) && x < line.x + line.dx * (z - line.z) / line.dz) inside = !inside;
  }
  return inside;
}

export function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}

export function exportLevel(document: LevelDocument): LevelFiles {
  const result = {} as LevelFiles;
  for (const name of ['data', 'graphics', 'clips'] as const)
    result[name] = new Uint8Array(sameBytes(document.files[name], document.original[name]) ? document.encoded[name] : document.files[name]);
  return result;
}
