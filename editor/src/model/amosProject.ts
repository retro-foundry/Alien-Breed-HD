/* Original editable project contract: LEVELSAVE lines 1987–2119 and
 * LEVELLOAD lines 2859–3023 in amiga/archive/amos/leveld.asc. These are source
 * files, not the shipped twolev.bin/graph.bin/clips runtime representation. */
import { BinaryReader, checkedInteger, LevelFormatError } from './binary';
import { BinaryWriter } from './writer';

export interface AmosFiles { data: Uint8Array; objects: Uint8Array; links: Uint8Array }
export interface AmosPosition { x: number; y: number }
export interface AmosGroup { conditions: number; openRule: number; closeRule: number; options: number[] }
export interface AmosLiftGroup extends AmosGroup { startPosition: number }
export interface AmosCorner {
  point: number; wallType: number; corner: number; lowerWall: number[]; upperWall: number[];
  door: number; lift: number; brightness: number; upperBrightness: number; vertexLight: number[];
}
export interface AmosZone {
  teleport: { zone: number; x: number; y: number }; brightness: number; upperBrightness: number;
  control: number; upperControl: number; corners: AmosCorner[];
  lowerHeights: number[]; upperHeights: number[]; materials: Uint8Array;
  door: number; lift: number; roofBump: number; floorBump: number;
  visibility: { zone: number; mask: number }[];
}
export interface AmosProject {
  controlPoints: { x: number; y: number; zone: number; upper: number }[];
  doors: AmosGroup[]; lifts: AmosLiftGroup[]; switches: { zone: number; wall: number }[];
  objectPoints: AmosPosition[]; points: AmosPosition[];
  // The AMOS saver writes NZ+1 slots, including an extra source-only slot.
  zoneCount: number; zones: AmosZone[];
  starts: { x: number; y: number; zone: number }[];
  introduction: Uint8Array; exitZone: number; backgroundSounds: number[]; echoes: number[];
  objects: Uint8Array; links: Uint8Array; trailing: Uint8Array;
}

export function readAmosProject(files: AmosFiles): AmosProject {
  if (files.data.length > 16 * 1024 * 1024) throw new Error('The AMOS project data exceeds 16 MiB.');
  if (files.objects.length !== 200 * 32) throw new Error('twolev.obj must contain the original 200 × 32-byte object bank.');
  if (files.links.length !== 100 * 100) throw new Error('twolev.links must contain the original 100 × 100-byte link bank.');
  const data = new BinaryReader(new Uint8Array(files.data), 'twolev.dat');
  let cursor = 0;
  const word = () => { const n = data.i16(cursor); cursor += 2; return n; };
  const long = () => { const n = data.i32(cursor); cursor += 4; return n; };
  const byte = () => { const n = data.u8(cursor); cursor++; return n; };
  const block = (size: number) => { data.require(cursor, size); const result = data.bytes.slice(cursor, cursor + size); cursor += size; return result; };
  const count = (value: number, maximum: number, name: string) => {
    if (!Number.isInteger(value) || value < 0 || value > maximum) throw new LevelFormatError(`invalid ${name} count (${value})`, data.name, cursor);
    return value;
  };
  const controlPoints = Array.from({ length: count(long() + 1, 100, 'control point') }, () => ({ x: long(), y: long(), zone: word(), upper: word() }));
  const doors = Array.from({ length: 21 }, (): AmosGroup => ({ conditions: word() & 65535, openRule: word(), closeRule: word(), options: Array.from({ length: 7 }, word) }));
  const lifts = Array.from({ length: 21 }, (): AmosLiftGroup => ({ conditions: word() & 65535, startPosition: word(), openRule: word(), closeRule: word(), options: Array.from({ length: 7 }, word) }));
  const switches = Array.from({ length: 8 }, () => ({ zone: word(), wall: word() }));
  const objectPoints = Array.from({ length: count(word(), 200, 'object point') }, () => ({ x: long(), y: long() }));
  const points = Array.from({ length: count(word() + 1, 601, 'point') }, () => ({ x: long(), y: long() }));
  const zoneCount = count(word(), 256, 'zone');
  const zones = Array.from({ length: zoneCount + 1 }, (): AmosZone => {
    const teleport = { zone: long(), x: long(), y: long() };
    const brightness = word(), upperBrightness = word(), control = word(), upperControl = word();
    const sides = count(word(), 10, 'room corner');
    const corners = Array.from({ length: sides + 1 }, (): AmosCorner => ({
      point: word(), wallType: word(), corner: word(), lowerWall: Array.from({ length: 4 }, word),
      upperWall: Array.from({ length: 4 }, word), door: word(), lift: word(), brightness: word(),
      upperBrightness: word(), vertexLight: Array.from({ length: 4 }, word),
    }));
    const lowerHeights = Array.from({ length: 4 }, long), upperHeights = Array.from({ length: 4 }, long);
    const materials = block(8), door = word(), lift = word(), roofBump = byte(), floorBump = byte();
    const visibility = Array.from({ length: 64 }, () => ({ zone: word(), mask: long() >>> 0 }));
    return { teleport, brightness, upperBrightness, control, upperControl, corners, lowerHeights, upperHeights, materials, door, lift, roofBump, floorBump, visibility };
  });
  const starts = Array.from({ length: 2 }, () => ({ x: long(), y: long(), zone: word() }));
  const introduction = block(1600), exitZone = word();
  const backgroundSounds = Array.from({ length: zoneCount }, long), echoes = Array.from({ length: zoneCount }, long);
  return { controlPoints, doors, lifts, switches, objectPoints, points, zoneCount, zones, starts, introduction, exitZone, backgroundSounds, echoes,
    objects: new Uint8Array(files.objects), links: new Uint8Array(files.links), trailing: data.bytes.slice(cursor) };
}

export function writeAmosProject(project: AmosProject): AmosFiles {
  const output = new BinaryWriter();
  const length = (values: { length: number }, expected: number, label: string) => {
    if (values.length !== expected) throw new Error(`${label} must contain ${expected} entries.`);
  };
  checkedInteger(project.controlPoints.length, 0, 100, 'Control point count');
  checkedInteger(project.objectPoints.length, 0, 200, 'Object count');
  checkedInteger(project.points.length, 0, 601, 'Point count');
  checkedInteger(project.zoneCount, 0, 256, 'Zone count');
  length(project.doors, 21, 'Door groups'); length(project.lifts, 21, 'Lift groups'); length(project.switches, 8, 'Switches');
  length(project.zones, project.zoneCount + 1, 'Zone slots'); length(project.starts, 2, 'Player starts');
  length(project.introduction, 1600, 'Introduction'); length(project.backgroundSounds, project.zoneCount, 'Background sounds'); length(project.echoes, project.zoneCount, 'Echoes');
  length(project.objects, 6400, 'Object bank'); length(project.links, 10000, 'Link bank');
  const words = (values: number[], expected: number, label: string) => { length(values, expected, label); values.forEach(value => output.word(value)); };
  output.long(project.controlPoints.length - 1);
  for (const point of project.controlPoints) { output.long(point.x); output.long(point.y); output.word(point.zone); output.word(point.upper); }
  for (const group of project.doors) { output.word(group.conditions); output.word(group.openRule); output.word(group.closeRule); words(group.options, 7, 'Door options'); }
  for (const group of project.lifts) { output.word(group.conditions); output.word(group.startPosition); output.word(group.openRule); output.word(group.closeRule); words(group.options, 7, 'Lift options'); }
  for (const entry of project.switches) { output.word(entry.zone); output.word(entry.wall); }
  output.word(project.objectPoints.length);
  for (const point of project.objectPoints) { output.long(point.x); output.long(point.y); }
  output.word(project.points.length - 1);
  for (const point of project.points) { output.long(point.x); output.long(point.y); }
  output.word(project.zoneCount);
  for (const zone of project.zones) {
    checkedInteger(zone.corners.length, 1, 11, 'Corner records (including closing corner)');
    output.long(zone.teleport.zone); output.long(zone.teleport.x); output.long(zone.teleport.y);
    output.word(zone.brightness); output.word(zone.upperBrightness); output.word(zone.control); output.word(zone.upperControl);
    output.word(zone.corners.length - 1);
    for (const corner of zone.corners) {
      output.word(corner.point); output.word(corner.wallType); output.word(corner.corner);
      words(corner.lowerWall, 4, 'Lower wall definition'); words(corner.upperWall, 4, 'Upper wall definition');
      output.word(corner.door); output.word(corner.lift); output.word(corner.brightness); output.word(corner.upperBrightness);
      words(corner.vertexLight, 4, 'Vertex light');
    }
    length(zone.lowerHeights, 4, 'Lower heights'); length(zone.upperHeights, 4, 'Upper heights');
    zone.lowerHeights.forEach(value => output.long(value)); zone.upperHeights.forEach(value => output.long(value));
    length(zone.materials, 8, 'Material definition'); output.write(zone.materials);
    output.word(zone.door); output.word(zone.lift); output.byte(zone.roofBump); output.byte(zone.floorBump);
    length(zone.visibility, 64, 'Original visibility cache');
    for (const entry of zone.visibility) { output.word(entry.zone); output.long(entry.mask); }
  }
  for (const start of project.starts) { output.long(start.x); output.long(start.y); output.word(start.zone); }
  output.write(project.introduction); output.word(project.exitZone);
  project.backgroundSounds.forEach(value => output.long(value)); project.echoes.forEach(value => output.long(value));
  output.write(project.trailing);
  return { data: output.finish(), objects: project.objects.slice(), links: project.links.slice() };
}
