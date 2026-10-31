import { BinaryReader, checkedInteger } from './binary';
import { type LevelDocument, type LevelFiles } from './level';
import { BinaryWriter } from './writer';
import { visibilityCells } from './visibilityImport';
import { portalVisibility } from './portalVisibility';
import { runtimeOrderingMask, type OrderingRoom } from './visibilityGeometry';

/** Rebuild derived visibility, clipping and rotation lists from imported source
 * geometry. Every original zone block is accounted for before replacement.
 * All other level records and graphics streams remain byte-preserved. */
export function buildVisibilityFiles(document: LevelDocument): LevelFiles {
  const zones = [...document.zones].sort((a, b) => a.exitListOffset - b.exitListOffset);
  if (!zones.length) throw new Error('A level needs a room before visibility can be built.');
  const prefixEnd = zones[0]!.exitListOffset;
  for (let i = 0; i < zones.length; i++) {
    const zone = zones[i]!, end = zones[i + 1]?.exitListOffset ?? document.files.data.length;
    if (zone.exitListOffset >= zone.offset || zone.dataEnd !== end)
      throw new Error(`Zone ${zone.id} has additional source data outside the supported layout. The original files have been kept.`);
  }
  const sourceData = new BinaryReader(document.files.data, 'twolev.bin');
  for (const offset of [22, 26, 30, 34, 38, 42, 46, 50])
    if (sourceData.i32(offset) >= prefixEnd) throw new Error('A non-zone table overlaps the zone data being rebuilt.');
  const graphics = document.files.graphics.slice(), graphicView = new DataView(graphics.buffer);
  const sourceGraphics = new BinaryReader(graphics, 'twolev.graph.bin'), table = sourceGraphics.i32(12);
  for (const zone of zones) {
    const stream = sourceGraphics.i32(table + zone.id * 8);
    if (stream <= 0 || sourceGraphics.i16(stream) !== zone.id || zone.fileId !== zone.id)
      throw new Error(`Zone ${zone.id} needs its graphics references normalised before visibility can be rebuilt.`);
  }
  const geometry = visibilityCells(document), compiled = portalVisibility(geometry.points, geometry.rooms, geometry.owners);
  const ordering = new Map(document.zones.map(zone => [zone.id, {
    id: zone.id, outline: zone.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z })),
    neighbours: zone.boundaryIds.map(id => document.lines[id]!.connect),
  } satisfies OrderingRoom]));
  const pointLinks = (document.files.clips.length - document.connectTableOffset) / 4;
  checkedInteger(pointLinks, 0, document.points.length, 'Solid-wall point link count');
  const data = new BinaryWriter(), clips = new BinaryWriter();
  data.write(document.files.data.subarray(0, prefixEnd));
  for (const zone of document.zones) {
    const boundary = data.offset;
    zone.boundaryIds.forEach(id => data.word(id)); data.word(-1);
    zone.wallIds.forEach(id => data.word(id)); data.word(-2);
    const start = data.offset;
    data.write(sourceData.bytes.subarray(zone.offset, zone.offset + 48));
    graphicView.setInt32(16 + zone.id * 4, start);
    data.patchWord(start + 32, checkedInteger(boundary - start, -32768, 32767, 'Boundary list offset'));
    const visible = compiled.visible.get(zone.id)!, oldOrder = new Map(zone.visibility.map((entry, i) => [entry.zone, i]));
    const rank = (id: number) => id === zone.id ? -1 : oldOrder.get(id) ?? zone.visibility.length + id;
    const targets = [...visible].sort((a, b) => rank(a) - rank(b));
    const rotate = new Set<number>();
    for (const target of targets) {
      data.word(target); data.word(target === zone.id ? -1 : 0);
      data.long(runtimeOrderingMask(ordering.get(zone.id)!, ordering.get(target)!, visible));
      const targetZone = document.zones.find(zone => zone.id === target)!;
      for (const graphic of [...targetZone.lower, ...targetZone.upper]) for (const point of graphic.points) rotate.add(point);
      if (target !== zone.id) {
        checkedInteger(clips.offset / 2, 0, 32767, 'Clip word offset');
        const bounds = compiled.clips.get(zone.id)!.get(target)!;
        for (const id of bounds.left) { checkedInteger(id, 0, pointLinks - 1, 'Left clip point'); clips.word(id); rotate.add(id); }
        clips.word(-1);
        for (const id of bounds.right) { checkedInteger(id, 0, pointLinks - 1, 'Right clip point'); clips.word(id); rotate.add(id); }
        clips.word(-2);
      }
    }
    data.long(-1);
    data.patchWord(start + 34, checkedInteger(data.offset - start, -32768, 32767, 'Rotation list offset'));
    [...rotate].sort((a, b) => a - b).forEach(id => data.word(id)); data.word(-1);
  }
  checkedInteger(clips.offset / 2, 0, 32767, 'Total clip word count');
  for (let id = 0; id < pointLinks; id++) { clips.word(compiled.previous[id]!); clips.word(compiled.next[id]!); }
  // Door/lift +8 pointers address LEVELGRAPHICS (Anims.s), which is unchanged
  // apart from ZoneAdds. They must never be relocated as level-data pointers.
  return { data: data.finish(), graphics, clips: clips.finish() };
}
