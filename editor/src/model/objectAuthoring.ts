import limits from './objectLimits.json' with { type: 'json' };
import { BinaryReader, checkedInteger } from './binary';
import { dataLayout } from './dataBuild';
import { parseLevel, type LevelDocument, type LevelObject } from './level';
import { objectPlayer, placeSourceObject, runtimeObjectPosition } from './sourceObjects';
import type { EditCommand } from './commands';
import { roomSettings } from './roomHeights';

export const objectSlotLimit = limits.objects;
export function objectTemplates(document: LevelDocument): readonly LevelObject[] {
  const reader = new BinaryReader(document.files.data, 'object templates');
  // level.c: original player shots, nasty shots and other-nasty/gib pools each
  // retain PLAYER_SHOT_SLOT_COUNT source records, before runtime expansion.
  const playerShots = reader.i32(34), nastyShots = reader.i32(38), span = limits.sourceShotSlots * 64;
  return document.objects.filter(object => object.type >= 0 && object.type <= 20 && ![2, 5, 11].includes(object.type)
    && objectPlayer(document, object) === undefined
    && !(object.offset >= playerShots && object.offset < playerShots + span)
    && !(object.offset >= nastyShots && object.offset < nastyShots + span * 2));
}

export interface ObjectPlacement { template: number; room: string; x: number; z: number; y: number; upper?: boolean }
export function floorMatchedObjectHeight(document: LevelDocument, template: LevelObject, room: string, upper = template.upper): number {
  const target = document.roomSource.rooms.find(item => item.id === room);
  if (!target) throw new Error('Choose the destination room.');
  const authored = document.sourceObjects?.find(item => item.id === template.id);
  const source = authored?.room ? document.roomSource.rooms.find(item => item.id === authored.room)
    : document.roomSource.rooms.find(item => item.originZone === template.zone);
  const from = source ? roomSettings(document, source) : document.zones.find(zone => zone.id === template.zone);
  if (!from) throw new Error('This template has no floor binding. Turn off floor matching and enter its height.');
  const to = roomSettings(document, target);
  if (upper && !to.upper.length) throw new Error('The destination room has no upper floor. Choose Lower floor or another room.');
  return Math.round((to[upper ? 'upperFloor' : 'floor'] + template.y - from[template.upper ? 'upperFloor' : 'floor']) * 2) / 2;
}
/** Append a real source record/CID, preserving every original slot and reserved
 * pool. Whole-table relocation stays one command; generated room geometry is
 * never written back into the editable source. */
export function createObject(document: LevelDocument, placement: ObjectPlacement, templateSource = document): { id: number; command: EditCommand } {
  if (templateSource.original !== document.original) throw new Error('This object template belongs to another project. Cross-project resource transfer is not supported yet.');
  const template = objectTemplates(templateSource).find(object => object.id === placement.template);
  if (!template) throw new Error('Choose an enemy, pickup or scenery template; player and projectile slots cannot be duplicated.');
  if (document.objects.length >= limits.objects) throw new Error(`The native ${limits.objects}-record object table is full, including reserved shot and avatar slots.`);
  const y = checkedInteger(placement.y * 2, -32768, 32767, 'Object height in half units');
  const position = runtimeObjectPosition(placement);
  const { reader, points } = dataLayout(document), start = reader.i32(30), objectPoints = reader.i16(20);
  checkedInteger(objectPoints, 0, limits.objects - 1, 'New object collision identity');
  checkedInteger(objectPoints + 2, 1, 2048, 'Object point table size including padding');
  // level.c expands shots starting at header[20], treating it as a count.
  // Source files also contain one extra padding point (and sometimes an
  // inactive dummy record referring to it). Move that padding, not a live CID.
  const paddingUsers = document.objects.filter(object => object.cid === objectPoints);
  for (const object of paddingUsers) {
    const authored = document.sourceObjects?.find(item => item.id === object.id);
    if (object.zone >= 0 || authored?.room || objectPlayer(document, object) !== undefined || object.type !== -1)
      throw new Error('The source padding point is used by an active or reserved object. Repair its position identity before adding objects.');
  }
  const id = document.objects.length, terminator = points - 2;
  checkedInteger(document.files.data.length + 72, 1, 16 * 1024 * 1024, 'Expanded source data size');
  const data = new Uint8Array(document.files.data.length + 72), view = new DataView(data.buffer);
  const padding = start - 8;
  data.set(reader.bytes.subarray(0, padding));
  view.setInt32(padding, position.x * 65536); view.setInt32(padding + 4, position.z * 65536);
  data.set(reader.bytes.subarray(padding, start), start);
  data.set(reader.bytes.subarray(start, terminator), start + 8);
  for (const object of paddingUsers) view.setInt16(object.offset + 8, objectPoints + 1);
  const record = terminator + 8;
  data.set(templateSource.files.data.subarray(template.offset, template.offset + 64), record);
  view.setInt16(record, objectPoints); view.setInt16(record + 4, y);
  data[record + 63] = (placement.upper ?? template.upper) ? 1 : 0;
  view.setInt16(record + 12, -1); view.setInt16(record + 26, -1);
  data.set(reader.bytes.subarray(terminator), record + 64);
  view.setInt16(20, objectPoints + 1);
  for (const field of [30, 34, 38, 46, 50]) view.setInt32(field, reader.i32(field) + 8);
  for (const field of [22, 26]) view.setInt32(field, reader.i32(field) + 72);
  const graphics = document.files.graphics.slice(), graphReader = new BinaryReader(graphics, 'object relocation graphics'), graphView = new DataView(graphics.buffer);
  const slots = (graphReader.i32(12) - 16) / 4;
  for (let zone = 0; zone < slots; zone++) {
    const at = 16 + zone * 4, offset = graphReader.i32(at);
    if (offset >= 0) graphView.setInt32(at, offset + 72);
  }
  const files = { data, graphics, clips: document.files.clips };
  const rewritten = { ...document, ...parseLevel(document.name, files, document.encoded, document.original,
    document.playtestSlot, document.exitZone, document.roomSource, document.drawGroups) };
  const sourceObjects = placeSourceObject(rewritten, id, placement.room, position.x, position.z);
  return { id, command: { label: `Add object from template ${template.id}`, patches: [], files: { before: document.files, after: files },
    sourceObjects: { before: document.sourceObjects, after: sourceObjects } } };
}

/** Defs.i / level.c's complete 64-byte source object record. A type owns
 * artwork, dimensions, health, AI and pickup parameters; changing byte 16
 * alone leaves the old type's incompatible state behind. Keep the target's
 * collision identity, height, logical/graphic room and storey binding. */
export function replaceObjectSettings(document: LevelDocument, id: number, templateId: number): EditCommand {
  const templates = objectTemplates(document);
  const object = templates.find(item => item.id === id), template = templates.find(item => item.id === templateId);
  if (!object || !template) throw new Error('Choose an editable enemy, pickup or scenery object and a supported settings template. Players and reserved projectile slots cannot change type.');
  if (document.sourceObjects?.find(item => item.id === id)?.room === null || object.zone < 0 && !document.sourceObjects?.some(item => item.id === id && item.room))
    throw new Error('Restore this removed object before changing its type.');
  const before = document.files.data.slice(object.offset, object.offset + 64);
  const after = document.files.data.slice(template.offset, template.offset + 64);
  for (const [start, end] of [[0, 2], [4, 6], [12, 14], [26, 28], [63, 64]]) after.set(before.subarray(start, end), start);
  return { label: `Change object ${id} settings from object ${templateId}`, patches: [{ file: 'data', offset: object.offset, before, after }] };
}
