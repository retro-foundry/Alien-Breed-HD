import { applyCommand, type EditCommand } from './commands';
import { createObject, objectSlotLimit, objectTemplates } from './objectAuthoring';
import { editableObjects, objectIsActive, objectDestinationRoom, runtimeObjectPosition } from './sourceObjects';
import type { LevelDocument } from './level';

/** Documents are immutable command snapshots: retain copied settings even if
 * the original objects are subsequently edited, deleted or undone. */
export interface ObjectClipboard { readonly source: LevelDocument; readonly ids: readonly number[] }

export function copyObjects(source: LevelDocument, ids: readonly number[]): ObjectClipboard {
  if (!ids.length || new Set(ids).size !== ids.length) throw new Error('Select one or more different objects to copy.');
  const templates = new Set(objectTemplates(source).map(object => object.id));
  const objects = editableObjects(source);
  for (const id of ids) if (!templates.has(id) || !objects.some(object => object.id === id && objectIsActive(object)))
    throw new Error(`Object ${id} cannot be copied. Select active enemies, pickups or scenery; player and projectile slots cannot be copied.`);
  return Object.freeze({ source, ids: Object.freeze([...ids]) });
}

export function pasteObjects(document: LevelDocument, clipboard: ObjectClipboard, x = 0, z = 0): { ids: number[]; command: EditCommand } {
  if (clipboard.source.original !== document.original) throw new Error('Copied objects belong to another project. Copy objects in this workspace before pasting. Cross-project resource transfer is not supported yet.');
  copyObjects(clipboard.source, clipboard.ids);
  if (document.objects.length + clipboard.ids.length > objectSlotLimit) throw new Error(`Pasting would exceed the native ${objectSlotLimit}-record capacity. No objects were pasted.`);
  const objects = editableObjects(clipboard.source);
  let working = document;
  const ids: number[] = [];
  for (const id of clipboard.ids) {
    const object = objects.find(object => object.id === id)!;
    const position = runtimeObjectPosition({ x: object.x + x, z: object.z + z });
    const room = objectDestinationRoom(document, object, position);
    if (!room) throw new Error(`Object ${id}'s pasted position is outside rooms supporting its floor. No objects were pasted.`);
    const proposal = createObject(working, { template: id, room: room.id, ...position, y: object.y, upper: object.upper }, clipboard.source);
    working = applyCommand(working, proposal.command); ids.push(proposal.id);
  }
  return { ids, command: { label: `Paste ${ids.length} objects`, patches: [], files: { before: document.files, after: working.files },
    sourceObjects: { before: document.sourceObjects, after: working.sourceObjects } } };
}
