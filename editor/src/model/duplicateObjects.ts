import { applyCommand, type EditCommand } from './commands';
import { createObject, objectTemplates, objectSlotLimit } from './objectAuthoring';
import { selectedObjects } from './objectGroups';
import { objectDestinationRoom, runtimeObjectPosition } from './sourceObjects';
import type { LevelDocument } from './level';

/** Build every record in an isolated document, then expose one relocation
 * command. Capacity or placement errors cannot append a partial selection. */
export function duplicateObjects(document: LevelDocument, ids: readonly number[], x: number, z: number): { ids: number[]; command: EditCommand } {
  const objects = selectedObjects(document, ids), templates = new Set(objectTemplates(document).map(object => object.id));
  if (document.objects.length + objects.length > objectSlotLimit) throw new Error(`This selection exceeds the native ${objectSlotLimit}-record capacity. No objects were duplicated.`);
  for (const object of objects) if (!templates.has(object.id)) throw new Error(`Object ${object.id} cannot be duplicated. Select supported enemies, pickups or scenery; no copies were created.`);
  let working = document; const created: number[] = [];
  for (const object of objects) {
    const position = runtimeObjectPosition({ x: object.x + x, z: object.z + z });
    const room = objectDestinationRoom(document, object, position);
    if (!room) throw new Error(`Object ${object.id}'s copy would be outside rooms supporting its floor. No objects were duplicated.`);
    const proposal = createObject(working, { template: object.id, room: room.id, ...position, y: object.y, upper: object.upper });
    working = applyCommand(working, proposal.command); created.push(proposal.id);
  }
  return { ids: created, command: { label: `Duplicate ${objects.length} objects`, patches: [],
    files: { before: document.files, after: working.files }, sourceObjects: { before: document.sourceObjects, after: working.sourceObjects } } };
}
