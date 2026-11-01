import type { PlayableLevel } from './playableBuild';
import { editableObjects, objectIsActive, objectPlayer } from './sourceObjects';

export interface NativeObjectPick { id: number; cid: number }

/** Original record slots survive compilation and native shot-pool expansion.
 * Require both record and point identities; never reinterpret an unknown runtime
 * record as a different authored object just because its position is nearby. */
export function resolveEngineObject(built: PlayableLevel, pick: NativeObjectPick | null | undefined): number | undefined {
  if (!pick || !Number.isInteger(pick.id) || !Number.isInteger(pick.cid) || pick.id < 0 || pick.cid < 0) return;
  const native = built.document.objects.find(object => object.id === pick.id && object.cid === pick.cid);
  const object = editableObjects(built.source).find(object => object.id === pick.id && object.cid === pick.cid);
  if (!native || native.zone < 0 || !object || !objectIsActive(object) || objectPlayer(built.source, object) !== undefined) return;
  return object.id;
}
