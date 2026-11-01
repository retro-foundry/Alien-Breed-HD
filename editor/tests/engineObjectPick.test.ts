import { expect, it } from 'vitest';
import { resolveEngineObject } from '../src/model/engineObjectPick';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { createObject, floorMatchedObjectHeight, objectTemplates } from '../src/model/objectAuthoring';
import { applyCommand, removeObject } from '../src/model/commands';
import { objectPlayer } from '../src/model/sourceObjects';
import { roomHeightFixture } from './roomHeightFixture';

it('maps a compiled copied actor back to its independent source record and CID', () => {
  const { document, from } = roomHeightFixture(), template = objectTemplates(document).find(object => !object.upper)!;
  const created = createObject(document, { template: template.id, room: from.id, x: 20128, z: 20180,
    y: floorMatchedObjectHeight(document, template, from.id, false), upper: false });
  const source = applyCommand(document, created.command), built = buildPlayableLevel(source, -1), native = built.document.objects.find(object => object.id === created.id)!;
  expect(resolveEngineObject(built, { id: created.id, cid: native.cid })).toBe(created.id);
  expect(native.zone).not.toBe(source.objects.find(object => object.id === created.id)!.zone);
  expect(resolveEngineObject(built, { id: template.id, cid: native.cid })).toBeUndefined();
  const deleted = buildPlayableLevel(applyCommand(source, removeObject(source, created.id)), -1);
  expect(resolveEngineObject(deleted, { id: created.id, cid: native.cid })).toBeUndefined();
});

it('rejects avatars, malformed picks, missing slots and mismatched native point identities', () => {
  const { document } = roomHeightFixture(), built = buildPlayableLevel(document, -1);
  const avatar = document.objects.find(object => objectPlayer(document, object) !== undefined)!;
  expect(resolveEngineObject(built, { id: avatar.id, cid: avatar.cid })).toBeUndefined();
  const object = objectTemplates(document).find(object => object.zone >= 0)!;
  expect(resolveEngineObject(built, { id: object.id, cid: object.cid })).toBe(object.id);
  for (const pick of [null, undefined, { id: -1, cid: 0 }, { id: .5, cid: 0 }, { id: object.id, cid: -1 },
    { id: object.id, cid: .5 }, { id: 10000, cid: 10000 }, { id: object.id, cid: object.cid + 1000 }])
    expect(resolveEngineObject(built, pick)).toBeUndefined();
});
