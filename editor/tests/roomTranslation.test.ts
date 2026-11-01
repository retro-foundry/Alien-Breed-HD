import { expect, it } from 'vitest';
import { outlineTranslation } from '../src/model/roomTranslation';

it('recognizes exact translation through inserted collinear portal points and either drawing direction', () => {
  const original = [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 128 }, { x: 0, z: 128 }];
  const target = [{ x: 64, z: 32 }, { x: 192, z: 32 }, { x: 192, z: 64 }, { x: 192, z: 128 }, { x: 192, z: 160 }, { x: 64, z: 160 }];
  const before = structuredClone(target);
  expect(outlineTranslation(original, target)).toEqual({ x: 64, z: 32 });
  expect(outlineTranslation(original.toReversed(), [...target.slice(2), ...target.slice(0, 2)])).toEqual({ x: 64, z: 32 });
  expect(target).toEqual(before);
});

it('does not classify changed shape or size as a translation', () => {
  const original = [{ x: 0, z: 0 }, { x: 128, z: 0 }, { x: 128, z: 128 }, { x: 0, z: 128 }];
  expect(outlineTranslation(original, original.map(point => ({ x: point.x * 2, z: point.z })))).toBeUndefined();
  expect(outlineTranslation(original, original.map((point, index) => index === 1 ? { ...point, x: point.x + 16 } : point))).toBeUndefined();
});
