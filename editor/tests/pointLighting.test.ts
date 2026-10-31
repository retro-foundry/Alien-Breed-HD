import { expect, it } from 'vitest';
import { importedDataTables } from '../src/model/dataBuild';
import { openLevel } from '../src/model/level';
import { buildPointLighting, interpolatePointBrightness, type PointLightingBinding } from '../src/model/pointLighting';
import { originalFiles } from './helpers';

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('preserves every original level %i point brightness word', level => {
  const document = openLevel('original', originalFiles(level));
  expect(buildPointLighting(document, document.points, new Map())).toEqual(importedDataTables(document).points);
});

it('samples signed base brightness separately from packed animation controls', () => {
  expect(interpolatePointBrightness(-15, 14, 1, 2)).toBe(0);
  expect(interpolatePointBrightness(-15, -2, 1, 2)).toBe(-8);
  expect(interpolatePointBrightness(0x7102, 0x7110, 1, 2)).toBe(0x7109);
  expect(interpolatePointBrightness(-3838, -3824, 1, 2)).toBe(-3831); // 0xf102 / 0xf110 -> 0xf109
  expect(interpolatePointBrightness(0x7102, 0x7210, 0, 2)).toBe(0x7102);
  expect(interpolatePointBrightness(0x7102, 0x7210, 2, 2)).toBe(0x7210);
  expect(() => interpolatePointBrightness(0x7102, 0x7210, 1, 2)).toThrow(/incompatible/);
  expect(() => interpolatePointBrightness(0x7102, -1, 1, 2)).toThrow(/incompatible/);
  expect(() => interpolatePointBrightness(0, 1, 1, 0)).toThrow(/denominator/);
});

it('copies, samples and explicitly assigns new point lighting without taking source coordinates', () => {
  const document = openLevel('generated', originalFiles(1)), count = document.points.length;
  const points = [...document.points, { x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }];
  const before = document.files.data.slice(), source = importedDataTables(document).points;
  const bindings = new Map<number, PointLightingBinding>([[count, { kind: 'copy', source: 0 }],
    [count + 1, { kind: 'segment', from: 0, to: 1, numerator: 1, denominator: 2 }],
    [count + 2, { kind: 'value', lowerBrightness: -10, upperBrightness: 0x7102 }]]);
  const built = buildPointLighting(document, points, bindings);
  expect(built[count]).toEqual({ ...source[0]!, x: 20000, z: 20000 });
  expect(built[count + 1]!.lowerBrightness).toBe(interpolatePointBrightness(source[0]!.lowerBrightness, source[1]!.lowerBrightness, 1, 2));
  expect(built[count + 2]).toEqual({ x: 20128, z: 20128, lowerBrightness: -10, upperBrightness: 0x7102 });
  expect(() => buildPointLighting(document, points, new Map())).toThrow(/no lighting assignment/);
  expect(() => buildPointLighting(document, [{ x: 20000, z: 20000 }], new Map())).toThrow(/no lighting assignment/);
  expect(() => buildPointLighting(document, document.points, new Map([[count, { kind: 'copy', source: 0 }]]))).toThrow(/target/);
  expect(document.files.data).toEqual(before);
});
