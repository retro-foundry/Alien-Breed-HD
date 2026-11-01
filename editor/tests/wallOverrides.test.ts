import { expect, it } from 'vitest';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { applyCommand, execute, numberPatch, redo, replaceWallAsset, undo } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeGameLevel, encodeProject, openArchive, openProjectCandidate } from '../src/model/project';
import { restoreWallOverrides, wallOverrideFiles } from '../src/model/wallOverrides';
import { wallOverrideFixture } from './wallOverrideFixture';

it('saves native assets separately from original levels, builds and exports them, and restores source from game and project archives', () => {
  const { document, sheet, replacement } = wallOverrideFixture();
  const changed = applyCommand(document, replaceWallAsset(document, sheet.id, replacement));
  const project = encodeProject(changed), entries = unzipSync(project);
  expect(JSON.parse(strFromU8(entries['alien-breed-project.json']!))).toMatchObject({ version: 9, wallAssets: [sheet.id] });
  expect(entries[`assets/walls/${sheet.id}.wad`]).toEqual(replacement);
  const saved = decodeProject(project); expect(saved.wallOverrides).toEqual([{ texture: sheet.id, bytes: replacement }]);
  expect(saved.files).toEqual(document.files);
  const built = buildPlayableLevel(saved, -1), game = encodeGameLevel(built.document), gameFiles = unzipSync(game);
  expect(gameFiles[`includes/walls/${sheet.name}`]).toEqual(replacement);
  expect(strFromU8(gameFiles['INSTALL.txt']!)).toContain('affect every level');
  const imported = openProjectCandidate(openArchive(game, 'game.zip')[0]!);
  expect(imported.wallOverrides).toEqual(saved.wallOverrides);
  expect(wallOverrideFiles(saved.wallOverrides)).toEqual({ [`includes/walls/${sheet.name}`]: replacement });
});

it('keeps asset edits immutable, stale-checked and undoable across ordinary byte edits and restore-original actions', () => {
  const { document, sheet, replacement } = wallOverrideFixture(), source = replacement.slice();
  const command = replaceWallAsset(document, sheet.id, source);
  let history = execute({ document, past: [], future: [] }, command); source.fill(0);
  expect(history.document.wallOverrides![0]!.bytes).toEqual(replacement);
  expect(() => applyCommand(history.document, command)).toThrow(/wall assets changed/);
  const offset = history.document.zones[0]!.offset + 26;
  history = execute(history, { label: 'Ordinary edit', patches: [numberPatch(history.document, 'data', offset, 1, 2)] });
  expect(history.document.wallOverrides![0]!.bytes).toEqual(replacement);
  history = execute(history, replaceWallAsset(history.document, sheet.id, null));
  expect(history.document.wallOverrides).toBeUndefined();
  history = undo(history); expect(history.document.wallOverrides![0]!.bytes).toEqual(replacement);
  history = undo(undo(history)); expect(history.document.wallOverrides).toBeUndefined();
  history = redo(history); expect(history.document.wallOverrides![0]!.bytes).toEqual(replacement);
  expect(execute(history, replaceWallAsset(history.document, sheet.id, replacement))).toBe(history);
});

it('rejects unsupported slots, truncated assets, duplicate slots and schema downgrades without silently discarding resources', () => {
  const { document, sheet, replacement } = wallOverrideFixture();
  for (const texture of [-1, 14, 0.5]) expect(() => replaceWallAsset(document, texture, null)).toThrow();
  expect(() => replaceWallAsset(document, sheet.id, replacement.slice(1))).toThrow(/native sheet size/);
  expect(() => restoreWallOverrides([{ texture: sheet.id, bytes: replacement }, { texture: sheet.id, bytes: replacement }])).toThrow(/duplicate/);
  const packedMagic = replacement.slice(); packedMagic.set([61, 83, 66, 61]);
  expect(() => restoreWallOverrides([{ texture: sheet.id, bytes: packedMagic }])).toThrow(/decoded/);
  const entries = unzipSync(encodeProject(applyCommand(document, replaceWallAsset(document, sheet.id, replacement))));
  const manifest = JSON.parse(strFromU8(entries['alien-breed-project.json']!));
  entries['alien-breed-project.json'] = strToU8(JSON.stringify({ ...manifest, version: 8 }));
  expect(() => decodeProject(zipSync(entries))).toThrow(/version 9/);
  entries['alien-breed-project.json'] = strToU8(JSON.stringify(manifest));
  const appearances = entries['room-appearances.json']!; delete entries['room-appearances.json'];
  expect(() => decodeProject(zipSync(entries))).toThrow(/editable source file list/);
  entries['room-appearances.json'] = appearances;
  delete entries[`assets/walls/${sheet.id}.wad`];
  expect(() => decodeProject(zipSync(entries))).toThrow(/wall asset list/);
  const grouped = { ...document, drawGroups: document.zones.map(zone => zone.id) };
  const withGroups = unzipSync(encodeProject(applyCommand(grouped, replaceWallAsset(grouped, sheet.id, replacement))));
  expect(decodeProject(zipSync(withGroups)).drawGroups).toEqual(grouped.drawGroups);
  delete withGroups['level/twolev.cfg'];
  expect(() => decodeProject(zipSync(withGroups))).toThrow(/editable source file list/);
});
