import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { zipSync, unzipSync, strToU8 } from 'fflate';
import { openLevel } from '../../src/model/level';
import { encodeProject, decodeProject } from '../../src/model/project';
import { beginRoomDraft, setDraftCorners, commitRoomDraft } from '../../src/model/roomSource';
import { execute, editRoomSource } from '../../src/model/commands';
import { originalFiles } from '../helpers';

test('incomplete room appearance survives saves/recovery and stops preview/export with an actionable build error', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept());
  const original = openLevel('Authored reactor', originalFiles(1));
  const change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]));
  const edited = execute({ document: original, past: [], future: [] }, editRoomSource(change)).document;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'authored.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(edited)) });
  await expect(page.locator('.project-picker strong')).toHaveText('Authored reactor');
  await expect(page.getByText(/This project has editable source changes/)).toBeVisible();
  const backup = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('source-backup.ab3d'); await (await backup).saveAs(path);
  expect(decodeProject(new Uint8Array(await readFile(path))).roomSource).toEqual(edited.roomSource);
  await page.getByRole('button', { name: 'Export level', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('needs an appearance');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('alert')).toContainText('needs an appearance', { timeout: 60000 });
  await expect(preview).toHaveAttribute('data-level-slot', '-1');
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload(); await expect(page.locator('.project-picker strong')).toHaveText('Authored reactor');
  await expect(page.getByText(/This project has editable source changes/)).toBeVisible();
  const broken = unzipSync(encodeProject(edited)); broken['room-source.json'] = strToU8('{');
  await page.locator('input[type=file]').setInputFiles({ name: 'damaged-source.ab3d', mimeType: 'application/zip', buffer: Buffer.from(zipSync(broken)) });
  await expect(page.getByRole('alert')).toContainText('room source is not valid JSON');
  await expect(page.locator('.project-picker strong')).toHaveText('Authored reactor');
  const recovered = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const recoveredPath = testInfo.outputPath('recovered-source.ab3d'); await (await recovered).saveAs(recoveredPath);
  expect(decodeProject(new Uint8Array(await readFile(recoveredPath))).roomSource).toEqual(edited.roomSource);
});
