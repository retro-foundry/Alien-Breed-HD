import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { sourceExitFixture } from '../sourceExitFixture';
import { applyCommand, setExitZone, setSourceStart } from '../../src/model/commands';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { decodeProject, encodeProject, openArchive } from '../../src/model/project';

test('selects a new concave room as exit and completes from each native collision cell', async ({ page }, info) => {
  test.setTimeout(120000);
  page.on('dialog', dialog => void dialog.accept());
  const fixture = sourceExitFixture(), source = applyCommand(fixture.document, setExitZone(fixture.document, -1));
  const built = buildPlayableLevel(source, -1), cells = built.topology!.cells.filter(cell => cell.sourceId === fixture.from.id);
  expect(cells.length).toBeGreaterThan(1);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const exit = page.getByRole('combobox', { name: 'Level exit', exact: true });
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  for (const [index, cell] of cells.entries()) {
    const points = cell.vertices.map(id => built.topology!.points[id]!);
    const x = Math.round(points.reduce((sum, p) => sum + p.x, 0) / points.length), z = Math.round(points.reduce((sum, p) => sum + p.z, 0) / points.length);
    const document = applyCommand(source, setSourceStart(source, 1, fixture.from.id, x, z));
    await page.locator('input[type=file]').setInputFiles({ name: `exit-cell-${index}.ab3d`, mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
    await exit.selectOption(`source:${fixture.from.id}`);
    await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(exit).toHaveValue('-1');
    await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(exit).toHaveValue(`source:${fixture.from.id}`);
    await page.getByRole('button', { name: 'Engine view', exact: true }).click();
    await expect(preview).toHaveAttribute('data-exit-zone', String(cells[0]!.id), { timeout: 60000 });
    await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
    await expect(preview.getByRole('status')).toHaveText('Playtest ended. Build and play again to restart.');
    if (index === 0) {
      const saving = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
      const path = info.outputPath('source-exit.ab3d'); await (await saving).saveAs(path);
      expect(decodeProject(new Uint8Array(await readFile(path))).roomSource.exitRoom).toBe(fixture.from.id);
      const exporting = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
      const gamePath = info.outputPath('source-exit-game.zip'); await (await exporting).saveAs(gamePath);
      const game = openArchive(new Uint8Array(await readFile(gamePath)), 'exit.zip')[0]!;
      for (const cell of cells) expect(game.drawGroups![cell.id]).toBe(game.exitZone);
    }
    await exit.selectOption('-1'); await expect(preview).toHaveAttribute('data-exit-zone', '-1');
    await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
    await expect(preview.getByRole('status')).toHaveText('Playtest running');
    await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  }
});
