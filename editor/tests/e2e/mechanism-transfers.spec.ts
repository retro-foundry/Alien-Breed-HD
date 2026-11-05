import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { mechanismTransferFixture } from '../mechanismTransferFixture';
import { applyCommand, editRoomSource } from '../../src/model/commands';
import { setCopiedMechanismRules } from '../../src/model/movingControllerCopies';
import { captureMechanismSettings } from '../../src/model/mechanismSettingsTransfer';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { engineMechanismState, engineMechanismTrace, engineMechanismControlState, enginePlayerState } from '../engineCapture';

for (const kind of ['door', 'lift'] as const) {
  test(`copies selected ${kind} settings across all controller types and runs the transferred native rules`, async ({ page }, info) => {
    test.setTimeout(90000);
    const fixture = mechanismTransferFixture(kind), copy = fixture.copied.copy;
    const document = applyCommand(fixture.document, editRoomSource(setCopiedMechanismRules(fixture.document, copy.room, kind, 1, kind === 'door' ? 4 : 2, kind === 'door' ? 1 : 3)));
    const source = captureMechanismSettings(document, { type: 'authored', kind, room: fixture.room.id });
    await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
    await page.locator('input[type=file]').setInputFiles({ name: 'mechanism-settings-transfer.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
    await expect(page.locator('.project-picker strong')).toHaveText(document.name);
    await page.getByRole('button', { name: new RegExp(`^Room ${copy.room.replace(/^room-/, '')}(?:\\s|$)`) }).click();
    const copiedSection = page.getByRole('region', { name: `copied ${kind} ${copy.room}`, exact: true });
    const transfer = copiedSection.getByRole('group', { name: `Copy ${kind} settings`, exact: true }); await transfer.locator('summary').click();
    await expect(transfer.getByRole('checkbox', { name: 'Trigger rules', exact: true })).toBeChecked();
    await expect(transfer.getByRole('checkbox', { name: 'Destinations', exact: true })).not.toBeChecked();
    await expect(transfer.getByRole('checkbox', { name: 'Key and switch requirements', exact: true })).not.toBeChecked();
    await transfer.getByRole('combobox').selectOption(`authored:${kind}:${fixture.room.id}`);
    await transfer.getByRole('checkbox', { name: 'Destinations', exact: true }).check();
    await transfer.getByRole('combobox').press('Escape');
    await expect(transfer.getByRole('combobox')).toHaveValue(''); await expect(transfer.getByRole('checkbox', { name: 'Destinations', exact: true })).not.toBeChecked();
    await expect(page.locator('.dirty-dot')).toHaveCount(0);
    await transfer.getByRole('combobox').selectOption(`authored:${kind}:${fixture.room.id}`);
    await transfer.getByRole('button', { name: 'Copy selected settings', exact: true }).click();
    const closing = copiedSection.getByRole('combobox', { name: `copied ${kind} ${copy.room} closing rule`, exact: true });
    await expect(closing).toHaveValue(String(source.closeRule)); await expect(copiedSection.getByRole('checkbox', { name: 'Key 1', exact: true })).toBeChecked();
    await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(closing).toHaveValue(kind === 'door' ? '1' : '3');
    await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(closing).toHaveValue(String(source.closeRule));
    await transfer.getByRole('combobox').selectOption(`authored:${kind}:${fixture.room.id}`);
    await transfer.getByRole('checkbox', { name: 'Key and switch requirements', exact: true }).check();
    await transfer.getByRole('button', { name: 'Copy selected settings', exact: true }).click();
    await expect(copiedSection.getByRole('checkbox', { name: 'Key 1', exact: true })).not.toBeChecked();
    await page.getByRole('button', { name: new RegExp(`^Zone ${String(fixture.copied.mechanism.zone).padStart(3, '0')}(?:\\s|$)`) }).click();
    const nativeTransfer = page.getByRole('group', { name: `Copy ${kind} settings`, exact: true }); await nativeTransfer.locator('summary').click();
    await nativeTransfer.getByRole('combobox').selectOption(`copy:${kind}:${copy.room}`);
    await nativeTransfer.getByRole('button', { name: 'Copy selected settings', exact: true }).click();
    await page.getByRole('button', { name: new RegExp(`^Room ${fixture.room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
    const authored = page.getByRole('region', { name: 'Create door or lift', exact: true }), form = authored.getByRole('form', { name: 'Authored mechanism settings' });
    await form.getByRole('combobox', { name: 'New mechanism opening rule', exact: true }).selectOption(kind === 'door' ? '5' : '3');
    await form.getByRole('button', { name: 'Apply mechanism', exact: true }).click();
    const authoredTransfer = authored.getByRole('group', { name: `Copy ${kind} settings`, exact: true }); await authoredTransfer.locator('summary').click();
    await authoredTransfer.getByRole('combobox').selectOption(`copy:${kind}:${copy.room}`);
    await authoredTransfer.getByRole('button', { name: 'Copy selected settings', exact: true }).click();
    await expect(form.getByRole('combobox', { name: 'New mechanism opening rule', exact: true })).toHaveValue(String(source.openRule));
    const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
    const path = info.outputPath(`copied-${kind}-settings.ab3d`); await (await download).saveAs(path); const saved = decodeProject(new Uint8Array(await readFile(path)));
    for (const owner of [{ type: 'native' as const, kind, id: 0 }, { type: 'copy' as const, kind, room: copy.room }, { type: 'authored' as const, kind, room: fixture.room.id }])
      expect(captureMechanismSettings(saved, owner)).toMatchObject({ openRule: source.openRule, closeRule: source.closeRule, conditions: 0 });
    const offset = document.mechanisms[0]!.offset;
    expect(saved.files.graphics.slice(offset + 4, offset + 14)).toEqual(document.files.graphics.slice(offset + 4, offset + 14));
    await page.getByRole('button', { name: new RegExp(`^Room ${copy.room.replace(/^room-/, '')}(?:\\s|$)`) }).click();
    await page.getByRole('button', { name: 'Engine view', exact: true }).click(); const preview = page.getByRole('region', { name: '3D game engine preview' });
    await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
    for (const index of [0, 1, 2]) {
      const state = await engineMechanismState(page, kind, index);
      expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0 });
      expect(await engineMechanismControlState(page, kind, index)).toEqual({ velocity: 0, timer: kind === 'door' ? 0 : null });
    }
    if (kind === 'door') {
      const built = buildPlayableLevel(saved, -1), controlled = (await engineMechanismState(page, kind, 1)).zone;
      const neighbours = new Set(built.topology!.cells.find(cell => cell.id === controlled)!.neighbours.filter(id => id >= 0));
      await expect.poll(async () => neighbours.has((await enginePlayerState(page)).zone)).toBe(true);
      await expect(page.locator('.inspector h2')).toHaveText(`Room ${copy.room.replace(/^room-/, '')}`);
    }
    await expect.poll(async () => Number(await preview.getAttribute('data-rendered-pixels'))).toBeGreaterThan(1000);
    await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
    const trace = await engineMechanismTrace(page, kind, 1, false);
    expect(trace.some(state => state.current !== fixture.copied.mechanism.current)).toBe(true);
    for (const state of trace) expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0 });
    await expect(preview.getByRole('alert')).toHaveCount(0);
  });

  test(`moves an authored ${kind} to another room with selection, Undo, saved source and native movement`, async ({ page }, info) => {
    test.setTimeout(90000);
    const fixture = mechanismTransferFixture(kind), document = fixture.document;
    await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
    await page.locator('input[type=file]').setInputFiles({ name: 'mechanism-move.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
    await expect(page.locator('.project-picker strong')).toHaveText(document.name);
    await page.getByRole('button', { name: new RegExp(`^Room ${fixture.room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
    const moving = page.getByRole('group', { name: `Move authored ${kind}`, exact: true }); await moving.locator('summary').click();
    await moving.getByRole('combobox').selectOption(fixture.copied.copy.room); await moving.getByRole('button', { name: 'Move mechanism', exact: true }).click();
    await expect(moving.getByRole('alert')).toContainText('copied controller');
    await moving.getByRole('button', { name: 'Cancel move draft', exact: true }).click(); await expect(moving.getByRole('combobox')).toHaveValue('');
    await expect(moving.getByRole('alert')).toHaveCount(0); await expect(page.locator('.dirty-dot')).toHaveCount(0);
    await moving.getByRole('combobox').selectOption(fixture.neighbour.id); await moving.getByRole('button', { name: 'Move mechanism', exact: true }).click();
    await expect(page.locator('.inspector h2')).toHaveText(`Room ${fixture.neighbour.id.replace(/^room-/, '')}`);
    await expect(moving).toBeVisible();
    await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(moving).toHaveCount(0);
    await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(moving).toBeVisible();
    const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
    const path = info.outputPath(`moved-${kind}.ab3d`); await (await download).saveAs(path); const saved = decodeProject(new Uint8Array(await readFile(path)));
    expect(saved.files).toEqual(document.files); expect(saved.roomSource.rooms.find(room => room.id === fixture.room.id)!.moving).toBeUndefined();
    expect(saved.roomSource.rooms.find(room => room.id === fixture.neighbour.id)!.moving).toEqual(document.roomSource.rooms.find(room => room.id === fixture.room.id)!.moving);
    const built = buildPlayableLevel(saved, -1), mechanism = built.document.mechanisms.filter(item => item.kind === kind).at(-1)!;
    expect(built.topology!.cells.find(cell => cell.id === mechanism.zone)!.sourceId).toBe(fixture.neighbour.id);
    await page.getByRole('button', { name: 'Engine view', exact: true }).click(); const preview = page.getByRole('region', { name: '3D game engine preview' });
    await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
    expect(await engineMechanismState(page, kind, mechanism.id)).toMatchObject({ zone: mechanism.zone, current: mechanism.current, plane: mechanism.current, unsynchronizedWalls: 0 });
    await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
    const trace = await engineMechanismTrace(page, kind, mechanism.id, true);
    expect(new Set(trace.map(state => state.current)).size).toBeGreaterThan(1);
    for (const state of trace) expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0 });
    await expect(preview.getByRole('alert')).toHaveCount(0);
  });
}
