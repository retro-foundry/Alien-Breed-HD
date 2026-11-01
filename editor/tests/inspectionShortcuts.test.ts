import { expect, it } from 'vitest';
import { inspectionShortcut, inspectionShortcutScope } from '../runtime/inspection-shortcuts.mjs';

const key = { key: 'z', ctrlKey: true, metaKey: false, shiftKey: false, altKey: false, repeat: false };
it('shares conventional primary and plain inspection shortcuts while retaining modifiers and repeat', () => {
  for (const letter of ['s', 'o', 'z', 'y', 'd', 'c', 'v', 'a']) for (const meta of [false, true]) {
    const input = { ...key, key: letter.toUpperCase(), ctrlKey: !meta, metaKey: meta, shiftKey: true, repeat: true };
    expect(inspectionShortcut(input)).toEqual(input);
  }
  for (const value of ['Delete', 'Escape', '/', '?', 'f', 'F', 'v', 'h', 'b'])
    expect(inspectionShortcut({ ...key, key: value, ctrlKey: false })?.key).toBe(value);
});
it('keeps native view/play keys and unsupported or composed modifier combinations out of the bridge', () => {
  for (const value of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Enter', 'Tab', ' ', 'w', 's', 'c', '1'])
    expect(inspectionShortcut({ ...key, key: value, ctrlKey: false })).toBeNull();
  for (const input of [{ ...key, key: 'x' }, { ...key, altKey: true }, { ...key, isComposing: true },
    null, undefined, 'Delete', {}, { ...key, ctrlKey: 'true' }, { ...key, repeat: 0 }, { ...key, key: 1 }])
    expect(inspectionShortcut(input)).toBeNull();
});
it('forwards only scalar key intent, discarding event targets and caller-supplied actions', () => {
  const input = { ...key, command: 'remove-everything', target: { value: 'secret field' }, extra: [1, 2] };
  expect(inspectionShortcut(input)).toEqual(key);
});
it('keeps workspace history/save/navigation available during rebuilds while selection commands require the current frame', () => {
  for (const value of ['z', 'y', 's', 'o']) expect(inspectionShortcutScope({ ...key, key: value })).toBe('workspace');
  for (const value of ['c', 'v', 'd', 'a']) expect(inspectionShortcutScope({ ...key, key: value })).toBe('frame');
  expect(inspectionShortcutScope({ ...key, key: 'Delete', ctrlKey: false })).toBe('frame');
  for (const value of ['Escape', '/', '?', 'f', 'b', 'v', 'h']) expect(inspectionShortcutScope({ ...key, key: value, ctrlKey: false })).toBe('workspace');
  expect(inspectionShortcutScope({ ...key, key: 'w', ctrlKey: false })).toBeNull();
});
