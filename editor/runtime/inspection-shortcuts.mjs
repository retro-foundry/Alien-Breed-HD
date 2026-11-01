// Shared by the native iframe and React. Forward scalar key intent, never a
// synthesized DOM event or a caller-provided editing command.
const primaryKeys = new Set(['s', 'o', 'z', 'y', 'd', 'c', 'v', 'a']);
const plainKeys = new Set(['delete', 'escape', '/', '?', 'f', 'v', 'h', 'b']);

export function inspectionShortcut(input) {
  if (!input || typeof input !== 'object' || typeof input.key !== 'string' || input.isComposing === true
    || !['ctrlKey', 'metaKey', 'altKey', 'shiftKey', 'repeat'].every(field => typeof input[field] === 'boolean') || input.altKey) return null;
  const key = input.key.toLowerCase(), primary = input.ctrlKey || input.metaKey;
  if (!(primary ? primaryKeys : plainKeys).has(key)) return null;
  return { key: input.key, ctrlKey: input.ctrlKey, metaKey: input.metaKey, altKey: false, shiftKey: input.shiftKey, repeat: input.repeat };
}

// History, saving and navigation operate on the current workspace even while
// its preview rebuilds. Selection commands need the matching captured document.
export function inspectionShortcutScope(input) {
  const shortcut = inspectionShortcut(input);
  if (!shortcut) return null;
  const key = shortcut.key.toLowerCase();
  return (shortcut.ctrlKey || shortcut.metaKey ? ['c', 'v', 'd', 'a'].includes(key) : key === 'delete') ? 'frame' : 'workspace';
}
