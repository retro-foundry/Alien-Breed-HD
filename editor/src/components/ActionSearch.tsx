import { useEffect, useRef, useState } from 'react';
import { Icon } from './Icon';

export interface EditorAction {
  id: string;
  label: string;
  description: string;
  keywords?: string;
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
}

export function ActionSearch({ actions, onChoose, onClose }: {
  actions: readonly EditorAction[]; onChoose: (action: EditorAction) => void; onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null), search = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState('');
  const words = query.toLocaleLowerCase().trim().split(/\s+/);
  const matches = actions.filter(action => words.every(word => `${action.label} ${action.description} ${action.keywords ?? ''}`.toLocaleLowerCase().includes(word)));
  useEffect(() => {
    const previous = document.activeElement, element = dialog.current!;
    element.showModal(); search.current!.focus();
    return () => { element.close(); if (previous instanceof HTMLElement && previous.isConnected) previous.focus(); };
  }, []);
  function moveFocus(direction: number) {
    const buttons = [...dialog.current!.querySelectorAll<HTMLButtonElement>('.action-results button:not(:disabled)')];
    if (!buttons.length) return;
    const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
    buttons[current < 0 ? direction > 0 ? 0 : buttons.length - 1 : (current + direction + buttons.length) % buttons.length]!.focus();
  }
  return <dialog ref={dialog} className="modal action-search" aria-label="Find an editor action" onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === event.currentTarget) { const bounds = event.currentTarget.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) onClose(); } }}
    onKeyDown={event => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); moveFocus(event.key === 'ArrowDown' ? 1 : -1); }
    }}>
    <button className="modal-close icon-button" aria-label="Close action search" onClick={onClose}><Icon name="close" /></button>
    <span className="eyebrow">TOOLS & ACTIONS</span><h2>What would you like to do?</h2>
    <label className="search-box"><Icon name="search" size={18} /><input ref={search} aria-label="Search editor actions" placeholder="Try draw, textures, objects or save…" value={query}
      onChange={event => setQuery(event.target.value)} onKeyDown={event => {
        if (event.key === 'Enter' && !event.ctrlKey && !event.metaKey) { event.preventDefault(); const first = matches.find(action => !action.disabled); if (first) onChoose(first); }
      }} /></label>
    <div className="action-results">{matches.map(action => <button key={action.id} disabled={action.disabled} onClick={() => onChoose(action)}>
      <span><strong>{action.label}</strong><small>{action.description}</small></span>{action.shortcut && <kbd>{action.shortcut}</kbd>}
    </button>)}</div>
    {!matches.length && <p role="status" className="help-no-results">No actions match. Try a tool name or a shorter word.</p>}
    <p className="action-search-hint"><kbd>↑</kbd> <kbd>↓</kbd> browse · <kbd>Enter</kbd> choose · <kbd>Esc</kbd> close</p>
  </dialog>;
}
