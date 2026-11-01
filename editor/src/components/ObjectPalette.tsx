import { useEffect, useRef, useState } from 'react';
import { loadObjectPalette, type ObjectPaletteChoice } from '../model/objectPalette';
import { Icon } from './Icon';
import './ObjectBuilder.css';

export function ObjectPalette({ onChoose, onCancel }: { onChoose: (choice: ObjectPaletteChoice) => void; onCancel: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null), search = useRef<HTMLInputElement>(null);
  const [choices, setChoices] = useState<readonly ObjectPaletteChoice[]>(), [query, setQuery] = useState(''), [error, setError] = useState(''), [retry, setRetry] = useState(0);
  useEffect(() => { const element = dialog.current!; element.showModal(); search.current!.focus(); return () => element.close(); }, []);
  useEffect(() => { let active = true; setError(''); loadObjectPalette().then(value => { if (active) setChoices(value); }).catch(error => { if (active) setError(String(error.message ?? error)); }); return () => { active = false; }; }, [retry]);
  const visible = choices?.filter(choice => choice.name.toLowerCase().includes(query.trim().toLowerCase()));
  return <dialog ref={dialog} className="modal object-builder object-palette" aria-label="Choose object to place" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>Add object</h2><p>Choose an object, then click the map to place it. Escape cancels.</p>
    <input ref={search} type="search" aria-label="Find object to place" placeholder="Find an object..." value={query} onChange={event => setQuery(event.target.value)} />
    {!choices && !error && <p role="status">Loading objects...</p>}
    {error && <div role="alert"><p>{error}</p><button type="button" onClick={() => setRetry(value => value + 1)}>Retry objects</button></div>}
    <div className="object-palette-groups">{(['Enemies', 'Pickups', 'Scenery'] as const).map(group => {
      const items = visible?.filter(choice => choice.group === group);
      return items?.length ? <section key={group} aria-label={group}><h3>{group}</h3><div className="object-palette-grid">{items.map(choice => <button type="button" key={choice.id} onClick={() => onChoose(choice)} title={`Place ${choice.name} on the map. Height follows the chosen floor.`}><Icon name="object" size={18} />{choice.name}</button>)}</div></section> : null;
    })}</div>
    {choices && !visible?.length && <p role="status">No matching objects.</p>}
    <button type="button" onClick={onCancel} title="Close without placing anything. Escape also cancels.">Cancel</button>
  </dialog>;
}
