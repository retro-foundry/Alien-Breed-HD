import { useEffect, useMemo, useRef, useState } from 'react';
import { objectName, type LevelDocument } from '../model/level';
import { createObject, floorMatchedObjectHeight, objectSlotLimit, objectTemplates } from '../model/objectAuthoring';
import './ObjectBuilder.css';
import { editableObjects } from '../model/sourceObjects';

export function ObjectBuilder({ document, room, cursor, layer, initialObject, onCancel, onCreate }: {
  document: LevelDocument; room?: string; cursor: { x: number; z: number }; onCancel: () => void;
  layer: 'lower' | 'upper';
  initialObject?: number;
  onCreate: (proposal: ReturnType<typeof createObject>) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const templates = objectTemplates(document);
  const initial = initialObject === undefined ? undefined : editableObjects(document).find(object => object.id === initialObject && templates.some(template => template.id === object.id));
  const [draft, setDraft] = useState({ template: initial ? String(initial.id) : '',
    room: initial ? initial.room ?? document.roomSource.rooms.find(room => room.originZone === initial.zone)?.id ?? '' : room ?? '',
    x: String(initial?.x ?? Math.round(cursor.x)), z: String(initial?.z ?? Math.round(cursor.z)), y: initial ? String(initial.y) : '' });
  const [matchFloor, setMatchFloor] = useState(!initial);
  const [floor, setFloor] = useState(initial ? initial.upper ? 'upper' : 'lower' : layer);
  const template = templates.find(item => String(item.id) === draft.template);
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  const result = useMemo(() => {
    if (!template || !draft.room || !draft.x.trim() || !draft.z.trim() || (!matchFloor && !draft.y.trim())) return {};
    try {
      const y = matchFloor ? floorMatchedObjectHeight(document, template, draft.room, floor === 'upper') : Number(draft.y);
      return { y, proposal: createObject(document, { template: template.id, room: draft.room, x: Number(draft.x), z: Number(draft.z), y, upper: floor === 'upper' }) };
    } catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
  }, [document, draft, matchFloor, template, floor]);
  return <dialog ref={dialog} className="modal object-builder" aria-label="Add object" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>{initial ? `Duplicate ${objectName(initial.type)} ${initial.id}` : 'Add enemy, pickup or scenery'}</h2>
    {initial && <p>Settings, coordinates, floor and exact height are copied. Choose a new position, then Create object. Nothing changes until you create the copy.</p>}
    <p>Choose an existing level object as its settings template, then its destination. A fresh record and position slot are created; the template stays intact. Cancel or Escape discards this draft.</p>
    <form onSubmit={event => { event.preventDefault(); if (result.proposal) onCreate(result.proposal); }}>
      <label className="select-field"><span>Object template</span><select required aria-label="New object template" value={draft.template}
        title="Copy an actual source object's type, artwork, health and type-specific settings. Players and reserved projectile records are excluded."
        onChange={event => { const object = templates.find(item => String(item.id) === event.target.value)!; setDraft(current => ({ ...current, template: event.target.value, y: object ? String(object.y) : '' })); }}>
        <option value="">Choose an object template</option>{templates.map(object => <option key={object.id} value={object.id}>{objectName(object.type)} · Object {object.id}</option>)}
      </select></label>
      {template && <p className="field-note">{objectName(template.type)} · {template.upper ? 'Upper floor' : 'Lower floor'}. Copied settings can be edited after creation.</p>}
      <label className="select-field"><span>Destination room</span><select required aria-label="New object room" value={draft.room} title="Choose an editable room, including newly drawn rooms. The position must lie inside its outline." onChange={event => setDraft(current => ({ ...current, room: event.target.value }))}>
        <option value="">Choose a room</option>{document.roomSource.rooms.map(room => <option key={room.id} value={room.id}>Room {room.id.replace(/^room-/, '')}{room.originZone === null ? ' (new)' : ` · Zone ${room.originZone}`}</option>)}
      </select></label>
      <label className="select-field"><span>Floor</span><select aria-label="New object floor" value={floor} title="Choose which storey contains this object. Upper needs a destination room with an upper floor." onChange={event => setFloor(event.target.value as 'lower' | 'upper')}>
        <option value="lower">Lower floor</option><option value="upper">Upper floor</option>
      </select></label>
      <div className="field-grid">{(['x', 'z'] as const).map(axis => <label className="number-field" key={axis}><span>{axis.toUpperCase()}</span>
        <input required type="number" step="any" min={-32768} max={32768 - 1 / 65536} aria-label={`New object ${axis.toUpperCase()}`} value={draft[axis]}
          title="Position inside the destination room. Decimal coordinates are converted automatically to the native position table."
          onChange={event => setDraft(current => ({ ...current, [axis]: event.target.value }))} /></label>)}</div>
      <label className="object-floor-match"><input type="checkbox" aria-label="Match template floor offset" checked={matchFloor} title="Keep the template's distance above or below its floor, following the destination floor height. Native object heights snap to half units." onChange={event => { if (!event.target.checked && result.y !== undefined) setDraft(current => ({ ...current, y: String(result.y) })); setMatchFloor(event.target.checked); }} />Match template floor offset</label>
      <label className="number-field"><span>Height</span><input required type="number" step={0.5} min={-16384} max={16383.5} readOnly={matchFloor} aria-label="New object height"
        title="Smaller values are higher. Uncheck floor matching to choose an absolute height. Objects use native half units."
        value={matchFloor ? result.y ?? '' : draft.y} onChange={event => setDraft(current => ({ ...current, y: event.target.value }))} /></label>
      {result.error && <p role="alert">{result.error}</p>}
      {result.proposal && <p role="status">Ready to add object {result.proposal.id}. {document.objects.length + 1} of {objectSlotLimit} native records used, including reserved slots.</p>}
      {!templates.length && <p role="alert">This level has no supported object template. Open a level containing an enemy, pickup or scenery record.</p>}
      <div className="object-builder-actions"><button type="button" title="Discard this object draft. Escape also cancels." onClick={onCancel}>Cancel object</button>
        <button type="submit" className="primary-button" disabled={!result.proposal} title="Add this object as one undoable edit. Its position has a fresh identity; existing objects, pools and player avatars are retained.">Create object</button></div>
    </form>
  </dialog>;
}
