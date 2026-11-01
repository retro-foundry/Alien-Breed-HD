import { useEffect, useMemo, useRef, useState } from 'react';
import { pasteObjects, type ObjectClipboard } from '../model/objectClipboard';
import { editableObjects, runtimeObjectPosition } from '../model/sourceObjects';
import type { LevelDocument } from '../model/level';
import './ObjectBuilder.css';

export function ObjectPaste({ document, clipboard, cursor, onCancel, onPaste }: {
  document: LevelDocument; clipboard: ObjectClipboard; cursor: { x: number; z: number };
  onCancel: () => void; onPaste: (proposal: ReturnType<typeof pasteObjects>) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const objects = useMemo(() => {
    const source = editableObjects(clipboard.source);
    return clipboard.ids.map(id => source.find(object => object.id === id)!);
  }, [clipboard]);
  const anchor = objects[0]!;
  const [x, setX] = useState(String(anchor.x)), [z, setZ] = useState(String(anchor.z));
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  const result = useMemo(() => {
    if (!x.trim() || !z.trim()) return {};
    try {
      const position = runtimeObjectPosition({ x: Number(x), z: Number(z) });
      return { proposal: pasteObjects(document, clipboard, position.x - anchor.x, position.z - anchor.z), position };
    } catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
  }, [document, clipboard, anchor.x, anchor.z, x, z]);
  return <dialog ref={dialog} className="modal object-builder" aria-label="Place pasted objects" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>Place {objects.length} copied {objects.length === 1 ? 'object' : 'objects'}</h2>
    <p>Choose where the first copied object goes. The whole selection keeps its spacing, settings, floors and exact heights. Nothing changes until Paste here. Escape or Cancel paste discards the placement.</p>
    <form onSubmit={event => { event.preventDefault(); if (result.proposal) onPaste(result.proposal); }}>
      <div className="field-grid">{(['x', 'z'] as const).map(axis => <label className="number-field" key={axis}><span>First object {axis.toUpperCase()}</span>
        <input required type="number" step="any" aria-label={`Paste anchor ${axis.toUpperCase()}`} value={axis === 'x' ? x : z}
          title="Destination coordinate of the first copied object. Every other object moves by the same amount; each destination must support its floor."
          onChange={event => (axis === 'x' ? setX : setZ)(event.target.value)} /></label>)}</div>
      <button type="button" title="Use the last position of the pointer over the map as the first object's destination, rounded to whole world units. All destinations are validated before applying."
        onClick={() => { setX(String(Math.round(cursor.x))); setZ(String(Math.round(cursor.z))); }}>Use map cursor</button>
      {result.error && <p role="alert">{result.error}</p>}
      {result.proposal && <p role="status">Ready to paste {objects.length} objects. First object: X {result.position.x}, Z {result.position.z}. One Undo removes the whole paste.</p>}
      <div className="object-builder-actions"><button type="button" title="Discard the placement without changing the project. Escape also cancels." onClick={onCancel}>Cancel paste</button>
        <button type="submit" className="primary-button" disabled={!result.proposal} title="Validate all destinations and paste the captured objects as one undoable edit.">Paste here</button></div>
    </form>
  </dialog>;
}
