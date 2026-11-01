import { useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { EditCommand } from '../model/commands';
import { editObjectGroup } from '../model/objectGroups';

export function ObjectGroupInspector({ document, ids, edit, clear, frame }: { document: LevelDocument; ids: readonly number[];
  edit: (command: EditCommand | (() => EditCommand)) => void; clear: () => void; frame: () => void }) {
  const [x, setX] = useState('0'), [z, setZ] = useState('0');
  return <aside className="inspector" aria-label="Object group inspector">
    <div className="panel-title">INSPECTOR · OBJECT GROUP</div><div className="inspector-scroll">
      <section className="property-section"><h2>{ids.length} selected objects</h2><p className="field-note">Shift, Ctrl or Command click objects in the map or list to add or remove them. Drag a selected marker to move the group; Escape cancels. Click an unselected object to select it alone.</p>
        <p className="field-note">Alignment uses the first selected object ({ids[0]}). Group edits are one undoable action; invalid destinations keep the whole group unchanged.</p>
        <button className="small-button" onClick={frame} title="Fit all selected objects into the map using their current authored coordinates. Shortcut: Shift + F. Changes only the view.">Frame selected objects</button>
        <button className="small-button" onClick={clear} title="Clear the whole object selection. Escape also clears it.">Clear object selection</button></section>
      <section className="property-section"><h3>Move by offset</h3><form onSubmit={event => { event.preventDefault(); if (x.trim() && z.trim()) edit(() => editObjectGroup(document, ids, { kind: 'move', x: Number(x), z: Number(z) })); }}>
        <div className="field-grid">{(['x', 'z'] as const).map(axis => <label className="number-field" key={axis}><span>{axis.toUpperCase()}</span><input type="number" required step="any" aria-label={`Group ${axis.toUpperCase()} offset`} value={axis === 'x' ? x : z}
          title="Add this offset to every selected object's coordinate. Decimal values are supported. Submit Move selected objects to apply both offsets together."
          onChange={event => (axis === 'x' ? setX : setZ)(event.target.value)} /></label>)}</div>
        <button className="small-button" type="submit" title="Validate every destination, then move the complete selection as one undoable edit.">Move selected objects</button>
      </form></section>
      <section className="property-section"><h3>Align to first selected</h3>{(['x', 'z'] as const).map(axis => <button className="small-button" key={axis} title={`Set every selected object's ${axis.toUpperCase()} to the first selected object's coordinate. Other coordinates and heights stay intact. All destinations must be valid.`}
        onClick={() => edit(() => editObjectGroup(document, ids, { kind: 'align', axis }))}>Align {axis.toUpperCase()}</button>)}</section>
      <section className="property-section"><button className="danger-button" title="Delete all selected objects as one action. Shortcut: Delete. Undo restores the complete group. Player avatars cannot be deleted."
        onClick={() => edit(() => editObjectGroup(document, ids, { kind: 'delete' }))}>Delete selected objects</button></section>
    </div>
  </aside>;
}
