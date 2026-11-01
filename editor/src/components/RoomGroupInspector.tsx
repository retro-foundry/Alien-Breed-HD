import type { SourceRoom } from '../model/roomSource';
import { useState } from 'react';

export function RoomGroupInspector({ rooms, frame, clear, remove, move }: {
  rooms: readonly SourceRoom[]; frame: () => void; clear: () => void; remove: () => void; move: (x: number, z: number) => void;
}) {
  const [x, setX] = useState('0'), [z, setZ] = useState('0');
  return <aside className="inspector" aria-label="Room group inspector"><div className="panel-title">INSPECTOR · ROOM GROUP</div><div className="inspector-scroll">
    <section className="property-section"><h2>{rooms.length} selected rooms</h2>
      <p className="field-note">Shift, Ctrl or Command click rooms in the map or list to add or remove them. Click a room normally to inspect it alone.</p>
      <p className="field-note">In 3D, drag LF/LC to change every selected floor/ceiling by the same amount. UF/UC are available when every zone has an upper storey. Height differences are preserved; Undo restores the whole group and Escape cancels a drag.</p>
      <p className="field-note">Group deletion is one undoable action. Linked starts, objects, teleports and portal appearances remain editable; repair any reported build dependencies before export.</p>
      <ul>{rooms.map(room => <li key={room.id}>Room {room.id.replace(/^room-/, '')}{room.originZone === null ? ' (new)' : ` · Zone ${room.originZone}`}</li>)}</ul>
      <button className="small-button" title="Fit every selected room outline in the map. Changes only the view. Shortcut: Shift+F." onClick={frame}>Frame selected rooms</button>
      <button className="small-button" title="Clear this room group. Escape also clears selection." onClick={clear}>Clear room selection</button>
    </section>
    <section className="property-section"><h3>Move rooms and links</h3>
      <p className="field-note">Whole-unit offsets move the selected rooms, their starts and objects, and incoming teleport destinations. Shared boundary corners follow through adjoining rooms; other neighbour corners stay anchored.</p>
      <form onSubmit={event => { event.preventDefault(); if (x.trim() && z.trim()) move(Number(x), Number(z)); }}>
        <div className="field-grid">{(['x', 'z'] as const).map(axis => <label className="number-field" key={axis}><span>{axis.toUpperCase()} offset</span>
          <input required type="number" step={1} aria-label={`Room group ${axis.toUpperCase()} offset`} value={axis === 'x' ? x : z}
            title="Translate the whole room group by this many world units. Shared portals and positional links move in the same undoable action. Whole units only."
            onChange={event => (axis === 'x' ? setX : setZ)(event.target.value)} /></label>)}</div>
        <button className="small-button" type="submit" disabled={!x.trim() || !z.trim() || (Number(x) === 0 && Number(z) === 0)} title="Validate the final geometry and moved positional links, then apply the complete translation as one source edit. Undo restores everything.">Move selected rooms</button>
      </form>
    </section>
    <section className="property-section"><button className="danger-button" title="Delete every selected room as one source edit. Undo restores the whole group. Shortcut: Delete." onClick={remove}>Delete selected rooms</button></section>
  </div></aside>;
}
