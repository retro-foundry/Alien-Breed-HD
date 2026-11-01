import type { SourceRoom } from '../model/roomSource';

export function RoomGroupInspector({ rooms, frame, clear, remove }: {
  rooms: readonly SourceRoom[]; frame: () => void; clear: () => void; remove: () => void;
}) {
  return <aside className="inspector" aria-label="Room group inspector"><div className="panel-title">INSPECTOR · ROOM GROUP</div><div className="inspector-scroll">
    <section className="property-section"><h2>{rooms.length} selected rooms</h2>
      <p className="field-note">Shift, Ctrl or Command click rooms in the map or list to add or remove them. Click a room normally to inspect it alone.</p>
      <p className="field-note">Group deletion is one undoable action. Linked starts, objects, teleports and portal appearances remain editable; repair any reported build dependencies before export.</p>
      <ul>{rooms.map(room => <li key={room.id}>Room {room.id.replace(/^room-/, '')}{room.originZone === null ? ' (new)' : ` · Zone ${room.originZone}`}</li>)}</ul>
      <button className="small-button" title="Fit every selected room outline in the map. Changes only the view. Shortcut: Shift+F." onClick={frame}>Frame selected rooms</button>
      <button className="small-button" title="Clear this room group. Escape also clears selection." onClick={clear}>Clear room selection</button>
    </section>
    <section className="property-section"><button className="danger-button" title="Delete every selected room as one source edit. Undo restores the whole group. Shortcut: Delete." onClick={remove}>Delete selected rooms</button></section>
  </div></aside>;
}
