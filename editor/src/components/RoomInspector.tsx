import type { SourceRoom } from '../model/roomSource';
import { NumberField } from './NumberField';

export function RoomInspector({ room, onCorner, onDelete }: {
  room: SourceRoom; onCorner: (index: number, axis: 'x' | 'z', value: number) => void; onDelete: () => void;
}) {
  return <section className="property-section"><h3>Room outline</h3>
    <p className="field-note">{room.vertices.length} corners · Surface template zone {room.recordZone}. Drag a highlighted map corner, or edit its coordinates below. Adjoining corners move together.</p>
    <details><summary title="Edit whole-coordinate corners. Enter applies one complete validated outline; Escape cancels a field edit.">Edit room corners</summary>
      {room.vertices.map((point, index) => <div className="room-corner-row" key={index}><span className="eyebrow">Corner {index + 1}</span><div className="field-grid">
        <NumberField label={`Corner ${index + 1} X`} displayLabel="X" value={point.x} help="Move this corner horizontally. Crossing outlines are rejected without changing the room." onCommit={value => onCorner(index, 'x', value)} />
        <NumberField label={`Corner ${index + 1} Z`} displayLabel="Z" value={point.z} help="Move this corner vertically on the overhead map. The editor handles room orientation." onCommit={value => onCorner(index, 'z', value)} />
      </div></div>)}
    </details>
    <p className="field-note">The outline applies to both floors. Preview and Export level rebuild the game files; Save project keeps the editable source.</p>
    <button className="danger-button" title="Delete this room outline as one undoable edit. Objects and links are retained and will need reassignment before building." onClick={onDelete}>Delete room</button>
  </section>;
}
