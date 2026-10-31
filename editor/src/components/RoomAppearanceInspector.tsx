import { useEffect, useState } from 'react';
import { editRoomAppearance, type EditCommand } from '../model/commands';
import type { LevelDocument } from '../model/level';
import { roomAppearanceChoices } from '../model/roomAppearance';
import type { SourceRoom } from '../model/roomSource';

export function RoomAppearanceInspector({ document, room, edit }: {
  document: LevelDocument; room: SourceRoom; edit: (command: EditCommand | (() => EditCommand)) => void;
}) {
  const choices = roomAppearanceChoices(document, room), selected = document.roomAppearances?.find(item => item.room === room.id);
  const [lower, setLower] = useState(selected ? String(selected.lowerWall) : '');
  const [upper, setUpper] = useState(selected?.upperWall === undefined || selected.upperWall === null ? '' : String(selected.upperWall));
  const [light, setLight] = useState(selected ? String(selected.lightingPoint) : '');
  useEffect(() => {
    setLower(selected ? String(selected.lowerWall) : ''); setUpper(selected?.upperWall === undefined || selected.upperWall === null ? '' : String(selected.upperWall));
    setLight(selected ? String(selected.lightingPoint) : '');
  }, [room.id, selected?.lowerWall, selected?.upperWall, selected?.lightingPoint]);
  const hasUpper = choices.template.upper.length > 0, imported = room.originZone !== null;
  return <section className="property-section"><h3>{imported ? 'Doorway closure appearance' : 'Room appearance'}</h3>
    <p className="field-note">{imported
      ? 'Choose the style and lighting for walls that close doorways after adjoining rooms are removed or moved. Existing wall styles stay as authored.'
      : "Choose wall styles and template lighting for the new perimeter. Walls fill this room's floor-to-ceiling height."}</p>
    <form onSubmit={event => { event.preventDefault(); edit(() => editRoomAppearance(document, {
      room: room.id, lowerWall: Number(lower), upperWall: hasUpper ? Number(upper) : null, lightingPoint: Number(light),
    })); }}>
      {[{ label: 'Lower wall style', records: choices.lower, value: lower, change: setLower },
        ...(hasUpper ? [{ label: 'Upper wall style', records: choices.upper, value: upper, change: setUpper }] : [])].map(field =>
        <label className="select-field" key={field.label}><span>{field.label}</span><select required aria-label={field.label} title="Choose an existing straight-wall style from this room's selected template. Texture, alignment and sampling are retained." value={field.value} onChange={event => field.change(event.target.value)}>
          <option value="">Choose a wall style</option>{field.records.map((record, index) => <option key={record.offset} value={record.offset}>Texture {record.texture} · Style {index + 1}</option>)}
        </select></label>)}
      <label className="select-field"><span>Corner lighting</span><select required aria-label="Room corner lighting" title="Use the selected template point's actual lower and upper lighting for newly generated walls." value={light} onChange={event => setLight(event.target.value)}>
        <option value="">Choose template lighting</option>{choices.points.map((id, index) => <option key={id} value={id}>Match template point {index + 1}</option>)}
      </select></label>
      <button className="small-button" type="submit" title="Apply the new-wall appearance as one undoable edit, then rebuild using preview or Export level." disabled={!lower || !light || (hasUpper && !upper)}>Apply appearance</button>
    </form>
  </section>;
}
