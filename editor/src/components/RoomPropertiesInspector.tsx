import type { LevelDocument } from '../model/level';
import type { EditCommand } from '../model/commands';
import { editRoomSource } from '../model/commands';
import { setRoomProperties, type SourceRoom } from '../model/roomSource';
import { roomSettings } from '../model/roomHeights';
import { captureRoomProperties, type RoomProperties } from '../model/roomProperties';
import { NumberField } from './NumberField';

export function RoomPropertiesInspector({ document, room, edit }: {
  document: LevelDocument; room: SourceRoom; edit: (command: EditCommand | (() => EditCommand)) => void;
}) {
  const settings = roomSettings(document, room);
  const change = (property: keyof RoomProperties, value: number | boolean) => edit(() => editRoomSource(setRoomProperties(document.roomSource, room.id,
    { ...captureRoomProperties(settings), [property]: value })));
  return <section className="property-section" aria-label="Room lighting and properties"><h3>Lighting</h3>
    {(['brightness', 'upperBrightness'] as const).filter(property => (property === 'brightness' ? settings.lower : settings.upper)
      .some(record => [8, 9, 10, 11].includes(record.type))).map(property => {
      const value = settings[property], animation = (value >>> 8) & 255, animated = animation >= 1 && animation <= 3;
      return <div className="light-control" key={property}>
        <label className="select-field"><span>{property === 'brightness' ? 'Lower light' : 'Upper light'}</span>
          <select aria-label={`${property} animation`} value={animated ? animation : 0} title="Choose steady lighting, pulse, flicker or fire for this room. Static lighting enables its brightness field."
            onChange={event => change(property, Number(event.target.value) << 8)}>
            <option value={0}>Static</option><option value={1}>Pulse</option><option value={2}>Flicker</option><option value={3}>Fire</option>
          </select></label>
        {!animated && <NumberField label={property === 'brightness' ? 'Brightness' : 'Upper brightness'} displayLabel={property === 'brightness' ? 'Flat surface brightness' : 'Upper flat surface brightness'} value={value} min={-128} max={127}
          help="Lower values brighten flat-shaded surfaces in this room. Materials with per-point shading use their lighting points. Enter applies, Escape cancels; Undo restores the previous value." onCommit={value => change(property, value)} />}
      </div>;
    })}
    <p className="field-note">Bump and smooth surfaces use room brightness. Walls and surfaces with per-point shading use material lighting points. Water uses its native distance shading. Inherit template properties restores lighting, water, sky and sound settings.</p>
    <button className="small-button" disabled={!room.properties} title="Remove this room's captured properties and inherit the current template's lighting, controls, water, sky and sound values. One Undo restores the snapshot."
      onClick={() => edit(() => editRoomSource(setRoomProperties(document.roomSource, room.id, null)))}>Inherit template properties</button>
  </section>;
}
