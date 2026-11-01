import { useEffect, useState } from 'react';
import { applyCommand, editRoomAppearance, editRoomSource, replaceWallAsset, type EditCommand } from '../model/commands';
import { encodeProject } from '../model/project';
import type { LevelDocument } from '../model/level';
import { roomAppearanceChoices } from '../model/roomAppearance';
import type { SourceRoom } from '../model/roomSource';
import { WallStyleBrowser } from './WallStyleBrowser';
import { clearRoomWallMaterials } from '../model/wallMaterials';
import { clearRoomPolygonMaterials } from '../model/polygonMaterials';

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
  const changeAsset = (id: number, bytes: Uint8Array | null, png?: Uint8Array) => {
    const command = replaceWallAsset(document, id, bytes, png);
    // Verify the complete portable archive before committing an image import.
    if (png) encodeProject(applyCommand(document, command));
    edit(command);
  };
  const paintedCount = ((room.wallMaterials?.length ?? 0) + (room.boundaryMaterials?.overrides.length ?? 0));
  return <section className="property-section"><h3>{imported ? 'Wall repair appearance' : 'Room appearance'}</h3>
    {!!room.polygonMaterials?.length && <div><p className="field-note">{room.polygonMaterials.length} floor/ceiling surfaces have independent patterns and repeat.</p>
      <button type="button" className="small-button" title="Restore this room's original floor-bank materials, including missing painted surfaces, as one undoable action."
        onClick={() => edit(() => editRoomSource(clearRoomPolygonMaterials(document, room.id)))}>Reset painted room surfaces</button></div>}
    {!!paintedCount && <div><p className="field-note">{paintedCount} source faces have independent painted materials. Reset removes all paints in this room, including faces no longer drawn after an outline edit.</p>
      <button type="button" className="small-button" title="Restore this room's inherited wall materials as one undoable edit. Repairs missing painted-face dependencies. Other rooms are unchanged."
        onClick={() => edit(() => editRoomSource(clearRoomWallMaterials(document, room.id)))}>Reset painted room walls</button></div>}
    <p className="field-note">{imported
      ? 'Choose the style and lighting for new boundary walls and changed doorways. Steps and lintels follow the destination heights; other wall styles stay as authored.'
      : "Choose wall styles and template lighting for the new perimeter. Walls fill this room's floor-to-ceiling height."}</p>
    <form onSubmit={event => { event.preventDefault(); edit(() => editRoomAppearance(document, {
      room: room.id, lowerWall: Number(lower), upperWall: hasUpper ? Number(upper) : null, lightingPoint: Number(light),
    })); }}>
      {[{ label: 'Lower wall style', records: choices.lower, value: lower, change: setLower },
        ...(hasUpper ? [{ label: 'Upper wall style', records: choices.upper, value: upper, change: setUpper }] : [])].map(field =>
        <label className="select-field" key={field.label}><span>{field.label}</span><select required aria-label={field.label} title="Choose an existing straight-wall style from this room's selected template. Texture, alignment and sampling are retained." value={field.value} onChange={event => field.change(event.target.value)}>
          <option value="">Choose a wall style</option>{field.records.map((record, index) => <option key={record.offset} value={record.offset}>Texture {record.texture} · Style {index + 1}</option>)}
        </select></label>)}
      <WallStyleBrowser key={`${room.id}:lower`} label="lower wall" records={choices.lower} value={lower} onChange={setLower}
        overrides={document.wallOverrides} baseline={document.original.data} onAssetChange={changeAsset} />
      {hasUpper && <WallStyleBrowser key={`${room.id}:upper`} label="upper wall" records={choices.upper} value={upper} onChange={setUpper}
        overrides={document.wallOverrides} baseline={document.original.data} onAssetChange={changeAsset} />}
      <label className="select-field"><span>Corner lighting</span><select required aria-label="Room corner lighting" title="Use the selected template point's actual lower and upper lighting for newly generated walls." value={light} onChange={event => setLight(event.target.value)}>
        <option value="">Choose template lighting</option>{choices.points.map((id, index) => <option key={id} value={id}>Match template point {index + 1}</option>)}
      </select></label>
      <button className="small-button" type="submit" title="Apply the new-wall appearance as one undoable edit, then rebuild using preview or Export level." disabled={!lower || !light || (hasUpper && !upper)}>Apply appearance</button>
    </form>
  </section>;
}
