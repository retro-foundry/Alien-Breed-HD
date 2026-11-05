import { normalPlaneChoices } from '../model/roomPlanes';
import { floorPatternOffsets } from '../model/floorTextures';
import { useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import { applyCommand, editRoomSource, type EditCommand } from '../model/commands';
import { roomSettings } from '../model/roomHeights';
import { ceilingIsSky, setSkyCeiling, setUpperStorey, storeyMaterials } from '../model/roomStoreys';
import { createRoomOpening, openingParts, type RoomOpening } from '../model/roomOpening';
import { buildPlayableLevel } from '../model/playableBuild';

export function RoomStoreyInspector({ document, room, edit }: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand | (() => EditCommand)) => void }) {
  const settings = roomSettings(document, room), hasUpper = settings.upper.length > 0 && settings.upperFloor > settings.upperCeiling;
  const materials = storeyMaterials(document, room), template = document.zones.find(zone => zone.id === room.recordZone)!;
  const [upperFloor, setUpperFloor] = useState(String(settings.ceiling)), [headroom, setHeadroom] = useState(String(settings.floor - settings.ceiling));
  const [floorMaterial, setFloorMaterial] = useState(materials.floor ? String(materials.floor.offset) : ''), [ceilingMaterial, setCeilingMaterial] = useState(materials.ceiling ? String(materials.ceiling.offset) : '');
  const [wall, setWall] = useState(''), [light, setLight] = useState(''), [upperWall, setUpperWall] = useState('');
  const [error, setError] = useState('');
  const [openingKind, setOpeningKind] = useState<'storeys' | 'sky'>(hasUpper ? 'storeys' : 'sky');
  const xs = room.vertices.map(p => p.x), zs = room.vertices.map(p => p.z);
  const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...zs), bottom = Math.max(...zs);
  const [bounds, setBounds] = useState<Record<keyof RoomOpening, string>>({ left: String(Math.round(left + (right - left) / 3)), right: String(Math.round(right - (right - left) / 3)), top: String(Math.round(top + (bottom - top) / 3)), bottom: String(Math.round(bottom - (bottom - top) / 3)) });
  const opening = Object.fromEntries(Object.entries(bounds).map(([key, value]) => [key, Number(value)])) as unknown as RoomOpening;
  let openingError = '';
  try { openingParts(room.vertices, opening); } catch (reason) { openingError = reason instanceof Error ? reason.message : String(reason); }
  const commit = (make: () => EditCommand) => {
    try { const command = make(); buildPlayableLevel(applyCommand(document, command), -1); edit(command); setError(''); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const lighting = [...new Set(template.lower.flatMap(record => record.points))];
  const styleFields = <>
    <label className="select-field"><span>Wall style</span><select aria-label="Storey wall style" required value={wall} onChange={event => setWall(event.target.value)}><option value="">Choose a wall style</option>{materials.walls.map((record, index) => <option key={record.offset} value={record.offset}>Texture {record.texture}, style {index + 1}</option>)}</select></label>
    <label className="select-field"><span>Match corner lighting</span><select aria-label="Storey corner lighting" required value={light} onChange={event => setLight(event.target.value)}><option value="">Choose existing lighting</option>{lighting.map((id, index) => <option key={id} value={id}>Room point {index + 1}</option>)}</select></label>
  </>;
  const openingForm = <details><summary>{hasUpper ? 'Create opening between storeys' : 'Create sky opening'}</summary><form onSubmit={event => { event.preventDefault(); commit(() => createRoomOpening(document, room.id, opening, { lowerWall: Number(wall), upperWall: !hasUpper || room.upperStorey ? null : Number(upperWall), lightingPoint: Number(light) }, openingKind).command); }}>
        <label className="select-field"><span>Opening destination</span><select aria-label="Opening destination" value={openingKind} onChange={event => setOpeningKind(event.target.value as 'storeys' | 'sky')}>
          {hasUpper && <option value="storeys">Between lower and upper storeys</option>}<option value="sky">Open top ceiling to sky</option>
        </select></label>
        <p className="field-note">{openingKind === 'storeys' ? 'Cut a rectangular hole through the lower ceiling and upper floor. You can see through it and fall into the lower storey.' : 'Cut a rectangular skylight in the top ceiling. The floor stays in place.'} The editor divides this room into the opening and surrounding connected zones. One Undo restores the original room.</p>
        <div className="field-grid">{(['left', 'right', 'top', 'bottom'] as const).map(field => <label className="number-field" key={field}><span>{field === 'top' || field === 'bottom' ? 'Z' : 'X'} {field}</span><input required type="number" step="1" aria-label={`Opening ${field}`} value={bounds[field]} onChange={event => setBounds(current => ({ ...current, [field]: event.target.value }))} /></label>)}</div>
        <svg viewBox={`${left} ${top} ${right - left} ${bottom - top}`} width="240" height="160" role="img" aria-label="Opening layout preview" style={{ width: '100%', background: '#141b22' }}>
          <polygon points={room.vertices.map(p => `${p.x},${p.z}`).join(' ')} fill="#365349" stroke="#b2de7a" strokeWidth={(right - left) / 150} />
          {!openingError && <rect x={opening.left} y={opening.top} width={opening.right - opening.left} height={opening.bottom - opening.top} fill="#141b22" stroke="#f0c878" strokeWidth={(right - left) / 100} />}
        </svg>
        {openingError && <p className="field-note" role="status">{openingError}</p>}
        {styleFields}
        {hasUpper && !room.upperStorey && <label className="select-field"><span>Upper wall style</span><select required aria-label="Opening upper wall style" value={upperWall} onChange={event => setUpperWall(event.target.value)}><option value="">Choose an upper wall style</option>{template.upper.filter(record => record.wall).map((record, index) => <option key={record.offset} value={record.offset}>Texture {record.texture}, style {index + 1}</option>)}</select></label>}
        <button type="submit" className="small-button" disabled={!!openingError || !wall || !light || hasUpper && !room.upperStorey && !upperWall}>Create opening</button>
      </form></details>;
  return <section className="property-section"><h3>Storeys and openings</h3>
    <p className="field-note">{hasUpper ? 'Two storeys share this outline. Each has its own floor, ceiling and textures.' : 'This room has one storey. Add an upper storey over the same outline below.'}</p>
    {error && <p className="field-note" role="alert">{error}</p>}
    {!hasUpper ? <details><summary>Add upper storey</summary><form onSubmit={event => { event.preventDefault(); commit(() => editRoomSource(setUpperStorey(document, room.id, { floor: Number(upperFloor), ceiling: Number(upperFloor) - Number(headroom), wall: Number(wall), lightingPoint: Number(light), floorMaterial: Number(floorMaterial), ceilingMaterial: Number(ceilingMaterial) }))); }}>
      <p className="field-note">The upper floor starts at the lower ceiling height. Choose its floor and ceiling starting patterns below, then edit them independently under Textures. Connect adjoining rooms or stairs to reach it in the game.</p>
      <label className="number-field"><span>Upper floor position</span><input aria-label="New upper floor position" type="number" required step="0.25" value={upperFloor} onChange={event => setUpperFloor(event.target.value)} /></label>
      <label className="number-field"><span>Upper room height</span><input aria-label="New upper room height" type="number" required min="0.25" step="0.25" value={headroom} onChange={event => setHeadroom(event.target.value)} /></label>
      {([{ name: 'floor', type: 1, value: floorMaterial, change: setFloorMaterial }, { name: 'ceiling', type: 2, value: ceilingMaterial, change: setCeilingMaterial }] as const).map(field => <label className="select-field" key={field.name}><span>Upper {field.name} pattern</span><select aria-label={`New upper ${field.name} pattern`} required value={field.value} onChange={event => field.change(event.target.value)}><option value="">Choose a pattern</option>{normalPlaneChoices(document, field.type, Number(field.value)).map(record => <option key={record.offset} value={record.offset}>Pattern {floorPatternOffsets.indexOf(record.texture!) + 1}</option>)}</select></label>)}
      {styleFields}
      <button type="submit" className="small-button" disabled={!wall || !light || !floorMaterial || !ceilingMaterial}>Create upper storey</button>
    </form></details> : <>
      <button type="button" className="small-button" onClick={() => commit(() => editRoomSource(setUpperStorey(document, room.id, null)))}>Remove upper storey</button>
      {openingForm}
    </>}
    {!hasUpper && openingForm}
    <h4>Ceiling or open sky</h4>
    <p className="field-note">Open to sky removes the top ceiling in the saved game. Show ceilings in the viewport only changes the editing view.</p>
    <label className="check-field"><input type="checkbox" aria-label={hasUpper ? 'Upper ceiling open to sky' : 'Ceiling open to sky'} checked={ceilingIsSky(document, room, hasUpper)}
      onChange={event => commit(() => editRoomSource(setSkyCeiling(document, room.id, hasUpper, event.target.checked)))} />{hasUpper ? 'Upper ceiling open to sky' : 'Ceiling open to sky'}</label>
  </section>;
}
