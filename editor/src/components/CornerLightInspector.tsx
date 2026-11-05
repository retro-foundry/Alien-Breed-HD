import { useEffect, useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import { editRoomSource, type EditCommand } from '../model/commands';
import { clearCornerLights, decodeCornerWord, encodeCornerWord, setCornerLight } from '../model/cornerLights';
import { BinaryReader } from '../model/binary';
import { roomPlaneRecords } from '../model/roomPlanes';
import { hasNativeWallMaterials } from '../model/roomMaterials';
import { roomSettings } from '../model/roomHeights';
import { ceilingIsSky } from '../model/roomStoreys';
import { polygonMaterialMode } from '../model/polygonMaterials';
import { applyRoomLightPreset, roomLightPresets } from '../model/roomLightPresets';

interface Draft { brightness: string; animation: number; strength: number; original?: number }
const draft = (word?: number): Draft => word === undefined ? { brightness: '', animation: 0, strength: 1 } : { ...decodeCornerWord(word), brightness: String(decodeCornerWord(word).brightness), original: word };
function LightFields({ label, value, change }: { label: string; value: Draft; change: (value: Draft) => void }) {
  const update = (fields: Partial<Draft>) => { const { original: _old, ...retained } = value; change({ ...retained, ...fields }); };
  return <fieldset><legend>{label} storey</legend>
    <label className="select-field"><span>Brightness</span><input aria-label={`${label} corner brightness`} type="number" min={-128} max={127} step={1} value={value.brightness} placeholder="Choose a value" onChange={event => update({ brightness: event.target.value })} /></label>
    <label className="select-field"><span>Animation</span><select aria-label={`${label} corner animation`} value={value.animation} onChange={event => update({ animation: Number(event.target.value) })}>
      {['Static', 'Pulse', 'Flicker', 'Fire'].map((label, index) => <option key={index} value={index}>{label}</option>)}
      {value.animation > 3 && <option value={value.animation}>Steady blend (original selector {value.animation})</option>}
    </select></label>
    <details><summary>Original animation selector</summary>
      <label className="select-field"><span>Selector (0–15)</span><input type="number" min={0} max={15} step={1} aria-label={`${label} original corner selector`} value={value.animation} onChange={event => { if (event.target.value !== '') update({ animation: Number(event.target.value) }); }} /></label>
      <p className="field-note">0 is Static; 1–3 are Pulse, Flicker and Fire. Original selectors 4–15 blend toward brightness 0, a steady value, using the chosen strength.</p>
    </details>
    <label className="select-field"><span>Animation strength</span><select aria-label={`${label} corner animation strength`} value={value.strength} disabled={!value.animation} onChange={event => update({ strength: Number(event.target.value) })}>
      {Array.from({ length: 16 }, (_, index) => <option key={index} value={index + 1}>{index + 1}/16</option>)}
    </select></label>
  </fieldset>;
}

function CornerLightForm({ document, room, edit }: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand) => void }) {
  const [corner, setCorner] = useState(0), [scope, setScope] = useState('all'), [error, setError] = useState('');
  const [preset, setPreset] = useState(''), [presetUpper, setPresetUpper] = useState(false);
  const point = room.vertices[Math.min(corner, room.vertices.length - 1)]!, record = scope === 'all' ? null : Number(scope);
  const current = room.cornerLights?.find(light => light.x === point.x && light.z === point.z && (light.record ?? null) === record);
  const [lower, setLower] = useState(() => draft(current?.lowerBrightness)), [upper, setUpper] = useState(() => draft(current?.upperBrightness));
  useEffect(() => { setLower(draft(current?.lowerBrightness)); setUpper(draft(current?.upperBrightness)); setError(''); }, [point.x, point.z, scope, current?.lowerBrightness, current?.upperBrightness]);
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const native = hasNativeWallMaterials(room.originZone, room.wallSources), faces = new Map<number, string>();
  const dimensions = roomSettings(document, room);
  for (const upper of [false, true]) {
    if (upper && !(dimensions.upper.length && dimensions.upperFloor > dimensions.upperCeiling)) continue;
    const records = [...(native ? upper ? template.upper : template.lower : []), ...roomPlaneRecords(document, room, upper)];
    const seen = new Set<number>();
    records.filter(item => item.points.length && (item.wall || (item.type === 1 || item.type === 2 && !ceilingIsSky(document, room, upper))
      && polygonMaterialMode(document, { room: room.id, record: item.offset, upper }) === 'normal')).forEach((item, index) => {
      if (seen.has(item.offset)) return; seen.add(item.offset);
      const kind = item.wall ? 'wall' : item.type === 2 ? 'ceiling' : 'floor';
      const label = `${upper ? 'Upper' : 'Lower'} ${kind}`;
      faces.set(item.offset, faces.has(item.offset) ? `${faces.get(item.offset)} / ${label}` : `${label} ${index + 1}`);
    });
  }
  const reader = new BinaryReader(document.files.data, 'original corner lighting');
  const apply = (reset: boolean) => {
    try {
      edit(editRoomSource(setCornerLight(document, room.id, Math.min(corner, room.vertices.length - 1), record, reset ? null : {
        lowerBrightness: lower.original ?? encodeCornerWord(Number(lower.brightness), lower.animation, lower.strength),
        upperBrightness: upper.original ?? encodeCornerWord(Number(upper.brightness), upper.animation, upper.strength),
      }))); setError('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const resetDraft = () => { setLower(draft(current?.lowerBrightness)); setUpper(draft(current?.upperBrightness)); setError(''); };
  return <div onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); resetDraft(); } }}>
    <fieldset><legend>Whole-room lighting</legend>
      <label className="select-field"><span>Storey</span><select aria-label="Lighting preset storey" value={presetUpper ? 'upper' : 'lower'} onChange={event => setPresetUpper(event.target.value === 'upper')}>
        <option value="lower">Lower</option><option value="upper" disabled={!(dimensions.upper.length && dimensions.upperFloor > dimensions.upperCeiling)}>Upper</option>
      </select></label>
      <label className="select-field"><span>Preset</span><select aria-label="Room lighting preset" value={preset} onChange={event => setPreset(event.target.value)}>
        <option value="">Choose a lighting preset</option>{roomLightPresets.map((item, index) => <option key={index} value={index}>{item.label}</option>)}
      </select></label>
      <p className="field-note">Apply a lighting preset to this storey’s normal floors and ceilings as one Undo. Glowing presets mix the original point animations across corners. Use the corner controls below to adjust individual points afterwards. Walls keep their separate lighting.</p>
      <button type="button" className="small-button" disabled={!preset} onClick={() => {
        try { edit(editRoomSource(applyRoomLightPreset(document, room.id, presetUpper, Number(preset)))); setError(''); }
        catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
      }}>Apply room lighting preset</button>
    </fieldset>
    <label className="select-field"><span>Corner</span><select aria-label="Lighting corner" value={Math.min(corner, room.vertices.length - 1)} onChange={event => setCorner(Number(event.target.value))}>
      {room.vertices.map((point, index) => <option key={index} value={index}>Corner {index + 1}: ({point.x}, {point.z})</option>)}
    </select></label>
    <label className="select-field"><span>Surfaces</span><select aria-label="Corner lighting surfaces" value={scope} onChange={event => setScope(event.target.value)}>
      <option value="all">All surfaces at this corner</option>{record !== null && !faces.has(record) && <option value={scope}>Missing face — reset its lighting</option>}{[...faces].map(([id, label]) => <option key={id} value={id}>{label}</option>)}
    </select></label>
    <p className="field-note">Choose lower and upper brightness explicitly. Smaller numbers are brighter. All surfaces is useful before bending an inserted corner; individual normal floors, ceilings and walls can keep separate lighting where they meet this corner. Water and retained bump surfaces use their own shading. Values move and copy with the corner.</p>
    <label className="select-field"><span>Copy original lighting</span><select aria-label="Copy original corner lighting" value="" onChange={event => {
      if (!event.target.value) return;
      const offset = document.pointBrightnessOffset + Number(event.target.value) * 4;
      setLower(draft(reader.i16(offset))); setUpper(draft(reader.i16(offset + 2)));
    }}><option value="">Choose an original point</option>{document.points.map(point => <option key={point.id} value={point.id}>Point {point.id}: ({point.x}, {point.z})</option>)}</select></label>
    <LightFields label="Lower" value={lower} change={setLower} /><LightFields label="Upper" value={upper} change={setUpper} />
    <p className="field-note">Pulse, flicker and fire use the original native animations. Strength controls their blend with the base brightness. Negative brightness uses Static. Copying an original point preserves its saved animation words.</p>
    {error && <p className="field-note" role="alert">{error}</p>}
    <button type="button" className="small-button" disabled={lower.brightness === '' || upper.brightness === ''} onClick={() => apply(false)}>Apply corner lighting</button>
    <button type="button" className="small-button" onClick={resetDraft}>Reset corner draft</button>
    <button type="button" className="small-button" disabled={!current} onClick={() => apply(true)}>Inherit corner lighting</button>
    {!!room.cornerLights?.length && <button type="button" className="small-button" onClick={() => edit(editRoomSource(clearCornerLights(document, room.id)))}>Reset room corner lighting</button>}
  </div>;
}

export function CornerLightInspector(props: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand) => void }) {
  const [open, setOpen] = useState(false);
  return <section className="property-section"><details onToggle={event => { if (event.target === event.currentTarget) setOpen(event.currentTarget.open); }}><summary>Corner lighting</summary>
    {open && <CornerLightForm {...props} />}
  </details></section>;
}
