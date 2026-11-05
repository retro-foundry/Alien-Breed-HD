import { useEffect, useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import type { EditCommand } from '../model/commands';
import { importedAnimationTables, type WaterAnimation } from '../model/animationBuild';
import { inheritedWaterGroups, setRoomWater, setWaterGroup, waterGroupSettings, waterMaterialChoices, waterStartHeight } from '../model/sourceWater';
import { roomSettings } from '../model/roomHeights';

interface Props { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand) => void; onInspect: (id: string) => void }
export function WaterInspector(props: Props) {
  let native: readonly WaterAnimation[];
  try { native = importedAnimationTables(props.document).tables.water; }
  catch (reason) { return <section className="property-section"><h3>Water and animation groups</h3><p role="alert" className="field-note">Water controls could not read the native animation tables: {reason instanceof Error ? reason.message : String(reason)} Your source is retained and can still be saved.</p></section>; }
  return <WaterEditor {...props} native={native} />;
}
function WaterEditor({ document, room, edit, onInspect, native }: Props & { native: readonly WaterAnimation[] }) {
  const inherited = inheritedWaterGroups(document, room), settings = roomSettings(document, room);
  const existing = document.zones.find(zone => zone.id === room.recordZone)!.lower.some(record => record.type === 7);
  const group = room.waterSurface === undefined ? inherited.length === 1 ? inherited[0]! : null : room.waterSurface === false ? null : room.waterSurface.group;
  const conflict = room.waterSurface === undefined && inherited.length > 1;
  const values = () => ({ material: room.waterSurface ? room.waterSurface.material === null ? 'existing' : String(room.waterSurface.material) : existing ? 'existing' : '', height: String(conflict ? settings.water : waterStartHeight(document, room)), group: group === null ? 'static' : String(group) });
  const [draft, setDraft] = useState(values), [error, setError] = useState(''), [groupId, setGroupId] = useState(group ?? 0);
  const [open, setOpen] = useState(false), [groupOpen, setGroupOpen] = useState(false);
  const configured = (id: number) => !!document.roomSource.waterGroups?.some(group => group.id === id) || !!native[id]?.entries.length;
  const groupValues = () => {
    const group = waterGroupSettings(document, groupId), present = configured(groupId);
    return { top: present ? String(group.top) : '', bottom: present ? String(group.bottom) : '', current: present ? String(group.current) : '', velocity: String(present ? group.velocity : 128), enabled: present ? group.enabled : true };
  };
  const [motion, setMotion] = useState(groupValues);
  useEffect(() => { setDraft(values()); setError(''); }, [room.id, room.waterSurface, document.original.data, document.original.graphics]);
  useEffect(() => setMotion(groupValues()), [groupId, document.roomSource.waterGroups, document.files.graphics]);
  const apply = (command: () => EditCommand) => { try { edit(command()); setError(''); return true; } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); return false; } };
  const members = document.roomSource.rooms.filter(room => room.waterSurface === undefined ? inheritedWaterGroups(document, room).includes(groupId) : !!room.waterSurface && room.waterSurface.group === groupId);
  const selected = draft.group === 'static' ? null : waterGroupSettings(document, Number(draft.group));
  const name = (room: SourceRoom) => room.originZone === null ? `Room ${room.id.replace(/^room-/, '')}` : `Zone ${room.originZone}`;
  return <section className="property-section"><details onToggle={event => { if (event.target === event.currentTarget) { setOpen(event.currentTarget.open); if (!event.currentTarget.open) setGroupOpen(false); } }}><summary>Water and animation groups</summary>
    {open && <>
    <p className="field-note">Water is a separate plane in the lower storey. It does not move the floor. Keep it still, or assign rooms to one group to make their water rise and fall together. Smaller height values are higher.</p>
    {error && <p className="field-note" role="alert">{error}</p>}
    {conflict && <p className="field-note" role="alert">This imported room has conflicting water groups. Choose one explicit group below to replace its assignments.</p>}
    {group !== null && !configured(group) && <p className="field-note" role="alert">This room's water group is not configured. Configure it below, or choose Static and apply the room water again.</p>}
    {room.waterSurface === false && <p className="field-note">This room's water is removed. Choose a pattern and Apply room water to add it again.</p>}
    <form onSubmit={event => {
      event.preventDefault();
      const group = draft.group === 'static' ? null : Number(draft.group);
      apply(() => setRoomWater(document, room.id, { material: draft.material === 'existing' ? null : Number(draft.material), height: group === null ? Number(draft.height) : Math.floor(waterGroupSettings(document, group).current * 4) / 4, group }));
    }}>
      <label className="select-field"><span>Water coverage and pattern</span><select required aria-label="Water coverage and pattern" value={draft.material} onChange={event => setDraft({ ...draft, material: event.target.value })}>
        <option value="">Choose water coverage</option>{existing && <option value="existing">Keep existing water shapes and patterns</option>}
        {waterMaterialChoices(document, room).map((record, index) => <option key={record.offset} value={record.offset}>Whole room: pattern {index + 1} (from {record.type === 7 ? 'water' : record.type === 1 ? 'floor' : 'ceiling'})</option>)}
      </select></label>
      <p className="field-note">Keeping existing shapes preserves partial pools. A whole-room pattern covers this room's entire outline. Build &amp; play shows the native water effect.</p>
      <label className="select-field"><span>Water movement</span><select aria-label="Room water animation" value={draft.group} onChange={event => setDraft({ ...draft, group: event.target.value })}><option value="static">Static (no animation)</option>{Array.from({ length: 21 }, (_, id) => <option key={id} value={id} disabled={!configured(id)}>Group {id + 1}{configured(id) ? waterGroupSettings(document, id).enabled ? '' : ' (paused)' : ' (configure first)'}</option>)}</select></label>
      {selected ? <p className="field-note">Group {selected.id + 1} starts at {selected.current}, between {selected.top} (top) and {selected.bottom} (bottom). Edit that group's settings below; all assigned rooms share them.</p>
        : <label className="number-field"><span>Static water height</span><input aria-label="Static water height" type="number" required min={-8192} max={8191.75} step={0.25} value={draft.height} onChange={event => setDraft({ ...draft, height: event.target.value })} /></label>}
      <button className="small-button" type="submit">Apply room water</button>
      <button className="small-button" type="button" onClick={() => { setDraft(values()); setError(''); }}>Reset water draft</button>
    </form>
    <button className="small-button" type="button" disabled={room.waterSurface === false} onClick={() => apply(() => setRoomWater(document, room.id, false))}>Remove room water</button>
    <button className="small-button" type="button" disabled={room.waterSurface === undefined} onClick={() => apply(() => setRoomWater(document, room.id, null))}>Inherit template water</button>
    <details onToggle={event => { if (event.target === event.currentTarget) setGroupOpen(event.currentTarget.open); }}><summary>Create or edit water animation group</summary>
      {groupOpen && <>
      <label className="select-field"><span>Animation group</span><select aria-label="Water animation group to edit" value={groupId} onChange={event => setGroupId(Number(event.target.value))}>{Array.from({ length: 21 }, (_, id) => <option key={id} value={id}>Group {id + 1}{configured(id) ? '' : ' (unused)'}</option>)}</select></label>
      <p className="field-note">Choose an unused group to create independent motion. Top and bottom are water heights, not floor heights. Enter a starting height inside that range. Water uses the original game's fixed rise/fall speed. Apply commits the settings; close this group editor to leave a draft unapplied.</p>
      <p className="field-note">After applying a new group, choose it under Water movement above and click Apply room water.</p>
      <form onSubmit={event => { event.preventDefault(); apply(() => setWaterGroup(document, { id: groupId, top: Number(motion.top), bottom: Number(motion.bottom), current: Number(motion.current), velocity: Number(motion.velocity), enabled: motion.enabled })); }}>
        {(['top', 'bottom', 'current'] as const).map(key => <label className="number-field" key={key}><span>{key === 'current' ? 'Starting height' : key === 'top' ? 'Top height' : 'Bottom height'}</span><input required type="number" min={-8192} max={8191.99609375} step={1 / 256} aria-label={`Water group ${key === 'current' ? 'starting' : key} height`} value={motion[key]} onChange={event => setMotion({ ...motion, [key]: event.target.value })} /></label>)}
        <label className="select-field"><span>Starting direction</span><select aria-label="Water group starting direction" value={motion.velocity} onChange={event => setMotion({ ...motion, velocity: event.target.value })}><option value="128">Down towards bottom</option><option value="-128">Up towards top</option>{!['128', '-128'].includes(motion.velocity) && <option value={motion.velocity}>Imported initial velocity {motion.velocity}</option>}</select></label>
        <label><input type="checkbox" aria-label="Animate water group" checked={motion.enabled} onChange={event => setMotion({ ...motion, enabled: event.target.checked })} /> Animate this group</label>
        <p className="field-note">Uncheck to hold every assigned room at the starting height. Rooms stay assigned so you can resume the group later.</p>
        <button type="submit" className="small-button">Apply water group</button><button type="button" className="small-button" onClick={() => { setMotion(groupValues()); setError(''); }}>Reset group draft</button>
      </form>
      <p className="field-note">Assigned rooms: {members.length || 'none'}. Editing the group updates all of them.</p>
      {members.map(room => <button key={room.id} type="button" className="small-button" onClick={() => onInspect(room.id)}>Show water in {name(room)}</button>)}
      </>}
    </details>
    </>}
  </details></section>;
}
