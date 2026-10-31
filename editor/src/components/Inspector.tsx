import { useEffect, useState } from 'react';
import { objectName, type LevelDocument } from '../model/level';
import { moveObject, numberPatch, removeObject, setMechanismRules, setStart, setSourceStart, setSourceObject, setTeleport, setZoneProperty, type EditCommand } from '../model/commands';
import { editableStarts } from '../model/sourceStarts';
import { sourceMatchesLevel } from '../model/roomSource';
import { editableObjects, objectIsActive, objectPlayer } from '../model/sourceObjects';
import { RoomPositionForm } from './RoomPositionForm';
import { TeleportInspector } from './TeleportInspector';
import { RoomAppearanceInspector } from './RoomAppearanceInspector';
import type { Selection } from './MapCanvas';
import { Icon } from './Icon';
import type { SourceRoom } from '../model/roomSource';
import { RoomInspector } from './RoomInspector';
import { NumberField } from './NumberField';

function PositionForm({ x, z, zone, onApply }: { x: number; z: number; zone: number; onApply: (x: number, z: number, zone: number) => void }) {
  const [draft, setDraft] = useState({ x, z, zone });
  useEffect(() => setDraft({ x, z, zone }), [x, z, zone]);
  return <form className="position-form" onSubmit={event => { event.preventDefault(); onApply(draft.x, draft.z, draft.zone); }}>
    <div className="field-grid">{(['x', 'z', 'zone'] as const).map(key => <label className="number-field" key={key}><span>{key === 'zone' ? 'Zone' : key.toUpperCase()}</span><input type="number" required aria-label={`Position ${key.toUpperCase()}`} title={key === 'zone' ? 'Zone containing this position. The coordinates must be inside it. Choose Apply position to commit.' : `${key.toUpperCase()} coordinate on the overhead map. Choose Apply position, or press Enter in this form, to commit.`} value={draft[key]} min={key === 'zone' ? 0 : -32768} max={key === 'zone' ? 255 : 32767} step={1} onChange={event => setDraft({ ...draft, [key]: Number(event.target.value) })} /></label>)}</div>
    <button type="submit" className="small-button" title="Validate and apply all three position fields together. Enter in this form also applies." disabled={draft.x === x && draft.z === z && draft.zone === zone}>Apply position</button>
  </form>;
}

function InfoRow({ label, value }: { label: string; value: string | number }) { return <div className="info-row"><span>{label}</span><span>{value}</span></div>; }

function SourceStartForm({ document, player, edit }: { document: LevelDocument; player: 1 | 2; edit: (command: EditCommand | (() => EditCommand)) => void }) {
  const start = editableStarts(document).find(start => start.player === player)!;
  const room = start.room ?? document.roomSource.rooms.find(room => room.originZone === start.zone)?.id ?? '';
  return <RoomPositionForm rooms={document.roomSource.rooms} identity={`start-${player}`} room={room} x={start.x} z={start.z} label="Player start room"
    onApply={(room, x, z) => edit(() => setSourceStart(document, player, room, x, z))} />;
}

export function Inspector({ document, selection, edit, sourceRoom, onRoomCorner, onRoomDelete, onShowDestination }: { document: LevelDocument; selection: Selection; edit: (command: EditCommand | (() => EditCommand)) => void;
  sourceRoom?: SourceRoom; onRoomCorner: (index: number, axis: 'x' | 'z', value: number) => void; onRoomDelete: () => void; onShowDestination: (room: string) => void }) {
  if (!selection && !sourceRoom) return <aside className="inspector"><div className="panel-title">INSPECTOR</div><div className="empty-inspector"><Icon name="select" size={32} /><h2>Select something</h2><p>Click a zone, object or player start to inspect it.</p><p className="muted">Drag objects and starts to reposition them. Hold Alt to ignore the grid.</p></div></aside>;
  const zone = selection?.kind === 'zone' ? document.zones.find(zone => zone.id === selection.id) : undefined;
  const object = selection?.kind === 'object' ? editableObjects(document).find(object => object.id === selection.id) : undefined;
  const sourceAvatar = object && !sourceMatchesLevel(document) ? objectPlayer(document, object) : undefined;
  const start = selection?.kind === 'start' ? editableStarts(document).find(start => start.player === selection.id) : undefined;
  const point = selection?.kind === 'point' ? document.points[selection.id] : undefined;
  const mechanisms = zone ? document.mechanisms.filter(mechanism => mechanism.zone === zone.id || mechanism.zone === zone.fileId) : [];
  const title = zone ? `Zone ${zone.id}` : object ? objectName(object.type) : start ? `Player ${start.player} start` : point ? `Point ${point.id}` : sourceRoom ? `Room ${sourceRoom.id.replace(/^room-/, '')}` : 'Selection';
  return <aside className="inspector">
    <div className="panel-title">INSPECTOR <span className="eyebrow">{selection?.kind.toUpperCase() ?? 'ROOM'}</span></div>
    <div className="inspector-heading"><span className="selection-symbol"><Icon name={zone || sourceRoom ? 'layers' : start ? 'pin' : 'object'} size={24} /></span><div><h2>{title}</h2><span className="muted">{zone ? `${zone.boundaryIds.length} edges · ${zone.upper.length > 0 ? 'two floors' : 'single floor'}` : object ? `Object ${object.id} · ${object.room === null ? 'Deleted' : object.room ? `Room ${object.room.replace(/^room-/, '')}` : `Zone ${object.zone}`}` : sourceRoom ? 'Editable room outline' : 'Level spawn point'}</span></div></div>
    <div className="inspector-scroll">
      {sourceRoom && <RoomInspector room={sourceRoom} onCorner={onRoomCorner} onDelete={onRoomDelete} />}
      {sourceRoom && <RoomAppearanceInspector document={document} room={sourceRoom} edit={edit} />}
      {sourceRoom && (!zone || !sourceMatchesLevel(document)) && <TeleportInspector document={document} room={sourceRoom} edit={edit} onShowDestination={onShowDestination} />}
      {zone && <>
        <section className="property-section"><h3>Geometry</h3><InfoRow label="Floor height" value={zone.floor} /><InfoRow label="Ceiling height" value={zone.ceiling} /><InfoRow label="Clearance" value={zone.floor - zone.ceiling} />
          {zone.upper.length > 0 && <><InfoRow label="Upper floor" value={zone.upperFloor} /><InfoRow label="Upper ceiling" value={zone.upperCeiling} /></>}
          <InfoRow label="Water height" value={zone.water < zone.floor && zone.water > zone.ceiling ? zone.water : 'Dry'} />
          <InfoRow label="Outdoor sky" value={zone.backdrop ? 'Yes' : 'No'} />
        </section>
        <section className="property-section"><h3>Lighting</h3>
          {(['brightness', 'upperBrightness'] as const).filter(property => property === 'brightness' || zone.upper.length > 0).map(property => {
            const value = zone[property], animation = (value >>> 8) & 255, animated = animation >= 1 && animation <= 3;
            return <div className="light-control" key={property}>
              <label className="select-field"><span>{property === 'brightness' ? 'Lower light' : 'Upper light'}</span><select value={animated ? animation : 0} aria-label={`${property} animation`} title="Choose steady lighting, a pulse, a flicker or a fire effect. Static mode enables the brightness field." onChange={event => edit(() => setZoneProperty(document, zone.id, property, Number(event.target.value) << 8))}><option value={0}>Static</option><option value={1}>Pulse</option><option value={2}>Flicker</option><option value={3}>Fire</option></select></label>
              {!animated && <NumberField help="Lower values make the room brighter; higher values make it darker." label={property === 'brightness' ? 'Brightness' : 'Upper brightness'} value={value} min={-128} max={127} onCommit={value => edit(() => setZoneProperty(document, zone.id, property, value))} />}
            </div>;
          })}
          <p className="field-note">Lower brightness values make the room brighter.</p>
        </section>
        {sourceMatchesLevel(document) && <section className="property-section"><h3>Teleporter</h3>
          <label className="check-field"><input type="checkbox" title="Teleport the player to the destination when stepping into this zone. Apply a destination inside the target zone." checked={zone.teleport.zone >= 0} onChange={event => { const spawn = document.starts[0]!; edit(() => setTeleport(document, zone.id, event.target.checked ? spawn.zone : -1, spawn.x, spawn.z)); }} />Enabled</label>
          {zone.teleport.zone >= 0 && <PositionForm x={zone.teleport.x} z={zone.teleport.z} zone={zone.teleport.zone} onApply={(x, z, target) => edit(() => setTeleport(document, zone.id, target, x, z))} />}
        </section>}
        {mechanisms.map(mechanism => <section className="property-section" key={`${mechanism.kind}-${mechanism.id}`}><h3>{mechanism.kind === 'door' ? 'Door' : 'Lift'} {mechanism.id}</h3>
          <InfoRow label="Closed / bottom" value={mechanism.bottom} /><InfoRow label="Open / top" value={mechanism.top} />
          <label className="select-field"><span>Opens when</span><select aria-label={`${mechanism.kind} ${mechanism.id} opening rule`} title="Choose what starts opening or raising this mechanism. All selected key and switch requirements must also be met." value={mechanism.openRule} onChange={event => edit(() => setMechanismRules(document, mechanism.kind, mechanism.id, mechanism.conditions, Number(event.target.value), mechanism.closeRule))}>
            {(mechanism.kind === 'door' ? ['Player uses it', 'Player touches it', 'Bullet hits it', 'Enemy touches it', 'Timer expires', 'Never'] : ['Player uses it', 'Player touches it', 'Timer expires', 'Never']).map((label, value) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label className="select-field"><span>Closes when</span><select aria-label={`${mechanism.kind} ${mechanism.id} closing rule`} title="Choose what starts closing or lowering this mechanism." value={mechanism.closeRule} onChange={event => edit(() => setMechanismRules(document, mechanism.kind, mechanism.id, mechanism.conditions, mechanism.openRule, Number(event.target.value)))}>
            {(mechanism.kind === 'door' ? ['Timer expires', 'Never'] : ['Player uses it', 'Player touches it', 'Timer expires', 'Never']).map((label, value) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <h4>Requires all selected</h4><div className="condition-grid">{Array.from({ length: 12 }, (_, bit) => <label className="check-field" key={bit}><input type="checkbox" title="This requirement must be met along with every other checked key or switch." checked={(mechanism.conditions & (1 << bit)) !== 0} onChange={event => edit(() => setMechanismRules(document, mechanism.kind, mechanism.id, event.target.checked ? mechanism.conditions | (1 << bit) : mechanism.conditions & ~(1 << bit), mechanism.openRule, mechanism.closeRule))} />{bit < 4 ? `Key ${bit + 1}` : `Switch ${bit - 3}`}</label>)}</div>
        </section>)}
        <section className="property-section"><h3>Visibility</h3><InfoRow label="Visible zones" value={zone.visibleZones.length} /><InfoRow label="Lower graphics" value={zone.lower.length} /><InfoRow label="Upper graphics" value={zone.upper.length} /></section>
      </>}
      {object && <>
        <section className="property-section"><h3>Position</h3>{sourceAvatar !== undefined
          ? <p className="field-note">This avatar follows player {sourceAvatar}. Use Starts in the navigator to change its position.</p> : !sourceMatchesLevel(document)
          ? <RoomPositionForm rooms={document.roomSource.rooms} identity={`object-${object.id}`} room={object.room ?? document.roomSource.rooms.find(room => room.originZone === object.zone)?.id ?? ''} x={object.x} z={object.z} step="any" label="Object room"
            onApply={(room, x, z) => edit(() => setSourceObject(document, object.id, room, x, z))} />
          : <PositionForm x={object.x} z={object.z} zone={object.zone} onApply={(x, z, zone) => edit(() => moveObject(document, object.id, x, z, zone))} />}
          <NumberField help="Vertical object position. Larger values are farther down in the level." label="Height" value={object.y} step={0.25} min={-8192} max={8191.75} onCommit={value => edit({ label: `Change object ${object.id} height`, patches: [numberPatch(document, 'data', object.offset + 4, value * 4, 2)] })} />
          <InfoRow label="Floor" value={object.upper ? 'Upper' : 'Lower'} />
        </section>
        <section className="property-section"><h3>Object</h3><InfoRow label="Type" value={objectName(object.type)} /><InfoRow label="Collision ID" value={object.cid} />
          {[0, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19].includes(object.type) && <NumberField label="Health" value={object.health} min={0} max={127} onCommit={value => edit({ label: `Change object ${object.id} health`, patches: [numberPatch(document, 'data', object.offset + 18, value, 1)] })} />}
        </section>
        <section className="property-section"><button className="danger-button" disabled={!objectIsActive(object) || sourceAvatar !== undefined} title={sourceAvatar !== undefined ? 'Player avatars are controlled through Starts in the navigator.' : 'Remove this object. Undo restores it (Ctrl / Command + Z).'} onClick={() => edit(() => removeObject(document, object.id))}>Delete object</button></section>
      </>}
      {start && <section className="property-section"><h3>Position</h3>{!sourceMatchesLevel(document)
        ? <SourceStartForm document={document} player={start.player} edit={edit} />
        : <PositionForm x={start.x} z={start.z} zone={start.zone} onApply={(x, z, zone) => edit(() => setStart(document, start.player, x, z, zone))} />}<p className="field-note">Drag the triangle on the map to place this start. The point must stay inside its room.</p></section>}
      {point && <section className="property-section"><h3>Coordinates</h3><InfoRow label="X" value={point.x} /><InfoRow label="Z" value={point.z} /></section>}
    </div>
  </aside>;
}
