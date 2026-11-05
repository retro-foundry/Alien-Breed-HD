import { RetireMechanism } from './RetireMechanism';
import { WaterInspector } from './WaterInspector';
import { CornerLightInspector } from './CornerLightInspector';
import { waterStartHeight } from '../model/sourceWater';
import { NavigationInspector } from './NavigationInspector';
import { RoomPortalInspector } from './RoomPortalInspector';
import { RoomSurfaceInspector } from './RoomSurfaceInspector';
import { ceilingIsSky } from '../model/roomStoreys';
import { AuthoredMechanismInspector } from './AuthoredMechanismInspector';
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
import { RoomHeightInspector } from './RoomHeightInspector';
import { RoomPropertiesInspector } from './RoomPropertiesInspector';
import { roomSettings } from '../model/roomHeights';
import { NumberField } from './NumberField';
import { MechanismRangeInspector } from './MechanismRangeInspector';
import { MechanismAttachments } from './MechanismAttachments';
import { MechanismRulesInspector } from './MechanismRulesInspector';
import { CopiedMechanismInspector } from './CopiedMechanismInspector';
import { CopyMechanismSettings } from './MechanismTransfers';
import { SwitchesInspector } from './SwitchesInspector';
import { mechanismsForRoom } from '../model/mechanismDependencies';
import { ObjectTypeInspector } from './ObjectTypeInspector';
import { objectTemplates } from '../model/objectAuthoring';
import { keyColour, keyColours, setKeyColour, ammunitionChoices, ammunitionType, setAmmunitionType, weaponPickupChoices, weaponPickupType, setWeaponPickupType } from '../model/objectSettings';

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

export function Inspector({ document, selection, edit, sourceRoom, onObjectDuplicate, onRoomCorner, onRoomCornerInsert, onRoomCornerRemove, onRoomRedraw, onRoomDelete, onShowDestination, onInspectRoom, onViewRoomSide, onPaintRoomTextures }: { document: LevelDocument; selection: Selection; edit: (command: EditCommand | (() => EditCommand)) => void;
  onObjectDuplicate: () => void; onViewRoomSide: () => void; onPaintRoomTextures: () => void;
  sourceRoom?: SourceRoom; onRoomCorner: (index: number, axis: 'x' | 'z', value: number) => void; onRoomCornerInsert: (index: number) => void; onRoomCornerRemove: (index: number) => void; onRoomRedraw: () => void; onRoomDelete: () => void; onShowDestination: (room: string) => void; onInspectRoom: (room: string) => void }) {
  if (!selection && !sourceRoom) return <aside className="inspector"><div className="panel-title">INSPECTOR</div><div className="empty-inspector"><Icon name="select" size={32} /><h2>Select something</h2><p>Click a zone, object or player start to inspect it.</p><p className="muted">Drag objects and starts to reposition them. Hold Alt to ignore the grid.</p></div></aside>;
  const zone = selection?.kind === 'zone' ? document.zones.find(zone => zone.id === selection.id) : undefined;
  const dimensions = sourceRoom ? roomSettings(document, sourceRoom) : zone;
  let waterHeight: number | string | undefined = zone?.water;
  if (sourceRoom) {
    try { waterHeight = waterStartHeight(document, sourceRoom); }
    catch (reason) { waterHeight = reason instanceof Error ? reason.message : String(reason); }
  }
  const outdoorSky = sourceRoom ? ceilingIsSky(document, sourceRoom, !!dimensions?.upper.length && dimensions.upperFloor > dimensions.upperCeiling) : zone?.backdrop;
  const object = selection?.kind === 'object' ? editableObjects(document).find(object => object.id === selection.id) : undefined;
  const sourceAvatar = object && !sourceMatchesLevel(document) ? objectPlayer(document, object) : undefined;
  const start = selection?.kind === 'start' ? editableStarts(document).find(start => start.player === selection.id) : undefined;
  const point = selection?.kind === 'point' ? document.points[selection.id] : undefined;
  const mechanisms = zone ? mechanismsForRoom(document, zone.id) : [];
  const title = zone ? `Zone ${zone.id}` : object ? objectName(object.type) : start ? `Player ${start.player} start` : point ? `Point ${point.id}` : sourceRoom ? `Room ${sourceRoom.id.replace(/^room-/, '')}` : 'Selection';
  return <aside className="inspector">
    <div className="panel-title">INSPECTOR <span className="eyebrow">{selection?.kind.toUpperCase() ?? 'ROOM'}</span></div>
    <div className="inspector-heading"><span className="selection-symbol"><Icon name={zone || sourceRoom ? 'layers' : start ? 'pin' : 'object'} size={24} /></span><div><h2>{title}</h2><span className="muted">{zone ? `${zone.boundaryIds.length} edges · ${dimensions!.upper.length > 0 && dimensions!.upperFloor > dimensions!.upperCeiling ? 'two storeys' : 'one storey'}` : object ? `Object ${object.id} · ${object.room === null ? 'Deleted' : object.room ? `Room ${object.room.replace(/^room-/, '')}` : `Zone ${object.zone}`}` : sourceRoom ? 'Editable room outline' : 'Level spawn point'}</span></div></div>
    <div className="inspector-scroll">
      {(zone || sourceRoom) && <p className="field-note">A zone is a map area with its own heights and settings. It can be a whole room, a corridor, or part of a larger room, such as a step. One visible room can contain several zones.</p>}
      {sourceRoom && <RoomSurfaceInspector key={`surfaces:${sourceRoom.id}`} document={document} room={sourceRoom} edit={edit} />}
      {sourceRoom && <RoomInspector room={sourceRoom} onCorner={onRoomCorner} onInsertCorner={onRoomCornerInsert} onRemoveCorner={onRoomCornerRemove} onRedraw={onRoomRedraw} onDelete={onRoomDelete} />}
      {sourceRoom && <CornerLightInspector key={`corner-lights:${sourceRoom.id}`} document={document} room={sourceRoom} edit={edit} />}
      {sourceRoom && <RoomPortalInspector key={`portals:${sourceRoom.id}`} document={document} room={sourceRoom} edit={edit} onInspect={onInspectRoom} />}
      {sourceRoom && <NavigationInspector key={`navigation:${sourceRoom.id}`} document={document} room={sourceRoom} edit={edit} />}
      {sourceRoom && <RoomAppearanceInspector document={document} room={sourceRoom} edit={edit} onPaintTextures={onPaintRoomTextures} />}
      {sourceRoom && <RoomHeightInspector document={document} room={sourceRoom} edit={edit} onViewSide={onViewRoomSide} />}
      {sourceRoom && <WaterInspector key={`water:${sourceRoom.id}`} document={document} room={sourceRoom} edit={edit} onInspect={onInspectRoom} />}
      {sourceRoom && <AuthoredMechanismInspector document={document} room={sourceRoom} edit={edit} onShowRoom={onInspectRoom} />}
      {sourceRoom && <CopiedMechanismInspector document={document} room={sourceRoom} edit={edit} onShowRoom={onInspectRoom} />}
      <SwitchesInspector document={document} room={sourceRoom} zone={zone?.id} edit={edit} frameRoom={onShowDestination} />
      {sourceRoom && (!zone || sourceRoom.properties) && <RoomPropertiesInspector document={document} room={sourceRoom} edit={edit} />}
      {sourceRoom && (!zone || !sourceMatchesLevel(document)) && <TeleportInspector document={document} room={sourceRoom} edit={edit} onShowDestination={onShowDestination} />}
      {zone && <>
        <section className="property-section"><h3>Geometry</h3><InfoRow label="Floor height" value={dimensions!.floor} /><InfoRow label="Ceiling height" value={dimensions!.ceiling} /><InfoRow label="Clearance" value={dimensions!.floor - dimensions!.ceiling} />
          {dimensions!.upper.length > 0 && <><InfoRow label="Upper floor" value={dimensions!.upperFloor} /><InfoRow label="Upper ceiling" value={dimensions!.upperCeiling} /></>}
          <InfoRow label="Water height" value={typeof waterHeight === 'string' ? waterHeight : waterHeight !== undefined && waterHeight < dimensions!.floor ? waterHeight : 'Dry'} />
          <InfoRow label="Outdoor sky" value={outdoorSky ? 'Yes' : 'No'} />
        </section>
        {!sourceRoom?.properties && <section className="property-section"><h3>Lighting</h3>
          {(['brightness', 'upperBrightness'] as const).filter(property => property === 'brightness' || zone.upper.length > 0).map(property => {
            const value = zone[property], animation = (value >>> 8) & 255, animated = animation >= 1 && animation <= 3;
            return <div className="light-control" key={property}>
              <label className="select-field"><span>{property === 'brightness' ? 'Lower light' : 'Upper light'}</span><select value={animated ? animation : 0} aria-label={`${property} animation`} title="Choose steady lighting, a pulse, a flicker or a fire effect. Static mode enables the brightness field." onChange={event => edit(() => setZoneProperty(document, zone.id, property, Number(event.target.value) << 8))}><option value={0}>Static</option><option value={1}>Pulse</option><option value={2}>Flicker</option><option value={3}>Fire</option></select></label>
              {!animated && <NumberField help="Lower values make the room brighter; higher values make it darker." label={property === 'brightness' ? 'Brightness' : 'Upper brightness'} value={value} min={-128} max={127} onCommit={value => edit(() => setZoneProperty(document, zone.id, property, value))} />}
            </div>;
          })}
          <p className="field-note">Lower brightness values make the room brighter.</p>
        </section>}
        {sourceMatchesLevel(document) && <section className="property-section"><h3>Teleporter</h3>
          <label className="check-field"><input type="checkbox" title="Teleport the player to the destination when stepping into this zone. Apply a destination inside the target zone." checked={zone.teleport.zone >= 0} onChange={event => { const spawn = document.starts[0]!; edit(() => setTeleport(document, zone.id, event.target.checked ? spawn.zone : -1, spawn.x, spawn.z)); }} />Enabled</label>
          {zone.teleport.zone >= 0 && <PositionForm x={zone.teleport.x} z={zone.teleport.z} zone={zone.teleport.zone} onApply={(x, z, target) => edit(() => setTeleport(document, zone.id, target, x, z))} />}
        </section>}
        {mechanisms.map(mechanism => <section className="property-section" key={`${mechanism.kind}-${mechanism.id}`}><h3>{mechanism.kind === 'door' ? 'Door' : 'Lift'} {mechanism.id}</h3>
          <InfoRow label="Closed / bottom" value={mechanism.bottom} /><InfoRow label="Open / top" value={mechanism.top} />
          <MechanismRangeInspector document={document} mechanism={mechanism} edit={edit} />
          <MechanismRulesInspector mechanism={mechanism} prefix={`${mechanism.kind} ${mechanism.id}`}
            change={(conditions, openRule, closeRule) => edit(() => setMechanismRules(document, mechanism.kind, mechanism.id, conditions, openRule, closeRule))} />
          <MechanismAttachments document={document} mechanism={mechanism} onShowRoom={onShowDestination} />
          <CopyMechanismSettings document={document} owner={{ type: 'native', kind: mechanism.kind, id: mechanism.id }} edit={edit} />
          <RetireMechanism document={document} owner={{ type: 'native', kind: mechanism.kind, id: mechanism.id }} edit={edit} onShowRoom={onInspectRoom} />
        </section>)}
        <section className="property-section"><h3>Visibility</h3><InfoRow label="Visible zones" value={zone.visibleZones.length} /><InfoRow label="Lower graphics" value={zone.lower.length} /><InfoRow label="Upper graphics" value={zone.upper.length} /></section>
      </>}
      {object && <>
        <section className="property-section"><ObjectTypeInspector document={document} id={object.id} edit={edit} />
          {object.type === 3 && <><label className="select-field"><span>Weapon pickup</span><select aria-label="Weapon pickup" value={weaponPickupType(document, object.id) ?? ''}
            title="Choose the weapon granted by this pickup. Updates its artwork, dimensions and grant together, preserving its floor-relative anchor. Undo restores the complete change."
            onChange={event => edit(() => setWeaponPickupType(document, object.id, Number(event.target.value)))}>
            <option value="" disabled>Custom artwork or weapon type</option>{weaponPickupChoices.map(choice => <option key={choice.id} value={choice.id}>{choice.name}</option>)}
          </select></label><p className="field-note">Keeps the pickup's floor position when changing its size. Custom records stay unchanged until you choose a weapon.</p></>}
          {object.type === 9 && <><label className="select-field"><span>Ammunition for</span><select aria-label="Ammunition weapon" value={ammunitionType(document, object.id) ?? ''}
            title="Choose which weapon receives this pickup. Updates its native ammunition type and artwork together. The game controls how much ammunition it grants."
            onChange={event => edit(() => setAmmunitionType(document, object.id, Number(event.target.value)))}>
            <option value="" disabled>Custom artwork or ammunition type</option>{ammunitionChoices.map(choice => <option key={choice.id} value={choice.id}>{choice.name}</option>)}
          </select></label><p className="field-note">Pickup amount follows the game's weapon data. Custom records stay unchanged until you choose a weapon.</p></>}
          {object.type === 4 && <><label className="select-field"><span>Key colour</span><select aria-label="Key colour" value={keyColour(document, object.id) ?? ''}
            title="Change the key artwork and matching door requirement together. Green is Key 1, Red Key 2, Yellow Key 3 and Blue Key 4. Undo restores both."
            onChange={event => edit(() => setKeyColour(document, object.id, Number(event.target.value)))}>
            <option value="" disabled>Custom artwork or condition bits</option>{keyColours.map((colour, index) => <option key={colour} value={index}>{colour} · Key {index + 1}</option>)}
          </select></label><p className="field-note">Unlocks mechanisms requiring the matching key. Custom records stay unchanged until you choose a colour.</p></>}
          {[0, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19].includes(object.type) && <NumberField label="Health" value={object.health} min={0} max={127} onCommit={value => edit({ label: `Change object ${object.id} health`, patches: [numberPatch(document, 'data', object.offset + 18, value, 1)] })} />}
        </section>
        <section className="property-section"><h3>Position</h3>{sourceAvatar !== undefined
          ? <p className="field-note">This avatar follows player {sourceAvatar}. Use Starts in the navigator to change its position.</p> : !sourceMatchesLevel(document)
          ? <RoomPositionForm rooms={document.roomSource.rooms} identity={`object-${object.id}`} room={object.room ?? document.roomSource.rooms.find(room => room.originZone === object.zone)?.id ?? ''} x={object.x} z={object.z} step="any" label="Object room"
            onApply={(room, x, z) => edit(() => setSourceObject(document, object.id, room, x, z))} />
          : <PositionForm x={object.x} z={object.z} zone={object.zone} onApply={(x, z, zone) => edit(() => moveObject(document, object.id, x, z, zone))} />}
          <NumberField help="Vertical object position in world units, using half-unit steps. Larger values are farther down in the level." label="Height" value={object.y} step={0.5} min={-16384} max={16383.5} onCommit={value => edit({ label: `Change object ${object.id} height`, patches: [numberPatch(document, 'data', object.offset + 4, value * 2, 2)] })} />
          <InfoRow label="Floor" value={object.upper ? 'Upper' : 'Lower'} />
        </section>
        <section className="property-section object-actions"><button disabled={!objectIsActive(object) || !objectTemplates(document).some(template => template.id === object.id)} title="Copy this object's settings, then click the map to place it. Its height offset follows the chosen floor. Shortcut: Ctrl / Command + D. Escape cancels." onClick={onObjectDuplicate} aria-label="Duplicate object">Duplicate</button>
          <button className="danger-button" disabled={!objectIsActive(object) || objectPlayer(document, object) !== undefined} title={objectPlayer(document, object) !== undefined ? 'Player avatars are controlled through Starts in the navigator.' : 'Remove this object. Shortcut: Delete. Undo restores it (Ctrl / Command + Z).'} onClick={() => edit(() => removeObject(document, object.id))} aria-label="Delete object">Delete</button></section>
      </>}
      {start && <section className="property-section"><h3>Position</h3>{!sourceMatchesLevel(document)
        ? <SourceStartForm document={document} player={start.player} edit={edit} />
        : <PositionForm x={start.x} z={start.z} zone={start.zone} onApply={(x, z, zone) => edit(() => setStart(document, start.player, x, z, zone))} />}<p className="field-note">Select the triangle, then drag its movement handle to place this start. The point must stay inside its room.</p></section>}
      {point && <section className="property-section"><h3>Coordinates</h3><InfoRow label="X" value={point.x} /><InfoRow label="Z" value={point.z} /></section>}
    </div>
  </aside>;
}
