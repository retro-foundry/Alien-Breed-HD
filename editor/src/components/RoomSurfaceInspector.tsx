import { floorPatternOffsets } from '../model/floorTextures';
import { ceilingIsSky } from '../model/roomStoreys';
import { addRoomPlane, roomPlaneRecords, normalPlaneChoices } from '../model/roomPlanes';
import { editRoomSource } from '../model/commands';
import { useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import type { EditCommand } from '../model/commands';
import { polygonPlane } from '../model/surfaces';
import type { PolygonMaterialTarget } from '../model/polygonMaterials';
import { PolygonPaintDialog } from './PolygonPaintDialog';
import { roomSettings } from '../model/roomHeights';

export function RoomSurfaceInspector({ document, room, edit }: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand) => void }) {
  const [error, setError] = useState('');
  const [target, setTarget] = useState<PolygonMaterialTarget | null>(null);
  const settings = roomSettings(document, room);
  const hasUpper = settings.upper.length > 0 && settings.upperFloor > settings.upperCeiling;
  return <section className="property-section room-surface-controls">
    <h3>Floor and ceiling textures</h3>
    <p className="field-note">Choose a surface below, click a pattern, then Apply surface material. You can edit ceilings here without finding them in 3D. Texture repeat controls how often the pattern tiles.</p>
    {error && <p role="alert" className="field-note">{error}</p>}
    {(hasUpper ? [false, true] : [false]).map(upper => {
      const records = roomPlaneRecords(document, room, upper);
      return <div key={String(upper)}>{(['floor', 'ceiling'] as const).map(plane => {
        const surfaces = records.filter(record => polygonPlane(record.type) === plane);
        const label = `${hasUpper ? upper ? 'Upper ' : 'Lower ' : ''}${plane}`;
        return <div key={plane}>
          {surfaces.map((record, index) => <button type="button" className="small-button" key={record.offset}
            disabled={plane === 'ceiling' && ceilingIsSky(document, room, upper)}
            title={`Choose the texture and repeat for this ${label.toLowerCase()}${surfaces.length > 1 ? ` surface ${index + 1}` : ''}. Apply saves one undoable edit.`}
            onClick={() => setTarget({ room: room.id, upper, record: record.offset })}>Edit {label.toLowerCase()} texture{surfaces.length > 1 ? ` ${index + 1}` : ''}</button>)}
          {!surfaces.length && <label className="select-field"><span>Create {label.toLowerCase()}</span><select aria-label={`Create ${label.toLowerCase()}`} value="" onChange={event => {
            if (!event.target.value) return;
            try { const record = Number(event.target.value); edit(editRoomSource(addRoomPlane(document, room.id, upper, record))); setTarget({ room: room.id, upper, record }); setError(''); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
          }}><option value="">Choose a starting material</option>{normalPlaneChoices(document, plane === 'floor' ? 1 : 2).map(record => <option key={record.offset} value={record.offset}>Pattern {floorPatternOffsets.indexOf(record.texture!) + 1}</option>)}</select></label>}
          {plane === 'ceiling' && ceilingIsSky(document, room, upper) && <p className="field-note">{label} is open to sky. Restore its ceiling below to paint it.</p>}
        </div>;
      })}</div>;
    })}
    {target?.room === room.id && <PolygonPaintDialog key={`${target.room}:${target.upper}:${target.record}`} document={document} target={target} onCancel={() => setTarget(null)} onApply={command => { edit(command); setTarget(null); }} />}
  </section>;
}
