import { useEffect, useRef, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { editRoomSource, type EditCommand } from '../model/commands';
import { setWallMaterial, wallMaterialChoices, type WallMaterialTarget } from '../model/wallMaterials';
import { WallStyleBrowser } from './WallStyleBrowser';
import './ObjectBuilder.css';

export function WallPaintDialog({ document, target, onCancel, onApply }: {
  document: LevelDocument; target: WallMaterialTarget; onCancel: () => void; onApply: (command: EditCommand) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null), choices = wallMaterialChoices(document);
  const room = document.roomSource.rooms.find(room => room.id === target.room)!;
  const template = document.zones.find(zone => zone.id === room.recordZone)!;
  const previous = room.wallMaterials?.find(item => item.upper === target.upper && item.record === target.record);
  const current = previous ?? (target.upper ? template.upper : template.lower).find(item => item.offset === target.record)!.wall!;
  const [style, setStyle] = useState(String(choices.find(item => item.wall!.texture === current.texture
    && item.wall!.verticalMask === current.verticalMask && item.wall!.verticalShift === current.verticalShift && item.wall!.horizontalMask === current.horizontalMask)?.offset ?? ''));
  const [error, setError] = useState('');
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  const apply = (value: number | null) => {
    try { onApply(editRoomSource(setWallMaterial(document, target, value))); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  };
  return <dialog ref={dialog} className="modal object-builder wall-paint-dialog" aria-label="Paint wall material" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>Paint this {target.upper ? 'upper' : 'lower'} wall</h2>
    <p>Changes this source face in Room {room.id.replace(/^room-/, '')}, including any pieces made when the room is split or a doorway cuts it. Other rooms and copied faces keep their own materials.</p>
    <p className="field-note">Keeps the wall's alignment, strip span, height, brightness, transparency and movement. Switch control graphics keep their native material. Cancel or Escape discards the draft; one Undo restores a committed paint.</p>
    <form onSubmit={event => { event.preventDefault(); if (style) apply(Number(style)); }}>
      <label className="select-field"><span>Wall material style</span><select aria-label="Picked wall material style" value={style}
        title="Choose native sheet sampling for only this source face. Geometry, alignment, lighting and animation remain unchanged. Apply commits one undoable edit."
        onChange={event => { setStyle(event.target.value); setError(''); }}><option value="" disabled>Choose a native material</option>
        {choices.map((record, index) => <option key={record.offset} value={record.offset}>Style {index + 1} · Material {record.texture}</option>)}</select></label>
      <WallStyleBrowser records={choices} value={style} onChange={setStyle} label="picked wall" overrides={document.wallOverrides}
        description="Choose a native sheet and its sampling. The selected wall keeps its own alignment, height, brightness and transparency. Apply the form to commit." />
      {error && <p role="alert">{error}</p>}
      <div className="object-builder-actions"><button type="button" onClick={onCancel} title="Discard this material draft. Escape also cancels.">Cancel wall paint</button>
        <button type="button" disabled={!previous} onClick={() => apply(null)} title="Remove only this face's material override. Uses its source material again, with one Undo.">Reset wall material</button>
        <button type="submit" className="primary-button" disabled={!style} title="Paint this source face and its generated pieces as one undoable edit.">Apply wall material</button></div>
    </form>
  </dialog>;
}
