import { useEffect, useRef, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { applyCommand, editRoomSource, type EditCommand } from '../model/commands';
import { setSurfaceMaterial, surfaceMaterialCurrent, hasSurfaceMaterialOverride, wallMaterialChoices, type SurfaceMaterialTarget } from '../model/wallMaterials';
import { setWallSettings, wallSettingsCurrent, type WallSettingsValues } from '../model/wallSettings';
import { wallRepeatWidths, type WallRepeat } from '../model/wallRepeat';
import { buildPlayableLevel } from '../model/playableBuild';
import { WallStyleBrowser } from './WallStyleBrowser';
import './ObjectBuilder.css';

export function WallPaintDialog({ document, target, onCancel, onApply }: {
  document: LevelDocument; target: SurfaceMaterialTarget; onCancel: () => void; onApply: (command: EditCommand) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null), choices = wallMaterialChoices(document);
  const room = document.roomSource.rooms.find(room => room.id === target.room)!;
  const previous = hasSurfaceMaterialOverride(document, target);
  const current = surfaceMaterialCurrent(document, target)!;
  const initialStyle = String(choices.find(item => item.wall!.texture === current.texture
    && item.wall!.verticalMask === current.verticalMask && item.wall!.verticalShift === current.verticalShift && item.wall!.horizontalMask === current.horizontalMask)?.offset ?? '');
  const [style, setStyle] = useState(initialStyle);
  const settings = wallSettingsCurrent(document, target);
  const [brightness, setBrightness] = useState(settings?.brightness === undefined ? '' : String(settings.brightness));
  const [horizontal, setHorizontal] = useState(settings?.horizontalOffset === undefined ? '' : String(settings.horizontalOffset));
  const [vertical, setVertical] = useState(settings?.verticalOffset === undefined ? '' : String(settings.verticalOffset));
  const [repeatWidth, setRepeatWidth] = useState(settings?.repeat ? String(settings.repeat.width) : '');
  const [repeatFit, setRepeatFit] = useState<WallRepeat['fit']>(settings?.repeat?.fit ?? 'natural');
  const [error, setError] = useState('');
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  const commit = (mode: 'apply' | 'reset-material' | 'reset-settings') => {
    try {
      let working = document;
      if (mode === 'apply' || mode === 'reset-settings') {
        const values: WallSettingsValues = { ...(brightness !== '' ? { brightness: Number(brightness) } : {}),
          ...(horizontal !== '' ? { horizontalOffset: Number(horizontal) } : {}), ...(vertical !== '' ? { verticalOffset: Number(vertical) } : {}),
          ...(repeatWidth !== '' ? { repeat: { width: Number(repeatWidth), fit: repeatFit } } : {}) };
        working = applyCommand(working, editRoomSource(setWallSettings(working, target, mode === 'reset-settings' || !Object.keys(values).length ? null : values)));
      }
      if (mode === 'reset-material' || mode === 'apply' && style !== initialStyle)
        working = applyCommand(working, editRoomSource(setSurfaceMaterial(working, target, mode === 'reset-material' ? null : Number(style))));
      buildPlayableLevel(working, -1);
      onApply(editRoomSource({ label: mode === 'reset-material' ? 'Reset wall material' : mode === 'reset-settings' ? 'Reset wall settings' : 'Edit wall material and settings', before: document.roomSource, after: working.roomSource }));
    }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  };
  return <dialog ref={dialog} className="modal object-builder wall-paint-dialog" aria-label="Paint wall material" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>Paint this {target.upper ? 'upper' : 'lower'} wall</h2>
    <p>Changes this {'index' in target ? 'authored boundary' : 'source face'} in Room {room.id.replace(/^room-/, '')}, including any pieces made when the room is split or a doorway cuts it. Other rooms and copied faces keep their own materials.</p>
    <p className="field-note">Choose a texture, repeat, alignment or brightness for just this wall. Heights, transparency and movement stay authored. Cancel or Escape discards the draft; Apply commits one Undo.</p>
    <form onSubmit={event => { event.preventDefault(); commit('apply'); }}>
      <label className="select-field"><span>Wall material style</span><select aria-label="Picked wall material style" value={style}
        title="Choose native sheet sampling for only this source face. Geometry, alignment, lighting and animation remain unchanged. Apply commits one undoable edit."
        onChange={event => { setStyle(event.target.value); setError(''); }}><option value="" disabled>Choose a native material</option>
        {choices.map((record, index) => <option key={record.offset} value={record.offset}>Style {index + 1} · Material {record.texture}</option>)}</select></label>
      <WallStyleBrowser records={choices} value={style} onChange={setStyle} label="picked wall" overrides={document.wallOverrides}
        description="Choose a native sheet and its sampling. Apply the form to commit." />
      <details><summary>Texture repeat</summary>
        <label className="select-field"><span>Pattern width</span><select aria-label="Wall repeat width" value={repeatWidth} onChange={event => setRepeatWidth(event.target.value)}>
          <option value="">Inherit original repeat</option>{wallRepeatWidths.map(width => <option key={width} value={width}>{width} texture columns</option>)}
        </select></label>
        <label className="select-field"><span>Fit to whole wall</span><select aria-label="Wall repeat fit" value={repeatFit} disabled={!repeatWidth} onChange={event => setRepeatFit(event.target.value as WallRepeat['fit'])}>
          <option value="natural">Natural size</option><option value="up">Next full repeat (round up)</option><option value="down">Full repeats only (round down)</option>
        </select></label>
        <p className="field-note">Pattern width chooses how many texture columns repeat. Natural size uses one column per two map units. Round up includes the next full repeat, even on an exact fit; round down keeps at least one full repeat. Wall pieces share one continuous pattern. Choose Inherit original repeat to clear only this repeat override.</p>
      </details>
      <details><summary>Alignment and brightness</summary>
        <p className="field-note">Leave a field empty to inherit the source value. Smaller brightness offsets make this wall brighter; larger values make it darker. This adds to its existing corner lighting.</p>
        <label className="select-field"><span>Brightness offset</span><input aria-label="Wall brightness offset" type="number" min={-32768} max={32767} step={1} value={brightness} placeholder="Inherit" onChange={event => setBrightness(event.target.value)} /></label>
        <label className="select-field"><span>Horizontal tile offset</span><input aria-label="Wall horizontal tile offset" type="number" min={-32768} max={32767} step={1} value={horizontal} placeholder="Inherit" onChange={event => setHorizontal(event.target.value)} /></label>
        <p className="field-note">One horizontal tile moves the starting texture column by 16. The chosen sheet must contain the full repeated strip.</p>
        <label className="select-field"><span>Vertical row offset</span><input aria-label="Wall vertical row offset" type="number" min={-32768} max={32767} step={1} value={vertical} placeholder="Inherit" onChange={event => setVertical(event.target.value)} /></label>
        <p className="field-note">One vertical unit moves the texture by one row; the pattern wraps. Moving walls retain this starting alignment through their native animation.</p>
      </details>
      {error && <p role="alert">{error}</p>}
      <div className="object-builder-actions"><button type="button" onClick={onCancel} title="Discard this material draft. Escape also cancels.">Cancel wall paint</button>
        <button type="button" disabled={!previous} onClick={() => commit('reset-material')} title="Remove only this face's material override. Saved repeat, alignment and brightness remain; draft edits are discarded.">Reset wall material</button>
        <button type="button" disabled={!settings} onClick={() => commit('reset-settings')} title="Restore inherited repeat, alignment and brightness, keeping the saved texture material. Draft edits are discarded.">Reset wall settings</button>
        <button type="submit" className="primary-button" disabled={!style && brightness === '' && horizontal === '' && vertical === '' && repeatWidth === ''} title="Apply this face's material, repeat, alignment and brightness as one undoable edit.">Apply wall material</button></div>
    </form>
  </dialog>;
}
