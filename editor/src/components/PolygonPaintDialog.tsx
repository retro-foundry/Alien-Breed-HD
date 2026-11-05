import { roomPlaneRecords } from '../model/roomPlanes';
import { polygonPlane } from '../model/surfaces';
import type { PolygonMaterialMode } from '../model/surfaces';
import { useEffect, useRef, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { editRoomSource, type EditCommand } from '../model/commands';
import { polygonMaterialCurrent, polygonMaterialMode, setPolygonMaterial, type PolygonMaterial, type PolygonMaterialTarget } from '../model/polygonMaterials';
import { floorPaletteLevels, floorPatternOffsets, floorRepeatScales, floorTextureRgba, loadFloorTextures, type FloorTextures } from '../model/floorTextures';
import './ObjectBuilder.css';

function Pattern({ assets, texture, shade, label }: { assets: FloorTextures; texture: number; shade: number; label: string }) {
  const canvas = useRef<HTMLCanvasElement>(null), [error, setError] = useState('');
  useEffect(() => {
    try {
      const context = canvas.current!.getContext('2d');
      if (!context) throw new Error('The browser cannot draw floor pattern previews.');
      const pixels = context.createImageData(64, 64); pixels.data.set(floorTextureRgba(assets, texture, shade));
      context.putImageData(pixels, 0, 0); setError('');
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  }, [assets, texture, shade]);
  return <><canvas ref={canvas} width={64} height={64} role="img" aria-label={`${label}, preview shade ${shade}`} />{error && <span role="alert">{error}</span>}</>;
}

export function PolygonPaintDialog({ document, target, onCancel, onApply }: {
  document: LevelDocument; target: PolygonMaterialTarget; onCancel: () => void; onApply: (command: EditCommand) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null), current = polygonMaterialCurrent(document, target)!;
  const room = document.roomSource.rooms.find(room => room.id === target.room)!;
  const record = roomPlaneRecords(document, room, target.upper).find(record => record.offset === target.record)!;
  const surfaceName = `${target.upper ? 'Upper' : 'Lower'} ${polygonPlane(record.type)}`;
  const previous = room.polygonMaterials?.some(item => item.record === target.record && item.upper === target.upper);
  const [mode, setMode] = useState<PolygonMaterialMode>(() => polygonMaterialMode(document, target));
  const [texture, setTexture] = useState(mode === 'smooth' ? 0 : current.texture), [scale, setScale] = useState(current.scale), [shade, setShade] = useState(0);
  const [assets, setAssets] = useState<FloorTextures>(), [error, setError] = useState(''), [retry, setRetry] = useState(0);
  const patterns = mode === 'smooth' ? [0] : floorPatternOffsets.includes(texture) ? floorPatternOffsets : [...floorPatternOffsets, texture];
  const scales = floorRepeatScales.includes(current.scale) ? floorRepeatScales : [...floorRepeatScales, current.scale];
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  useEffect(() => {
    let active = true; setError(''); setAssets(undefined);
    loadFloorTextures(mode).then(assets => { if (active) setAssets(assets); }).catch(error => { if (active) setError(String(error.message ?? error)); });
    return () => { active = false; };
  }, [mode, retry]);
  const apply = (value: Pick<PolygonMaterial, 'texture' | 'scale' | 'mode'> | null) => {
    try { onApply(editRoomSource(setPolygonMaterial(document, target, value))); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  };
  return <dialog ref={dialog} className="modal object-builder wall-paint-dialog" aria-label="Paint floor or ceiling material" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>{surfaceName} texture</h2>
    <p>Choose how this floor or ceiling looks in Room {room.id.replace(/^room-/, '')}, then pick a pattern below. Apply changes the selected surface.</p>
    <p className="field-note">Use Normal pattern for floors and ceilings. The unused bump choices are disabled. Existing bump materials are kept when you open or save a map.</p>
    <form onSubmit={event => { event.preventDefault(); apply({ texture, scale, mode }); }}>
      <label className="select-field"><span>Material mode</span><select aria-label="Surface material mode" value={mode} title="Use Normal pattern. Unused bump choices are disabled; existing saved bump materials are preserved." onChange={event => { setMode(event.target.value as PolygonMaterialMode); setTexture(0); }}><option value="normal">Normal pattern</option><option value="bump" disabled>Chunky bump pattern (unused)</option><option value="smooth" disabled>Smooth bump texture (unused)</option></select></label>
      <p className="field-note" role="status">{mode === 'normal'
        ? 'Normal pattern: a standard textured surface, shaded using lighting points. Start here for ordinary floors and ceilings.'
        : mode === 'bump'
          ? 'This surface already uses chunky bump. Keep its saved material or choose Normal pattern to replace it.'
          : 'This surface already uses smooth bump with one fixed texture. Keep its saved material or choose Normal pattern to replace it.'}</p>
      {mode !== 'smooth' && <label className="select-field"><span>Pattern</span><select aria-label="Surface pattern" value={texture} title="Choose an original pattern from the selected bank for only the picked surface." onChange={event => setTexture(Number(event.target.value))}>
        {patterns.map((value, index) => <option key={value} value={value}>{index < 16 ? `Pattern ${index + 1}` : 'Keep imported pattern'}</option>)}
      </select></label>}
      <label className="select-field"><span>Texture repeat</span><select aria-label="Surface texture repeat" value={scale} title="Higher repeat factors make the pattern smaller and repeat more often. The original editor supports 1/32× through 4×." onChange={event => setScale(Number(event.target.value))}>
        {scales.map(value => <option key={value} value={value}>{value < -5 || value > 2 ? 'Keep imported repeat' : value < 0 ? `1/${2 ** -value}×` : `${2 ** value}×${value === 0 ? ' (native default)' : ''}`}</option>)}
      </select></label>
      <label className="select-field"><span>Preview shade {shade}</span><input aria-label="Floor preview shade" type="range" min={0} max={floorPaletteLevels - 1} value={shade} title="Show the original palette shades without changing the room lighting." onChange={event => setShade(Number(event.target.value))} /></label>
      <p className="field-note">Thumbnails show texture colours at the chosen preview shade. Inspect the applied surface in Engine view to see its in-game effect. Apply is one undoable edit; Cancel or Escape discards your choices.</p>
      {assets ? <div className="wall-style-grid" role="group" aria-label="Floor pattern choices">{patterns.map((value, index) => <button type="button" className="wall-style-tile" key={value}
        aria-pressed={mode === 'smooth' || texture === value} aria-label={`Choose surface pattern ${index + 1}`} title="Select this pattern as a draft. Apply commits the surface edit." onClick={() => setTexture(value)}>
        <Pattern assets={assets} texture={mode === 'smooth' ? 0 : value} shade={shade} label={mode === 'smooth' ? 'Smooth bump texture' : index < 16 ? `Pattern ${index + 1}` : 'Imported pattern'} /><span>{mode === 'smooth' ? 'Smooth bump texture' : index < 16 ? `Pattern ${index + 1}` : 'Imported pattern'}</span>
      </button>)}</div> : !error && <p role="status">Loading native floor patterns…</p>}
      {error && <div role="alert"><p>{error}</p><button type="button" onClick={() => setRetry(value => value + 1)}>Retry floor materials</button></div>}
      <div className="object-builder-actions"><button type="button" onClick={onCancel} title="Discard the pattern and repeat draft. Escape also cancels.">Cancel surface paint</button>
        <button type="button" disabled={!previous} onClick={() => apply(null)} title="Restore only this surface's original pattern and repeat as one undoable action.">Reset surface material</button>
        <button type="submit" className="primary-button" title="Apply the chosen pattern and repeat to this source surface as one undoable edit.">Apply surface material</button></div>
    </form>
  </dialog>;
}
