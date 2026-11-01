import { useEffect, useRef, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { editRoomSource, type EditCommand } from '../model/commands';
import { polygonMaterialCurrent, setPolygonMaterial, type PolygonMaterial, type PolygonMaterialTarget } from '../model/polygonMaterials';
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
  const previous = room.polygonMaterials?.some(item => item.record === target.record && item.upper === target.upper);
  const [texture, setTexture] = useState(current.texture), [scale, setScale] = useState(current.scale), [shade, setShade] = useState(0);
  const [assets, setAssets] = useState<FloorTextures>(), [error, setError] = useState(''), [retry, setRetry] = useState(0);
  const patterns = floorPatternOffsets.includes(current.texture) ? floorPatternOffsets : [...floorPatternOffsets, current.texture];
  const scales = floorRepeatScales.includes(current.scale) ? floorRepeatScales : [...floorRepeatScales, current.scale];
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  useEffect(() => {
    let active = true; setError('');
    loadFloorTextures().then(assets => { if (active) setAssets(assets); }).catch(error => { if (active) setError(String(error.message ?? error)); });
    return () => { active = false; };
  }, [retry]);
  const apply = (value: Pick<PolygonMaterial, 'texture' | 'scale'> | null) => {
    try { onApply(editRoomSource(setPolygonMaterial(document, target, value))); }
    catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  };
  return <dialog ref={dialog} className="modal object-builder wall-paint-dialog" aria-label="Paint floor or ceiling material" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>Paint floor / ceiling pattern</h2>
    <p>Choose one of the original 16 normal patterns and its world-aligned repeat. Changes this source surface and its compiled pieces in Room {room.id.replace(/^room-/, '')}. Lower and upper surfaces, other rooms and copies keep their own settings.</p>
    <p className="field-note">Keeps height, shape, brightness, surface type and moving-controller links. Preview shade changes only the thumbnails. Apply is one undoable edit; Cancel or Escape discards the draft.</p>
    <form onSubmit={event => { event.preventDefault(); apply({ texture, scale }); }}>
      <label className="select-field"><span>Pattern</span><select aria-label="Surface pattern" value={texture} title="Choose an original floor-bank pattern for only the picked surface." onChange={event => setTexture(Number(event.target.value))}>
        {patterns.map((value, index) => <option key={value} value={value}>{index < 16 ? `Pattern ${index + 1}` : 'Keep imported pattern'}</option>)}
      </select></label>
      <label className="select-field"><span>Texture repeat</span><select aria-label="Surface texture repeat" value={scale} title="Higher repeat factors make the pattern smaller and repeat more often. The original editor supports 1/32× through 4×." onChange={event => setScale(Number(event.target.value))}>
        {scales.map(value => <option key={value} value={value}>{value < -5 || value > 2 ? 'Keep imported repeat' : value < 0 ? `1/${2 ** -value}×` : `${2 ** value}×${value === 0 ? ' (native default)' : ''}`}</option>)}
      </select></label>
      <label className="select-field"><span>Preview shade {shade}</span><input aria-label="Floor preview shade" type="range" min={0} max={floorPaletteLevels - 1} value={shade} title="Show the original palette shades without changing the room lighting." onChange={event => setShade(Number(event.target.value))} /></label>
      {assets ? <div className="wall-style-grid" role="group" aria-label="Floor pattern choices">{patterns.map((value, index) => <button type="button" className="wall-style-tile" key={value}
        aria-pressed={texture === value} aria-label={`Choose surface pattern ${index + 1}`} title="Select this pattern as a draft. Apply commits the surface edit." onClick={() => setTexture(value)}>
        <Pattern assets={assets} texture={value} shade={shade} label={index < 16 ? `Pattern ${index + 1}` : 'Imported pattern'} /><span>{index < 16 ? `Pattern ${index + 1}` : 'Imported pattern'}</span>
      </button>)}</div> : !error && <p role="status">Loading native floor patterns…</p>}
      {error && <div role="alert"><p>{error}</p><button type="button" onClick={() => setRetry(value => value + 1)}>Retry floor materials</button></div>}
      <div className="object-builder-actions"><button type="button" onClick={onCancel} title="Discard the pattern and repeat draft. Escape also cancels.">Cancel surface paint</button>
        <button type="button" disabled={!previous} onClick={() => apply(null)} title="Restore only this surface's original pattern and repeat as one undoable action.">Reset surface material</button>
        <button type="submit" className="primary-button" title="Apply the chosen pattern and repeat to this source surface as one undoable edit.">Apply surface material</button></div>
    </form>
  </dialog>;
}
