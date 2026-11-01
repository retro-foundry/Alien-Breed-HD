import { useEffect, useRef, useState } from 'react';
import { pngDimensions } from '../model/png';
import { convertWallRgba } from '../model/wallImage';
import type { WallSheet, WallTexture } from '../model/wallTextures';
import { WallSheetImage } from './WallSheetImage';
import type { WallOverride } from '../model/wallOverrides';

interface Draft { bytes: Uint8Array; source: Uint8Array; width: number; height: number; changedPixels: number; fitted: Uint8ClampedArray }
function FittedImage({ pixels, width, height }: { pixels: Uint8ClampedArray; width: number; height: number }) {
  const canvas = useRef<HTMLCanvasElement>(null), [error, setError] = useState('');
  useEffect(() => {
    try {
      const context = canvas.current!.getContext('2d'); if (!context) throw new Error('The browser cannot draw the PNG preview.');
      const image = context.createImageData(width, height); image.data.set(pixels); context.putImageData(image, 0, 0); setError('');
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  }, [pixels, width, height]);
  return <><canvas ref={canvas} width={width} height={height} role="img" aria-label="Fitted PNG source" />{error && <span role="alert">{error}</span>}</>;
}
export function PngWallImporter({ sheet, texture, label, baseline, override, onChange }: {
  sheet: WallSheet; texture: WallTexture; label: string; baseline: Uint8Array;
  override?: WallOverride;
  onChange: (texture: number, bytes: Uint8Array | null, sourcePng?: Uint8Array) => void;
}) {
  const request = useRef(0), [draft, setDraft] = useState<Draft>(), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => { setDraft(undefined); setError(''); setBusy(false); return () => { ++request.current; }; }, [baseline, texture.bytes, override, sheet.id]);
  const sourcePng = override?.sourcePng;
  const cancel = () => { ++request.current; setDraft(undefined); setError(''); setBusy(false); };
  return <div className="png-wall-importer" onKeyDown={event => { if (event.key === 'Escape' && (draft || busy || error)) { event.preventDefault(); event.stopPropagation(); cancel(); } }}>
    <p className="field-note">Import a PNG to this shared material. It is resized automatically to {sheet.width} × {sheet.rows} and matched to this sheet's colours; all 32 lighting palettes stay intact. Transparent pixels blend with its background colour. Review the converted image before applying. The original PNG is kept in your project.</p>
    <label className="select-field"><span>Import PNG image</span><input type="file" accept="image/png,.png" aria-label={`Import ${label} PNG`}
      title="Choose a PNG. Preview resizes it and matches its native palette. Apply PNG creates one undoable shared-sheet edit; Cancel or Escape keeps the current material."
      onChange={async event => {
        const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; if (!file) return;
        const id = ++request.current; setError(''); setDraft(undefined); setBusy(true);
        try {
          if (file.size > 16 * 1024 * 1024) throw new Error('Choose a PNG smaller than 16 MiB.');
          const source = new Uint8Array(await file.arrayBuffer()), dimensions = pngDimensions(source);
          const bitmap = await createImageBitmap(new Blob([new Uint8Array(source)], { type: 'image/png' }), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
          if (id !== request.current) { bitmap.close(); return; }
          let converted, fitted;
          try {
            const canvas = document.createElement('canvas'); canvas.width = sheet.width; canvas.height = sheet.rows;
            const context = canvas.getContext('2d'); if (!context) throw new Error('The browser cannot convert PNG images.');
            context.imageSmoothingEnabled = false; context.drawImage(bitmap, 0, 0, sheet.width, sheet.rows);
            fitted = context.getImageData(0, 0, sheet.width, sheet.rows).data;
            converted = convertWallRgba(texture, fitted);
          } finally { bitmap.close(); }
          if (id === request.current) setDraft({ ...converted, fitted, source, ...dimensions });
        } catch (error) { if (id === request.current) setError(error instanceof Error ? error.message : String(error)); }
        finally { if (id === request.current) setBusy(false); }
      }} /></label>
    {busy && <p role="status">Converting PNG…</p>}
    {error && <p role="alert">{error}</p>}
    {draft && <div className="png-import-preview" role="group" aria-label={`${label} PNG conversion preview`}>
      <div className="png-comparison"><figure><figcaption>Fitted PNG</figcaption><FittedImage pixels={draft.fitted} width={sheet.width} height={sheet.rows} /></figure>
        <figure><figcaption>Native conversion</figcaption><WallSheetImage sheet={sheet} texture={{ ...texture, bytes: draft.bytes }} shade={0} /></figure></div>
      <p role="status">{draft.width} × {draft.height} → {sheet.width} × {sheet.rows}. {draft.changedPixels} of {sheet.width * sheet.rows} pixels adjusted for native colour and opacity. Applying changes every wall using this material.</p>
      <button type="button" title="Apply this converted sheet and keep the original PNG as one undoable project asset edit." onClick={() => {
        try { onChange(sheet.id, draft.bytes, draft.source); setDraft(undefined); setError(''); }
        catch (error) { setError(error instanceof Error ? error.message : String(error)); }
      }}>Apply PNG</button>
    </div>}
    {(draft || busy || error) && <button type="button" title="Discard this PNG conversion. The current project material stays intact." onClick={cancel}>Cancel PNG</button>}
    {sourcePng && <button type="button" title="Download the unchanged original PNG kept with this material. It can be edited and imported again." onClick={() => {
      const url = URL.createObjectURL(new Blob([new Uint8Array(sourcePng)], { type: 'image/png' })), link = document.createElement('a');
      link.href = url; link.download = `${sheet.name.replace(/\.wad$/i, '')}-source.png`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 60000);
    }}>Download original PNG</button>}
    {override && <button type="button" title="Restore the packaged original sheet as one undoable edit. All walls using this material update together; Undo brings back your imported image."
      onClick={() => onChange(sheet.id, null)}>Restore original sheet</button>}
  </div>;
}
