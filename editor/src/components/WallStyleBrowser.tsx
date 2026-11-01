import { useEffect, useRef, useState } from 'react';
import type { Graphic } from '../model/graphics';
import { loadWallCatalog, loadWallTexture } from '../model/wallAssets';
import { wallTextureRgba, type WallSheet, type WallTexture } from '../model/wallTextures';
import './WallStyleBrowser.css';

function SheetImage({ sheet, texture, shade }: { sheet: WallSheet; texture: WallTexture; shade: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    try {
      const context = canvas.current!.getContext('2d');
      if (!context) throw new Error('The browser cannot draw wall material thumbnails.');
      const pixels = context.createImageData(texture.width, texture.height); pixels.data.set(wallTextureRgba(texture, shade));
      context.putImageData(pixels, 0, 0); setError('');
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  }, [texture, shade]);
  return <><canvas ref={canvas} width={texture.width} height={texture.height} role="img"
    aria-label={`${sheet.name} texture sheet, ${texture.width} by ${texture.height}, shade ${shade}`} />{error && <span role="alert">{error}</span>}</>;
}

function Browser({ records, value, onChange, label }: { records: readonly Graphic[]; value: string; onChange: (value: string) => void; label: string }) {
  const [assets, setAssets] = useState<ReadonlyMap<number, { sheet: WallSheet; texture: WallTexture }>>();
  const [error, setError] = useState(''), [retry, setRetry] = useState(0), [query, setQuery] = useState(''), [shade, setShade] = useState(16);
  const ids = [...new Set(records.map(record => record.texture))].sort((a, b) => a! - b!).join(',');
  useEffect(() => {
    let active = true; setError(''); setAssets(undefined);
    void loadWallCatalog().then(async catalog => {
      const result = await Promise.all(ids.split(',').filter(Boolean).map(async id => {
        const sheet = catalog.find(sheet => sheet.id === Number(id));
        if (!sheet) throw new Error(`Wall material ${id} is not in the packaged native texture catalog.`);
        return [sheet.id, { sheet, texture: await loadWallTexture(sheet) }] as const;
      }));
      if (active) setAssets(new Map(result));
    }).catch(error => { if (active) setError(error instanceof Error ? error.message : String(error)); });
    return () => { active = false; };
  }, [ids, retry]);
  return <div className="wall-browser">
    <p className="field-note">Whole texture sheets at the chosen preview shade. Each style keeps its authored alignment, sampling and transparency. Choose a tile, then apply the form.</p>
    <label className="select-field"><span>Find material or style</span><input aria-label={`${label} material search`} value={query} title="Filter this template's wall styles by material name, texture number or style number." onChange={event => setQuery(event.target.value)} /></label>
    <label className="select-field"><span>Preview shade {shade}</span><input type="range" min="0" max="31" value={shade} aria-label={`${label} preview shade`} title="Preview the texture's authored brightness palettes. This changes thumbnails only; saved room lighting is chosen separately." onChange={event => setShade(Number(event.target.value))} /></label>
    {error ? <div role="alert"><p>{error}</p><button type="button" title="Retry loading the real material assets. Your selected styles and project are kept." onClick={() => setRetry(value => value + 1)}>Retry materials</button></div>
      : !assets ? <p role="status">Loading native wall materials…</p> : <div className="wall-style-grid" role="group" aria-label={`${label} material choices`}>
        {records.map((record, index) => ({ record, index, asset: assets.get(record.texture!)! })).filter(({ record, index, asset }) =>
          `${asset.sheet.name} ${record.texture} style ${index + 1}`.toLowerCase().includes(query.toLowerCase().trim())).map(({ record, index, asset }) =>
          <button type="button" className="wall-style-tile" key={record.offset} aria-pressed={value === String(record.offset)}
            aria-label={`Choose ${label} style ${index + 1}: ${asset.sheet.name}`} title="Select this authored wall style. Apply the form to commit it as an undoable edit." onClick={() => onChange(String(record.offset))}>
            <SheetImage {...asset} shade={shade} /><span>{asset.sheet.name.replace(/\.wad$/i, '')}</span><small>Style {index + 1} · Texture {record.texture}</small>
          </button>)}
      </div>}
    {assets && !records.some((record, index) => `${assets.get(record.texture!)!.sheet.name} ${record.texture} style ${index + 1}`.toLowerCase().includes(query.toLowerCase().trim())) && <p role="status">No matching wall styles.</p>}
  </div>;
}

export function WallStyleBrowser(props: { records: readonly Graphic[]; value: string; onChange: (value: string) => void; label: string }) {
  const [open, setOpen] = useState(false);
  return <details className="wall-style-browser" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary title="Browse this template's wall styles using the original texture sheets and palettes. Enter or Space opens the browser.">Browse {props.label} materials</summary>
    {open && <Browser {...props} />}
  </details>;
}
