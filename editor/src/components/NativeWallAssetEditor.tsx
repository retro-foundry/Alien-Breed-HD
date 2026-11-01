import { useEffect, useRef, useState } from 'react';
import type { WallSheet } from '../model/wallTextures';
import { readWallTexture } from '../model/wallTextures';
import type { WallOverride } from '../model/wallOverrides';

export function NativeWallAssetEditor({ sheet, label, baseline, overrides, onChange }: {
  sheet: WallSheet; label: string; baseline: Uint8Array; overrides?: readonly WallOverride[];
  onChange: (texture: number, bytes: Uint8Array | null) => void;
}) {
  const request = useRef(0), [error, setError] = useState('');
  useEffect(() => { setError(''); return () => { ++request.current; }; }, [baseline, overrides, sheet.id]);
  return <details className="wall-native-tools"><summary title="Advanced native wall asset replacement. This shared sheet is saved with the project and copied into game exports.">Native sheet tools</summary>
    <p className="field-note">{sheet.name}: {sheet.width} × {sheet.rows}. Import immediately replaces this shared sheet for every wall using it, as one undoable edit. Apply appearance controls the chosen wall style separately.</p>
    <label className="select-field"><span>Import native wall sheet</span><input type="file" accept=".wad" aria-label={`Import ${label} native sheet`}
      title="Replace this shared material with an existing native WAD of the same dimensions. Packed SB files are decoded automatically; source level files stay intact."
      onChange={async event => {
        const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; if (!file) return;
        const id = ++request.current; setError('');
        try {
          if (!/\.wad$/i.test(file.name) || file.size > 16 * 1024 * 1024) throw new Error('Choose a native .wad wall sheet smaller than 16 MiB.');
          const texture = readWallTexture(new Uint8Array(await file.arrayBuffer()), sheet);
          if (id === request.current) onChange(sheet.id, texture.bytes);
        } catch (error) { if (id === request.current) setError(error instanceof Error ? error.message : String(error)); }
      }} /></label>
    {error && <p role="alert">{error}</p>}
  </details>;
}
