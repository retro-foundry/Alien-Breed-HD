import { useEffect, useRef, useState } from 'react';
import { wallTextureRgba, type WallSheet, type WallTexture } from '../model/wallTextures';

export function WallSheetImage({ sheet, texture, shade }: { sheet: WallSheet; texture: WallTexture; shade: number }) {
  const canvas = useRef<HTMLCanvasElement>(null), [error, setError] = useState('');
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
