import type { Page } from '@playwright/test';

export interface EngineCamera { zone: number; x: number; z: number; upper: number; angle: number }
export async function captureEngine(page: Page, camera: EngineCamera): Promise<number> {
  return page.evaluate(camera => new Promise<number>((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!;
    const id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine frame capture timed out')); }, 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'captured' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive); resolve(event.data.checksum);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'capture', id, camera }, location.origin);
  }), camera);
}
