import type { Page } from '@playwright/test';
import type { RoomProperties } from '../src/model/roomProperties';

export async function engineRoomProperties(page: Page, zone: number): Promise<RoomProperties> {
  return page.evaluate(zone => new Promise((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!, id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine room property query timed out')); }, 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'zone-properties' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive); resolve(event.data.properties);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'zone-properties', id, zone }, location.origin);
  }), zone);
}

export interface EngineCamera { zone: number; x: number; z: number; upper: number; angle: number; height?: number }
export async function engineMechanismTrace(page: Page, kind: 'door' | 'lift', index: number, cycle: boolean): Promise<Awaited<ReturnType<typeof engineMechanismState>>[]> {
  return page.evaluate(({ kind, index, cycle }) => new Promise((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!, id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine mechanism trace timed out')); }, 45000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'mechanism-trace' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive);
      if (event.data.error) reject(new Error(`${event.data.error} Last state: ${JSON.stringify(event.data.samples.at(-1))}`));
      else resolve(event.data.samples);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'mechanism-trace', id, kind, index, cycle }, location.origin);
  }), { kind, index, cycle });
}
export async function engineMechanismState(page: Page, kind: 'door' | 'lift', index: number): Promise<{ zone: number; current: number; top: number; bottom: number; wallCount: number; plane: number; unsynchronizedWalls: number }> {
  return page.evaluate(({ kind, index }) => new Promise((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!, id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine mechanism query timed out')); }, 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'mechanism-state' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive); resolve(event.data.mechanism);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'mechanism-state', id, kind, index }, location.origin);
  }), { kind, index });
}
export async function engineObjectState(page: Page, object: number): Promise<{ id: number; cid: number; zone: number; type: number; x: number; z: number; yFixed: number; health: number }> {
  return page.evaluate(object => new Promise((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!, id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine object query timed out')); }, 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'object-state' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive); resolve(event.data.object);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'object-state', id, object }, location.origin);
  }), object);
}
export async function enginePlayerFeetFixed(page: Page): Promise<number> {
  return page.evaluate(() => new Promise<number>((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!, id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine vertical position query timed out')); }, 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'player-state' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive); resolve(event.data.feetFixed);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'player-state', id }, location.origin);
  }));
}
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

// Keep the position/play-state contract stable as optional inspection fields
// grow. Exact movement assertions must not silently become partial assertions.
export async function enginePlayerState(page: Page): Promise<{ zone: number; x: number; z: number; playing: boolean }> {
  const { zone, x, z, playing } = await queryPlayerSnapshot(page);
  return { zone, x, z, playing };
}

export async function enginePlayerInventory(page: Page): Promise<{ conditions: number; ammunition: number[]; weapons: boolean[] }> {
  const { conditions, ammunition, weapons } = await queryPlayerSnapshot(page);
  return { conditions, ammunition, weapons };
}

async function queryPlayerSnapshot(page: Page): Promise<{ zone: number; x: number; z: number; conditions: number; ammunition: number[]; weapons: boolean[]; playing: boolean }> {
  return page.evaluate(() => new Promise((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!;
    const id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine player query timed out')); }, 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'player-state' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive);
      const { zone, x, z, conditions, ammunition, weapons, playing } = event.data; resolve({ zone, x, z, conditions, ammunition, weapons, playing });
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'player-state', id }, location.origin);
  }));
}
