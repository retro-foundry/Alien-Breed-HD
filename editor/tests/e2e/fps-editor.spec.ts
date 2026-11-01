import { expect, test, type Page } from '@playwright/test';
import { enginePickDetails } from '../engineCapture';

async function camera(page: Page): Promise<{x:number;z:number;y:number;angle:number;look:number}> {
  return page.evaluate(() => new Promise((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!, id = crypto.randomUUID();
    const timer = setTimeout(() => reject(new Error('Camera query timed out')), 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.data?.type !== 'camera-state' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive); resolve(event.data.camera);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({source:'ab3d-editor',type:'camera-state',id},location.origin);
  }));
}

test('default opens textured FPS editing, moves and looks, places objects and preserves camera through Undo', async ({page}, testInfo) => {
  test.setTimeout(90000);
  await page.goto('/');
  await expect(page.getByRole('combobox',{name:'Editing view',exact:true})).toHaveValue('perspective');
  const preview = page.getByRole('region',{name:'3D editing view',exact:true});
  await expect(preview.getByRole('status')).toContainText('zones loaded',{timeout:60000});
  expect(Number(await preview.getAttribute('data-rendered-pixels'))).toBeGreaterThan(1000);
  await expect(page.locator('canvas[aria-label="Editing viewport"]')).toHaveCount(0);
  const canvas=page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  const before=await camera(page);
  await canvas.focus(); await page.keyboard.down('d'); await page.waitForTimeout(180); await page.keyboard.up('d');
  const moved=await camera(page); expect(Math.hypot(moved.x-before.x,moved.z-before.z)).toBeGreaterThan(1);
  const bounds=(await canvas.boundingBox())!;
  await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2); await page.mouse.down({button:'right'});
  await page.mouse.move(bounds.x+bounds.width/2+25,bounds.y+bounds.height/2+15,{steps:5}); await page.mouse.up({button:'right'});
  await expect.poll(async()=> (await camera(page)).angle).not.toBe(moved.angle);
  await canvas.focus(); await page.keyboard.press('Escape');
  await page.getByRole('button',{name:'Add object',exact:true}).click();
  await page.getByRole('dialog',{name:'Choose object to place'}).getByRole('button',{name:'Blue key',exact:true}).click();
  const points=Array.from({length:81},(_,i)=>({x:(i%9+1)/10,y:(Math.floor(i/9)+1)/10}));
  const picks=await enginePickDetails(page,points), index=picks.findIndex(p=>p.surface?.type===1 && !p.object);
  expect(index).toBeGreaterThanOrEqual(0);
  const position=await camera(page), rev=await preview.getAttribute('data-revision');
  await canvas.click({position:{x:points[index]!.x*bounds.width,y:points[index]!.y*bounds.height}});
  await expect(preview).not.toHaveAttribute('data-revision',rev!);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await camera(page)).toEqual(position);
  await canvas.focus(); await page.keyboard.press('Escape');
  await expect(page.getByRole('button',{name:'Cancel placement',exact:true})).toHaveCount(0);
  const changed=await preview.getAttribute('data-revision');
  await page.getByRole('button',{name:'Undo',exact:true}).click();
  await expect(preview).not.toHaveAttribute('data-revision',changed!);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await camera(page)).toEqual(position);
  await page.screenshot({path:testInfo.outputPath('default-textured-fps.png')});
  await preview.getByRole('button',{name:'Playtest',exact:true}).click();
  const engine=page.getByRole('region',{name:'3D game engine preview',exact:true});
  await engine.getByRole('button',{name:'Build & play',exact:true}).click();
  await expect(engine.getByRole('status')).toHaveText('Playtest running');
  await page.getByRole('button',{name:'Editor view',exact:true}).click();
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  await expect(preview.getByRole('button',{name:'Playtest',exact:true})).toBeVisible();
});
