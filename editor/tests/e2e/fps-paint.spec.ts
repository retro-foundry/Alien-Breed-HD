import { expect, test } from '@playwright/test';
import { enginePickDetails, engineInspectionCamera } from '../engineCapture';

test('paint directly in default textured FPS editor without moving its camera', async({page})=>{
  await page.goto('/');
  const preview=page.getByRole('region',{name:'3D editing view',exact:true});
  await expect(preview.getByRole('status')).toContainText('zones loaded',{timeout:60000});
  const points=Array.from({length:81},(_,i)=>({x:(i%9+1)/10,y:(Math.floor(i/9)+1)/10}));
  const {x,z,y,angle,look}=await engineInspectionCamera(page);
  const before=await enginePickDetails(page,points), index=before.findIndex(p=>p.surface?.type===1 && !p.object);
  expect(index).toBeGreaterThanOrEqual(0);
  const canvas=page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), bounds=(await canvas.boundingBox())!;
  await canvas.click({position:{x:points[index]!.x*bounds.width,y:points[index]!.y*bounds.height}});
  await page.getByRole('button',{name:'Paint this surface',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Paint floor or ceiling material'}), pattern=dialog.getByRole('combobox',{name:'Surface pattern',exact:true});
  await pattern.selectOption((await pattern.inputValue())==='771'?'0':'771');
  const revision=(await preview.getAttribute('data-revision'))!;
  await dialog.getByRole('button',{name:'Apply surface material',exact:true}).click();
  await expect(preview).not.toHaveAttribute('data-revision',revision); await expect(preview.getByRole('status')).toContainText('zones loaded');
  const after=await enginePickDetails(page,points);
  expect(await engineInspectionCamera(page)).toMatchObject({x,z,y,angle,look});
  expect(after.map(p=>p.pixel)).not.toEqual(before.map(p=>p.pixel));
  const next=(await preview.getAttribute('data-revision'))!;
  await page.getByRole('button',{name:'Undo',exact:true}).click();
  await expect(preview).not.toHaveAttribute('data-revision',next); await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect((await enginePickDetails(page,points)).map(p=>p.pixel)).toEqual(before.map(p=>p.pixel));
});
