import {expect,test} from '@playwright/test';
import {roomHeightFixture,twoStoreyHeightTemplate} from '../roomHeightFixture';
import {encodeProject,decodeProject} from '../../src/model/project';
import {roomSettings,roomHeightFields} from '../../src/model/roomHeights';
import {spatialEdit} from '../../src/model/editorScene';
import {applyCommand} from '../../src/model/commands';
import {readFile} from 'node:fs/promises';

for(const view of ['perspective','orthographic'])test(`${view}: selected zones share height drags, cancellation and one Undo`,async({page},info)=>{
 const fixture=roomHeightFixture(twoStoreyHeightTemplate());
 const document=applyCommand(fixture.document,spatialEdit(fixture.document,{kind:'height',room:fixture.from.id,field:'floor',value:roomSettings(fixture.document,fixture.from).floor-8}));
 const ids=[fixture.from.id,fixture.to.id],settings=(d:typeof document)=>ids.map(id=>roomSettings(d,d.roomSource.rooms.find(r=>r.id===id)!)),before=settings(document);
 await page.goto('/?view=top');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
 await page.locator('input[type=file]').setInputFiles({name:'group-heights.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(document))});
 for(const [i,id] of ids.entries())await page.getByRole('button',{name:new RegExp(`^Room ${id.replace(/^room-/,'')}`)}).click({modifiers:i?['Shift']:[]});
 await page.getByRole('combobox',{name:'Editing view',exact:true}).selectOption(view);await page.getByRole('button',{name:'Frame selected rooms',exact:true}).click();
 const canvas=page.locator('canvas[aria-label="Editing viewport"]'),camera=await canvas.getAttribute('data-camera');
 const save=async(name:string)=>{const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save project',exact:true}).click();const path=info.outputPath(name+'.ab3d');await(await download).saveAs(path);return decodeProject(new Uint8Array(await readFile(path)));};
 for(const field of roomHeightFields){
  const upper=field.startsWith('upper'),plane=field.toLowerCase().endsWith('floor')?'floor':'ceiling',handle=page.getByRole('button',{name:`Drag ${upper?'upper':'lower'} ${plane} height`,exact:true});
  await expect(handle).toBeVisible();await expect(handle).toHaveAttribute('title',/all 2 selected zones/);
  const r=(await handle.boundingBox())!,drag=async()=>{await page.mouse.move(r.x+r.width/2,r.y+r.height/2);await page.mouse.down();await page.mouse.move(r.x+r.width/2,r.y+r.height/2+(plane==='floor'?-4:4),{steps:4});};
  await drag();await page.keyboard.press('Escape');await page.mouse.up();await expect(page.getByRole('button',{name:'Undo',exact:true})).toBeDisabled();
  await drag();await page.mouse.up();await expect(page.getByRole('alert')).toHaveCount(0);await expect(page.getByRole('heading',{name:'2 selected rooms',exact:true})).toBeVisible();
  const changed=settings(await save(field)),delta=changed[1]![field]-before[1]![field];expect(delta).not.toBe(0);
  for(let i=0;i<2;i++)for(const key of roomHeightFields)expect(changed[i]![key]).toBe(before[i]![key]+(key===field?delta:0));
  await page.getByRole('button',{name:'Undo',exact:true}).click();await expect(page.getByRole('button',{name:'Undo',exact:true})).toBeDisabled();expect(settings(await save(field+'-undo')).map(s=>roomHeightFields.map(k=>s[k]))).toEqual(before.map(s=>roomHeightFields.map(k=>s[k])));
  await expect(canvas).toHaveAttribute('data-camera',camera!);await expect(page.locator('canvas[aria-label="Textured 3D geometry"]')).not.toHaveAttribute('data-material-count','0');
 }
 await page.screenshot({path:info.outputPath(view+'-group-heights.png')});
});
