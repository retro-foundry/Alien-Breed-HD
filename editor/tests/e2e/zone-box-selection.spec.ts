import {expect,test} from '@playwright/test';
import {roomHeightFixture} from '../roomHeightFixture';
import {createObject,objectTemplates} from '../../src/model/objectAuthoring';
import {applyCommand} from '../../src/model/commands';
import {encodeProject,decodeProject} from '../../src/model/project';
import {editorScene} from '../../src/model/editorScene';
import {projectPoint,type EditorCamera} from '../../src/model/editorCamera';
import {readFile} from 'node:fs/promises';

for(const view of ['top','orthographic','perspective'])test(`${view}: left gestures select only, Shift adds zones, explicit handles still move`,async({page},info)=>{
 const fixture=roomHeightFixture(),proposal=createObject(fixture.document,{template:objectTemplates(fixture.document).find(o=>o.type===0)!.id,room:fixture.from.id,x:20192,z:20160,y:-8}),document=applyCommand(fixture.document,proposal.command);
 await page.goto('/?view=top');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
 await page.locator('input[type=file]').setInputFiles({name:'selection.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(document))});
 const from=page.getByRole('button',{name:new RegExp(`^Room ${fixture.from.id.replace(/^room-/,'')}`)}),to=page.getByRole('button',{name:new RegExp(`^Room ${fixture.to.id.replace(/^room-/,'')}`)});
 await from.click();await to.click({modifiers:['Shift']});if(view!=='top')await page.getByRole('combobox',{name:'Editing view',exact:true}).selectOption(view);
 await page.getByRole('button',{name:'Frame selected rooms',exact:true}).click();await page.getByRole('button',{name:'Clear room selection',exact:true}).click();
 const canvas=page.locator(view==='top'?'canvas[aria-label="Level map"]':'canvas[aria-label="Editing viewport"]');
 const state=async()=>view==='top'?JSON.stringify(await canvas.evaluate(node=>['data-world-x','data-world-z','data-world-scale'].map(key=>node.getAttribute(key)))):(await canvas.getAttribute('data-camera'))!;
 const before=await state(),bounds=(await canvas.boundingBox())!;
 const point=async(x:number,z:number,y=0)=>{
  if(view==='top'){const v=await canvas.evaluate(n=>({x:Number(n.getAttribute('data-world-x')),z:Number(n.getAttribute('data-world-z')),scale:Number(n.getAttribute('data-world-scale'))}));return{x:bounds.x+bounds.width/2+(x-v.x)*v.scale,y:bounds.y+bounds.height/2-(z-v.z)*v.scale};}
  const camera=JSON.parse((await canvas.getAttribute('data-camera'))!) as EditorCamera,p=projectPoint({x,z,y},camera,bounds.width,bounds.height);return{x:bounds.x+p.x,y:bounds.y+p.y};
 };
 const a=await point(20064,20192),b=await point(21064,20192);
 await page.mouse.click(a.x,a.y);await expect(page.locator('.inspector-heading h2')).toHaveText(`Room ${fixture.from.id.replace(/^room-/,'')}`);expect(await state()).toBe(before);
 await page.keyboard.down('Shift');await page.mouse.click(b.x,b.y);await page.keyboard.up('Shift');await expect(page.getByRole('heading',{name:'2 selected rooms',exact:true})).toBeVisible();expect(await state()).toBe(before);
 const vertices=editorScene(document,'both').faces.filter(f=>f.room===fixture.from.id||f.room===fixture.to.id).flatMap(f=>f.points),projected=await Promise.all(vertices.map(p=>point(p.x,p.z,p.y)));
 const start={x:Math.min(...projected.map(p=>p.x))-12,y:Math.min(...projected.map(p=>p.y))-12},end={x:Math.max(...projected.map(p=>p.x))+12,y:Math.max(...projected.map(p=>p.y))+12};
 for(const reverse of [false,true]){await page.getByRole('button',{name:'Clear room selection',exact:true}).click();const x=reverse?end:start,y=reverse?start:end;await page.mouse.move(x.x,x.y);await page.mouse.down();await page.mouse.move(y.x,y.y,{steps:5});await page.mouse.up();await expect(page.getByRole('heading',{name:'2 selected rooms',exact:true})).toBeVisible();expect(await state()).toBe(before);}
 await page.getByRole('button',{name:'Clear room selection',exact:true}).click();
 const marker=await point(20192,20160,-8);await page.mouse.move(marker.x,marker.y);await page.mouse.down();await page.mouse.move(marker.x+25,marker.y+15,{steps:5});await page.mouse.up();expect(await state()).toBe(before);await expect(page.getByRole('button',{name:'Undo',exact:true})).toBeDisabled();
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save project',exact:true}).click();const path=info.outputPath('selection-only.ab3d');await(await download).saveAs(path);
 const saved=decodeProject(new Uint8Array(await readFile(path)));expect(saved.roomSource).toEqual(document.roomSource);expect(saved.sourceObjects).toEqual(document.sourceObjects);expect(saved.sourceStarts).toEqual(document.sourceStarts);
 await page.mouse.click(marker.x,marker.y);const handle=page.getByRole('button',{name:view==='top'?'Move selection':'Move object X',exact:true});await expect(handle).toBeVisible();const h=(await handle.boundingBox())!;
 if(view==='top'){await page.mouse.move(h.x+h.width/2,h.y+h.height/2);await page.mouse.down();await page.mouse.move(h.x+h.width/2+25,h.y+h.height/2,{steps:5});await page.mouse.up();}else{await handle.focus();await handle.press('ArrowRight');}await expect(page.getByRole('button',{name:'Undo',exact:true})).toBeEnabled();await page.getByRole('button',{name:'Undo',exact:true}).click();expect(await state()).toBe(before);
 await page.mouse.move(bounds.x+40,bounds.y+40);await page.mouse.down({button:'middle'});await page.mouse.move(bounds.x+70,bounds.y+60,{steps:3});await page.mouse.up({button:'middle'});expect(await state()).not.toBe(before);await expect(page.getByRole('button',{name:'Undo',exact:true})).toBeDisabled();
 await page.screenshot({path:info.outputPath(`${view}-left-selection.png`)});
});
