import { expect,test } from '@playwright/test';
import { encodeProject,decodeProject } from '../../src/model/project';
import { roomHeightFixture } from '../roomHeightFixture';
import { texturedEditorScene } from '../../src/model/texturedEditorScene';
import { pickSceneFace } from '../../src/model/editorScene';
import { screenRay,dot,type EditorCamera } from '../../src/model/editorCamera';
import { readFile } from 'node:fs/promises';

test('Walk navigates the editable textured mesh, places objects and keeps its camera through edits',async({page},info)=>{
 const fixture=roomHeightFixture(),mesh=texturedEditorScene(fixture.document);
 await page.goto('/?view=top');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
 await page.locator('input[type=file]').setInputFiles({name:'walk.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(fixture.document))});
 await page.getByRole('combobox',{name:'Editing view',exact:true}).selectOption('perspective');
 const canvas=page.locator('canvas[aria-label="Editing viewport"]');await page.getByRole('button',{name:'Walk',exact:true}).click();
 await expect(page.locator('iframe')).toHaveCount(0);const start=(await canvas.getAttribute('data-camera'))!;
 await canvas.focus();await page.keyboard.down('d');await page.waitForTimeout(150);await page.keyboard.up('d');await expect(canvas).not.toHaveAttribute('data-camera',start);
 const bounds=(await canvas.boundingBox())!;const moved=await canvas.getAttribute('data-camera');
 await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);await page.mouse.down({button:'right'});await page.mouse.move(bounds.x+bounds.width/2+15,bounds.y+bounds.height/2+8,{steps:4});await page.mouse.up({button:'right'});await expect(canvas).not.toHaveAttribute('data-camera',moved!);
 await expect(page.locator('.dirty-dot')).toHaveCount(0);
 const cameraText=(await canvas.getAttribute('data-camera'))!,camera=JSON.parse(cameraText) as EditorCamera;
 await page.getByRole('button',{name:'Add object',exact:true}).click();await page.getByRole('dialog',{name:'Choose object to place'}).getByRole('button',{name:'Blue key',exact:true}).click();
 let point:{x:number;y:number}|undefined;
 for(let y=.6;y<.95 && !point;y+=.05)for(let x=.2;x<.9 && !point;x+=.1){const ray=screenRay(x*bounds.width,y*bounds.height,camera,bounds.width,bounds.height),hit=pickSceneFace(mesh.faces.filter(f=>dot(f.normal,ray.direction)<0),ray);if(hit?.face.kind==='floor')point={x:x*bounds.width,y:y*bounds.height};}
 expect(point).toBeDefined();await canvas.click({position:point!});await page.keyboard.press('Escape');
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save project',exact:true}).click();const path=info.outputPath('placed.ab3d');await(await download).saveAs(path);
 expect(decodeProject(new Uint8Array(await readFile(path))).objects).toHaveLength(fixture.document.objects.length+1);await expect(canvas).toHaveAttribute('data-camera',cameraText);
 await page.getByRole('button',{name:'Undo',exact:true}).click();await expect(canvas).toHaveAttribute('data-camera',cameraText);
 await page.screenshot({path:info.outputPath('walk-editable-textures.png')});
});
