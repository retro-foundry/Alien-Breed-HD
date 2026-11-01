import { expect,test } from '@playwright/test';
import { encodeProject } from '../../src/model/project';
import { roomHeightFixture,twoStoreyHeightTemplate } from '../roomHeightFixture';
import { texturedEditorScene } from '../../src/model/texturedEditorScene';
import { pickSceneFace } from '../../src/model/editorScene';
import { screenRay,dot,type EditorCamera } from '../../src/model/editorCamera';

test('paint walls and floors directly in the editable textured 3D view with portable Undo',async({page},info)=>{
 const fixture=roomHeightFixture(twoStoreyHeightTemplate()),mesh=texturedEditorScene(fixture.document);
 await page.goto('/?view=top');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
 await page.locator('input[type=file]').setInputFiles({name:'paint.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(fixture.document))});
 await page.getByRole('button',{name:new RegExp(`^Room ${fixture.from.id.replace(/^room-/,'')}`)}).click();await page.getByRole('button',{name:'Paint room textures',exact:true}).click();
 const canvas=page.locator('canvas[aria-label="Editing viewport"]'),texture=page.locator('canvas[aria-label="Textured 3D geometry"]');
 await expect(canvas).toHaveAttribute('data-view','perspective');await expect(page.locator('iframe')).toHaveCount(0);await expect(texture).not.toHaveAttribute('data-material-count','0');
 const hash=()=>texture.evaluate(node=>{const c=node as HTMLCanvasElement,g=c.getContext('webgl2')!,b=new Uint8Array(c.width*c.height*4);g.readPixels(0,0,c.width,c.height,g.RGBA,g.UNSIGNED_BYTE,b);let h=0;for(const n of b)h=(Math.imul(h,31)+n)>>>0;return h;});
 const cameraText=(await canvas.getAttribute('data-camera'))!,camera=JSON.parse(cameraText) as EditorCamera,bounds=(await canvas.boundingBox())!;
 for(const kind of ['floor','wall'] as const){
  let point:{x:number;y:number}|undefined;
  for(let y=.2;y<.85 && !point;y+=.025)for(let x=.2;x<.85 && !point;x+=.025){const ray=screenRay(x*bounds.width,y*bounds.height,camera,bounds.width,bounds.height),hit=pickSceneFace(mesh.faces.filter(f=>dot(f.normal,ray.direction)<0),ray);if(hit?.face.kind===kind && hit.face.room===fixture.from.id)point={x:x*bounds.width,y:y*bounds.height};}
  expect(point).toBeDefined();const before=await hash();await canvas.click({position:point!});
  await page.getByRole('button',{name:kind==='wall'?'Paint this wall':'Paint this surface',exact:true}).click();
  if(kind==='floor'){const dialog=page.getByRole('dialog',{name:'Paint floor or ceiling material'}),pattern=dialog.getByRole('combobox',{name:'Surface pattern',exact:true});await pattern.selectOption((await pattern.inputValue())==='771'?'0':'771');await dialog.getByRole('button',{name:'Apply surface material',exact:true}).click();}
  else{const dialog=page.getByRole('dialog',{name:'Paint wall material'}),styles=dialog.getByRole('combobox',{name:'Picked wall material style'}),value=await styles.inputValue(),options=await styles.locator('option').evaluateAll(options=>options.map(o=>(o as HTMLOptionElement).value).filter(Boolean));await styles.selectOption(options.find(v=>v!==value)!);await dialog.getByRole('button',{name:'Apply wall material',exact:true}).click();}
  await expect.poll(hash).not.toBe(before);await expect(canvas).toHaveAttribute('data-camera',cameraText);await page.getByRole('button',{name:'Undo',exact:true}).click();await expect.poll(hash).toBe(before);
 }
 await page.screenshot({path:info.outputPath('paint-editable-geometry.png')});
});
