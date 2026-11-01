import { cameraBasis, type EditorCamera } from './editorCamera';
import type { TexturedFace } from './texturedEditorScene';
import type { LevelDocument } from './level';
import { loadWallCatalog, loadWallTexture } from './wallAssets';
import { loadFloorTextures } from './floorTextures';
import type { PolygonMaterialMode } from './surfaces';

export interface IndexedMaterial { pixels: Uint8Array; width: number; height: number; palette: Uint8Array; colours: number; shades: number; rows: number }
function paletteRgba(bytes: Uint8Array): Uint8Array {
  const result=new Uint8Array(bytes.length*2);
  for(let i=0;i<bytes.length;i+=2){const rgb=bytes[i]!*256+bytes[i+1]!, at=i*2;result[at]=((rgb>>8)&15)*17;result[at+1]=((rgb>>4)&15)*17;result[at+2]=(rgb&15)*17;result[at+3]=255;}
  return result;
}
export async function loadSpatialMaterials(faces: readonly TexturedFace[], document: LevelDocument): Promise<ReadonlyMap<string,IndexedMaterial>> {
  const keys=[...new Set(faces.map(face=>face.material))], catalog=keys.some(k=>k.startsWith('wall:'))?await loadWallCatalog():[];
  const entries=await Promise.all(keys.map(async key=>{
    const [kind,id,offset]=key.split(':');
    if(kind==='wall'){
      const sheet=catalog.find(s=>s.id===Number(id));if(!sheet)throw new Error(`Wall sheet ${id} is missing from the editor assets.`);
      const texture=await loadWallTexture(sheet,document.wallOverrides), words=(texture.bytes.length-2048)/2, width=256,height=Math.ceil(words*3/width), pixels=new Uint8Array(width*height);
      for(let i=0;i<words;i++){const word=texture.bytes[2048+i*2]!*256+texture.bytes[2049+i*2]!;for(let j=0;j<3;j++)pixels[i*3+j]=(word>>(j*5))&31;}
      return [key,{pixels,width,height,palette:paletteRgba(texture.bytes.subarray(0,2048)),colours:32,shades:32,rows:sheet.rows}] as const;
    }
    const bank=await loadFloorTextures(id as PolygonMaterialMode), pixels=new Uint8Array(4096), texture=Number(offset);
    for(let y=0;y<64;y++)for(let x=0;x<64;x++)pixels[y*64+x]=bank.pixels[texture+y*1024+x*4]!;
    return [key,{pixels,width:64,height:64,palette:paletteRgba(bank.palette),colours:256,shades:15,rows:64}] as const;
  }));
  return new Map<string,IndexedMaterial>(entries);
}
const vertexSource=`#version 300 es
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec2 uv;
layout(location=2) in float brightness;
uniform vec3 eye, rightAxis, upAxis, backAxis;
uniform vec2 viewport;
uniform float distanceToTarget, perspective;
out vec2 texcoord;
out float lightValue;
out float viewDepth;
void main(){
 vec3 delta=position-eye;float depth=-dot(delta,backAxis);float w=mix(distanceToTarget,depth,perspective);
 float focal=min(viewport.x,viewport.y)*.9;float farPlane=distanceToTarget+200000.;
 float z=mix(w*((depth-.1)*2./(farPlane-.1)-1.),depth*(farPlane+.1)/(farPlane-.1)-(.2*farPlane)/(farPlane-.1),perspective);
 gl_Position=vec4(dot(delta,rightAxis)*focal*2./viewport.x,dot(delta,upAxis)*focal*2./viewport.y,z,w);
 texcoord=uv;lightValue=brightness;viewDepth=depth;
}`;
const fragmentSource=`#version 300 es
precision highp float;
precision highp usampler2D;
uniform usampler2D pixels;
uniform sampler2D palette;
uniform int wall, pixelWidth, verticalMask, horizontalMask, phase, verticalOffset, verticalShift, transparentWall;
uniform float heightSpan;
in vec2 texcoord;
in float lightValue;
in float viewDepth;
out vec4 colour;
void main(){
 ivec2 coord;int shade;
 if(wall==1){
  int column=(int(floor(texcoord.x))&horizontalMask)+phase;
  int row=int(floor(float(verticalOffset)+texcoord.y*heightSpan))&verticalMask;
  int at=((column/3)*(1<<verticalShift)+row)*3+(column%3);
  coord=ivec2(at%pixelWidth,at/pixelWidth);
  float d6=clamp(floor(lightValue)*2.,0.,64.);shade=int(min(31.,ceil(d6/2.)));
 } else {
  coord=ivec2(int(floor(texcoord.x))&63,int(floor(texcoord.y))&63);
  float index=clamp(floor(lightValue)+5.,0.,28.);shade=int(ceil(index/2.));
 }
 uint index=texelFetch(pixels,coord,0).r;
 if(transparentWall==1 && index==0u)discard;
 colour=texelFetch(palette,ivec2(int(index),shade),0);
}`;
interface GpuMaterial { pixels: WebGLTexture; palette: WebGLTexture; source: IndexedMaterial }
export class SpatialRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly buffer: WebGLBuffer;
  private materials=new Map<string,GpuMaterial>();
  private materialSource?: ReadonlyMap<string,IndexedMaterial>;
  private faces: readonly TexturedFace[]=[];
  constructor(private canvas: HTMLCanvasElement){
    const gl=canvas.getContext('webgl2',{alpha:false,antialias:true,preserveDrawingBuffer:true});if(!gl)throw new Error('Textured editing needs WebGL2. Enable browser hardware acceleration and retry. Your map can still be edited in axis views.');
    this.gl=gl;
    const compile=(type:number,source:string)=>{const shader=gl.createShader(type)!;gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){const error=gl.getShaderInfoLog(shader);gl.deleteShader(shader);throw new Error(`3D editing shader: ${error}`);}return shader;};
    const vertex=compile(gl.VERTEX_SHADER,vertexSource),fragment=compile(gl.FRAGMENT_SHADER,fragmentSource),program=gl.createProgram()!;
    gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);gl.deleteShader(vertex);gl.deleteShader(fragment);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(`3D editing renderer: ${gl.getProgramInfoLog(program)}`);
    this.program=program;this.vao=gl.createVertexArray()!;this.buffer=gl.createBuffer()!;
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    for(const [location,count,offset] of [[0,3,0],[1,2,12],[2,1,20]]){gl.enableVertexAttribArray(location!);gl.vertexAttribPointer(location!,count!,gl.FLOAT,false,24,offset!);}
    gl.useProgram(program);gl.uniform1i(gl.getUniformLocation(program,'pixels'),0);gl.uniform1i(gl.getUniformLocation(program,'palette'),1);
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.frontFace(gl.CCW);
  }
  setScene(faces: readonly TexturedFace[], materials: ReadonlyMap<string,IndexedMaterial>){
    if(this.faces===faces && this.materialSource===materials)return;
    const gl=this.gl;
    if(this.materialSource!==materials){
    this.materialSource=materials;
    for(const material of this.materials.values()){gl.deleteTexture(material.pixels);gl.deleteTexture(material.palette);}this.materials.clear();
    for(const [key,source] of materials){
      if(Math.max(source.width,source.height)>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw new Error(`Material ${key} exceeds this device's texture limit.`);
      const upload=(width:number,height:number,bytes:Uint8Array,indexed:boolean)=>{const texture=gl.createTexture()!;gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.texImage2D(gl.TEXTURE_2D,0,indexed?gl.R8UI:gl.RGBA8,width,height,0,indexed?gl.RED_INTEGER:gl.RGBA,gl.UNSIGNED_BYTE,bytes);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return texture;};
      this.materials.set(key,{pixels:upload(source.width,source.height,source.pixels,true),palette:upload(source.colours,source.shades,source.palette,false),source});
    }
    }
    this.faces=faces;
    const vertices:number[]=[];
    for(const face of faces)for(let i=1;i<face.vertices.length-1;i++)for(const vertex of [face.vertices[0]!,face.vertices[i]!,face.vertices[i+1]!])vertices.push(vertex.position.x,vertex.position.y,vertex.position.z,vertex.u,vertex.v,vertex.brightness);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW);
  }
  draw(camera:EditorCamera,width:number,height:number,ceilings:boolean,cutaway:boolean,wireframe=false){
    const gl=this.gl,ratio=window.devicePixelRatio||1;this.canvas.width=Math.round(width*ratio);this.canvas.height=Math.round(height*ratio);
    gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(.086,.11,.14,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);gl.bindVertexArray(this.vao);
    const uniform=(name:string)=>gl.getUniformLocation(this.program,name),basis=cameraBasis(camera);
    for(const [name,value] of [['eye',basis.eye],['rightAxis',basis.right],['upAxis',basis.up],['backAxis',basis.back]] as const)gl.uniform3f(uniform(name),value.x,value.y,value.z);
    gl.uniform2f(uniform('viewport'),width,height);gl.uniform1f(uniform('distanceToTarget'),camera.distance);gl.uniform1f(uniform('perspective'),camera.perspective?1:0);
    if(cutaway)gl.enable(gl.CULL_FACE);else gl.disable(gl.CULL_FACE);
    if(wireframe)return;
    let first=0;
    for(const face of this.faces){
      const count=(face.vertices.length-2)*3,material=this.materials.get(face.material);if(!material)throw new Error(`Missing 3D material ${face.material}.`);
      if(face.kind!=='ceiling'||ceilings){
        gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,material.pixels);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,material.palette);
        const wall=face.graphic.wall;
        gl.uniform1i(uniform('wall'),wall?1:0);gl.uniform1i(uniform('pixelWidth'),material.source.width);gl.uniform1i(uniform('transparentWall'),wall?.type===13?1:0);
        if(wall){const rows=material.source.rows,mask=wall.verticalMask&&wall.verticalShift?wall.verticalMask:rows-1,shift=wall.verticalMask&&wall.verticalShift?wall.verticalShift:Math.log2(rows),span=(wall.bottomFixed-wall.topFixed)/256;
          gl.uniform1i(uniform('verticalMask'),mask);gl.uniform1i(uniform('verticalShift'),shift);gl.uniform1i(uniform('horizontalMask'),wall.horizontalMask);gl.uniform1i(uniform('phase'),(wall.horizontalOffset*16<<16)>>16);gl.uniform1i(uniform('verticalOffset'),wall.texture===11?0:wall.verticalOffset);
          gl.uniform1f(uniform('heightSpan'),wall.texture===11?rows/span:rows<64?Math.max(64,span)/span:1);
        }
        gl.drawArrays(gl.TRIANGLES,first,count);
      }
      first+=count;
    }
  }
  dispose(){const gl=this.gl;for(const material of this.materials.values()){gl.deleteTexture(material.pixels);gl.deleteTexture(material.palette);}gl.deleteBuffer(this.buffer);gl.deleteVertexArray(this.vao);gl.deleteProgram(this.program);gl.getExtension('WEBGL_lose_context')?.loseContext();}
}
