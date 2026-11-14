/* Independent original fixed-point column reference for native batching. */
#ifndef AB3D_DOS_WALL_SPAN_CASES_H
#define AB3D_DOS_WALL_SPAN_CASES_H
#ifdef AB3D_NATIVE_RGB565
static void check_wall_spans(void)
{
 static const unsigned masks[]={0,15,31,63,127,255,0x55,511};
 static const unsigned heights[]={0,1,2,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,199};
 static const unsigned widths[]={0,1,2,3,4,7,8,15,16,17,31,63,64,65,127,128,129,255,319,320};
 static const uint32_t steps[]={0,1,4096,65535,65536,65537,0x80000000u,UINT32_MAX,0x3fffffffu};
 static const uint32_t starts[]={0,1,65535,65536,UINT32_MAX,0x80000000u,0x1fffffffu,0x01000001u};
 uint8_t *texture=malloc(320*512);assert(texture);DosWallSpan spans[320];RendererPixel palette[32];
 uint32_t seed=0x68af1903u;
 for(unsigned i=0;i<320*512;i++){seed=seed*1664525u+1013904223u;texture[i]=(uint8_t)(seed>>27);}
 check_wall_pair_workspace_init();
 unsigned checked=0;
 for(unsigned test=0;test<1576;test++){
  unsigned tails=test>=1256,forced=tails || test<40,index=tails?(test-1256)%40:forced?test:test-40;
  unsigned mi=forced?(index/8)+1:(index/152)%8,mode=forced?(index/2)%4:(index/38)%4;
  unsigned hi=forced?17:(index/2)%19,align=index&1,mask=masks[mi];
  unsigned width=forced?128:widths[(index/3+mode*7)%20],height=tails?128+(test-1256)/40:heights[hi];
  unsigned pitch=forced?320:index%11==0?319:index%13==0?328:320;
  if(width>pitch)width=pitch;
  for(unsigned c=0;c<32;c++){seed=seed*1664525u+1013904223u;palette[c]=(RendererPixel)(seed>>8);}
  for(unsigned x=0;x<width;x++){
   unsigned top=forced?10:(test+x/2)%17,h=height;
   if(!forced && test%17==0){top=x&1?150:10;h=8;}
   else if(!forced && x&1 && mode!=2 && mode!=3)top+=x%7;
   spans[x].texels=texture+((x*73+test*7)%320)*512;
   spans[x].top=(int16_t)top;spans[x].bottom=(int16_t)(top+(int)h-1);
   spans[x].caps=h?((top && (test+x)&1)?1:0)|((test+x)&2):0;
   spans[x].y=starts[(test+x/2)%8];spans[x].dy=steps[(test+x/2)%9];
   if(x&1){
    if(mode==0)spans[x].dy=spans[x-1].dy+1;
    else if(mode==1){spans[x].dy=spans[x-1].dy;spans[x].y=spans[x-1].y+65537u;}
    else if(mode==2){spans[x].dy=spans[x-1].dy;spans[x].y=spans[x-1].y;}
    else {uint32_t period=(mask+1u)<<16;spans[x].dy=spans[x-1].dy+period;spans[x].y=spans[x-1].y+period;}
   }
  }
  memset(oracle_surfaces,0x91,sizeof(oracle_surfaces));
  if(test%41==0)dos_raster_release_wall_pairs();
  else dos_raster_prepare_wall_pairs(&wall_pair_test_storage[(test/37)&1]->lookup);
  dos_raster_wall_spans(oracle_surfaces[0]+align,pitch,spans,width,palette,mask);
  for(unsigned x=0;x<width;x++)if(spans[x].top<=spans[x].bottom){
   uint32_t y=spans[x].y;
   for(int row=spans[x].top;row<=spans[x].bottom;row++,y+=spans[x].dy)
    oracle_surfaces[1][align+row*pitch+x]=palette[spans[x].texels[(y>>16)&mask]];
   if(spans[x].caps&1)oracle_surfaces[1][align+(spans[x].top-1)*pitch+x]=oracle_surfaces[1][align+spans[x].top*pitch+x];
   if(spans[x].caps&2)oracle_surfaces[1][align+(spans[x].bottom+1)*pitch+x]=oracle_surfaces[1][align+spans[x].bottom*pitch+x];
  }
  if(memcmp(oracle_surfaces[0],oracle_surfaces[1],sizeof(oracle_surfaces[0]))){
   for(unsigned i=0;i<ORACLE_SURFACE_PIXELS;i++)if(oracle_surfaces[0][i]!=oracle_surfaces[1][i]){
    fprintf(stderr,"column spans case=%u mask=%u mode=%u width=%u height=%u pitch=%u align=%u mismatch=%u actual=%04x expected=%04x\n",test,mask,mode,width,height,pitch,align,i,oracle_surfaces[0][i],oracle_surfaces[1][i]);break;
   }abort();
  }
  check_wall_pair_workspace_guards();checked++;
 }
 dos_raster_wall_spans(NULL,320,NULL,0,NULL,255);
 dos_raster_release_wall_pairs();free(wall_pair_test_storage[0]);free(wall_pair_test_storage[1]);free(texture);
 printf("PASS: %u independent column-span cases retain separate/common/equal/modulo-equivalent sampling, zero/full steps, masks, alignments, all eight-row group/tail boundaries, nonoverlap, clipping, caps, palette/workspace changes, legacy paths and full guards\n",checked);
}


#endif
#endif
