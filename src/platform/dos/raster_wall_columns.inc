/* WallRoutine3 native columns: keep both original 16.16 walks and palette
 * words. Programs live in the existing bounded row-code workspace. DOS draws
 * synchronously; every immediate is complete before CALL flushes prefetch.
 * Independent walks use EAX/EBP. Equal steps use EAX plus a fixed EBP offset;
 * equal starts/steps share the extracted sample. DL/DH select one DWORD pair.
 * EBX is the first step, ESI the caller-owned pair table, EDI the output.
 * ECX/EDX are scratch. Early-clobber outputs keep the fixed EBP role distinct
 * even when input values happen to match. Only the common vertical interval
 * is paired; separate pieces and caps retain the original source samples. */
enum { WALL_COLUMN_BODY=39,WALL_COLUMN_TAIL=336 };

enum { WALL_COLUMN_COMMON=416,WALL_COLUMN_COMMON_TAIL=736,WALL_COLUMN_EQUAL=832,WALL_COLUMN_EQUAL_TAIL=1088 };
static void wall_column_program_part(unsigned base,unsigned tail,const uint8_t *body,unsigned bytes,unsigned shift,unsigned shift0,unsigned shift1,unsigned dest)
{
 static const uint8_t footer[19]={0x81,0xc7,0,0,0,0,0x81,0xff,0,0,0,0,0x0f,0x85,0,0,0,0,0xc3};
 for(unsigned row=0;row<8;row++){
  uint8_t *p=wall_row_code+base+row*bytes;uint32_t offset=row*640u;
  memcpy(p,body,bytes);p[shift0]=(uint8_t)shift;if(shift1)p[shift1]=(uint8_t)shift;memcpy(p+dest,&offset,4);
 }
 memcpy(wall_row_code+tail,wall_row_code+base,bytes);
 for(unsigned which=0;which<2;which++){
  unsigned first=which?tail:base,n=which?bytes:8*bytes;
  uint32_t stride=which?640:5120;int32_t back=-(int32_t)(n+18);
  uint8_t *p=wall_row_code+first+n;memcpy(p,footer,sizeof(footer));memcpy(p+2,&stride,4);memcpy(p+14,&back,4);
 }
}
static void wall_column_program_prepare(unsigned shift)
{
 static const uint8_t arbitrary[39]={
  0x89,0xc1,0xc1,0xe9,0,0x8a,0x91,0,0,0,0,
  0x89,0xe9,0xc1,0xe9,0,0x8a,0xb1,0,0,0,0,
  0x8b,0x0c,0x96,0x89,0x8f,0,0,0,0,0x01,0xd8,0x81,0xc5,0,0,0,0};
 static const uint8_t common[34]={
  0x89,0xc1,0xc1,0xe9,0,0x8a,0x91,0,0,0,0,
  0x8d,0x0c,0x28,0xc1,0xe9,0,0x8a,0xb1,0,0,0,0,
  0x8b,0x0c,0x96,0x89,0x8f,0,0,0,0,0x01,0xd8};
 static const uint8_t equal[28]={
  0x89,0xc1,0xc1,0xe9,0,0x8a,0x91,0,0,0,0,
  0x8a,0xb1,0,0,0,0,0x8b,0x0c,0x96,0x89,0x8f,0,0,0,0,0x01,0xd8};
 _Static_assert(WALL_COLUMN_EQUAL_TAIL+28+19<sizeof(wall_row_code),"common programs exceed row workspace");
 wall_column_program_part(0,WALL_COLUMN_TAIL,arbitrary,39,shift,4,15,27);
 wall_column_program_part(WALL_COLUMN_COMMON,WALL_COLUMN_COMMON_TAIL,common,34,shift,4,16,28);
 wall_column_program_part(WALL_COLUMN_EQUAL,WALL_COLUMN_EQUAL_TAIL,equal,28,shift,4,0,22);
}

__attribute__((noinline,optimize("O2")))
static void wall_column_pair_independent(RendererPixel *out,unsigned count,const uint8_t *column0,const uint8_t *column1,
                     uint32_t y0,uint32_t dy0,uint32_t y1,uint32_t dy1,unsigned shift)
{
 uint32_t source0=(uintptr_t)column0,source1=(uintptr_t)column1,address=(uintptr_t)out;
 unsigned rotate=shift-16;
 y0<<=rotate;y1<<=rotate;dy0<<=rotate;dy1<<=rotate;
#define WALL_COLUMN_PATCH_INDEPENDENT(OFFSET) do{memcpy(wall_row_code+(OFFSET)+7,&source0,4);memcpy(wall_row_code+(OFFSET)+18,&source1,4);memcpy(wall_row_code+(OFFSET)+35,&dy1,4);}while(0)
 WALL_COLUMN_PATCH_INDEPENDENT(0);WALL_COLUMN_PATCH_INDEPENDENT(39);WALL_COLUMN_PATCH_INDEPENDENT(78);WALL_COLUMN_PATCH_INDEPENDENT(117);
 WALL_COLUMN_PATCH_INDEPENDENT(156);WALL_COLUMN_PATCH_INDEPENDENT(195);WALL_COLUMN_PATCH_INDEPENDENT(234);WALL_COLUMN_PATCH_INDEPENDENT(273);WALL_COLUMN_PATCH_INDEPENDENT(WALL_COLUMN_TAIL);
#undef WALL_COLUMN_PATCH_INDEPENDENT
 unsigned full=count&~7u;
 if(full){
  uint32_t end=address+full*640u;memcpy(wall_row_code+8*WALL_COLUMN_BODY+8,&end,4);
  __asm__ volatile("xorl %%edx,%%edx\n\tcall %c[code]"
   : "+&a"(y0),"+&r"(y1),"+&D"(address)
   : "b"(dy0),"S"(wall_pair_lookup->pixels),[code] "i"(wall_row_code)
   : "ecx","edx","cc","memory");
 }
 count&=7;
 if(count){
  uint32_t end=address+count*640u;memcpy(wall_row_code+WALL_COLUMN_TAIL+WALL_COLUMN_BODY+8,&end,4);
  __asm__ volatile("xorl %%edx,%%edx\n\tcall %c[code]"
   : "+&a"(y0),"+&r"(y1),"+&D"(address)
   : "b"(dy0),"S"(wall_pair_lookup->pixels),[code] "i"(wall_row_code+WALL_COLUMN_TAIL)
   : "ecx","edx","cc","memory");
 }
}

__attribute__((noinline,optimize("O2")))
static void wall_column_pair_common(RendererPixel *out,unsigned count,const uint8_t *column0,const uint8_t *column1,
                       uint32_t y,uint32_t dy,uint32_t offset,int equal)
{
 uint32_t source0=(uintptr_t)column0,source1=(uintptr_t)column1,address=(uintptr_t)out;
#define WALL_COLUMN_PATCH_COMMON(OFF,S1) do{memcpy(wall_row_code+(OFF)+7,&source0,4);memcpy(wall_row_code+(OFF)+(S1),&source1,4);}while(0)
 if(equal){
  WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+0,13);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+28,13);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+56,13);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+84,13);
  WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+112,13);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+140,13);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+168,13);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL+196,13);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_EQUAL_TAIL,13);
 }else{
  WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+0,19);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+34,19);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+68,19);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+102,19);
  WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+136,19);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+170,19);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+204,19);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON+238,19);WALL_COLUMN_PATCH_COMMON(WALL_COLUMN_COMMON_TAIL,19);
 }
#undef WALL_COLUMN_PATCH_COMMON
#define WALL_COLUMN_CALL_COMMON(CODE) __asm__ volatile("xorl %%edx,%%edx\n\tcall %c[code]" \
 : "+&a"(y),"+&D"(address),"+&r"(offset) : "b"(dy),"S"(wall_pair_lookup->pixels),[code] "i"(wall_row_code+(CODE)) \
 : "ecx","edx","cc","memory")
 unsigned full=count&~7u;
 if(full){
  uint32_t end=address+full*640u;
  if(equal){memcpy(wall_row_code+WALL_COLUMN_EQUAL+8*28+8,&end,4);WALL_COLUMN_CALL_COMMON(WALL_COLUMN_EQUAL);}
  else{memcpy(wall_row_code+WALL_COLUMN_COMMON+8*34+8,&end,4);WALL_COLUMN_CALL_COMMON(WALL_COLUMN_COMMON);}
 }
 count&=7;
 if(count){
  uint32_t end=address+count*640u;
  if(equal){memcpy(wall_row_code+WALL_COLUMN_EQUAL_TAIL+28+8,&end,4);WALL_COLUMN_CALL_COMMON(WALL_COLUMN_EQUAL_TAIL);}
  else{memcpy(wall_row_code+WALL_COLUMN_COMMON_TAIL+34+8,&end,4);WALL_COLUMN_CALL_COMMON(WALL_COLUMN_COMMON_TAIL);}
 }
#undef WALL_COLUMN_CALL_COMMON
}
__attribute__((always_inline)) static inline
void wall_column_pair(RendererPixel *out,unsigned count,const uint8_t *column0,const uint8_t *column1,
               uint32_t y0,uint32_t dy0,uint32_t y1,uint32_t dy1,unsigned shift)
{
 unsigned rotate=shift-16;
 if((dy0<<rotate)==(dy1<<rotate)){
  uint32_t y=y0<<rotate,offset=(y1<<rotate)-y;
  wall_column_pair_common(out,count,column0,column1,y,dy0<<rotate,offset,offset==0);
 }else wall_column_pair_independent(out,count,column0,column1,y0,dy0,y1,dy1,shift);
}

static void wall_column_piece(RendererPixel *out,unsigned pitch,const DosWallSpan *s,const RendererPixel *palette,
                       unsigned mask,int top,int bottom)
{
 if(top>bottom)return;
 dos_raster_wall(out+top*pitch,bottom-top+1,pitch,s->texels,palette,s->y+(uint32_t)(top-s->top)*s->dy,s->dy,mask);
}
void dos_raster_wall_spans(RendererPixel *out,unsigned pitch,const DosWallSpan *spans,unsigned count,const RendererPixel *palette,unsigned mask)
{
 if(!count)return;
 unsigned area=0,shift=mask==15?28:mask==31?27:mask==63?26:mask==127?25:mask==255?24:0;
 if(pitch==320 && shift)for(unsigned x=0;x+1<count;x+=2){
  int first=spans[x].top>spans[x+1].top?spans[x].top:spans[x+1].top;
  int last=spans[x].bottom<spans[x+1].bottom?spans[x].bottom:spans[x+1].bottom;
  if(last-first+1>=8)area+=(last-first+1)*2;
 }
 int paired=area>=8192 && wall_pairs_ready(palette,area,1);
 if(paired)wall_column_program_prepare(shift);
 unsigned x=0;
 for(;x+1<count;x+=2){
  const DosWallSpan *a=spans+x,*b=a+1;
  int first=a->top>b->top?a->top:b->top,last=a->bottom<b->bottom?a->bottom:b->bottom;
  if(paired && last-first+1>=8){
   wall_column_piece(out+x,pitch,a,palette,mask,a->top,first-1);wall_column_piece(out+x,pitch,a,palette,mask,last+1,a->bottom);
   wall_column_piece(out+x+1,pitch,b,palette,mask,b->top,first-1);wall_column_piece(out+x+1,pitch,b,palette,mask,last+1,b->bottom);
   wall_column_pair(out+first*pitch+x,last-first+1,a->texels,b->texels,
    a->y+(uint32_t)(first-a->top)*a->dy,a->dy,b->y+(uint32_t)(first-b->top)*b->dy,b->dy,shift);
  }else{wall_column_piece(out+x,pitch,a,palette,mask,a->top,a->bottom);wall_column_piece(out+x+1,pitch,b,palette,mask,b->top,b->bottom);}
 }
 if(x<count)wall_column_piece(out+x,pitch,spans+x,palette,mask,spans[x].top,spans[x].bottom);
 for(x=0;x<count;x++)if(spans[x].top<=spans[x].bottom){
  if(spans[x].caps&DOS_WALL_CAP_TOP)out[(spans[x].top-1)*pitch+x]=out[spans[x].top*pitch+x];
  if(spans[x].caps&DOS_WALL_CAP_BOTTOM)out[(spans[x].bottom+1)*pitch+x]=out[spans[x].bottom*pitch+x];
 }
}
