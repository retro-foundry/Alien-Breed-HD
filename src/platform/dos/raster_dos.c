#include "raster_dos.h"
#include "sprite_format.h"
#ifdef AB3D_NATIVE_RGB565
#include "sprite_alpha_dos.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static void fill_words(int16_t *out, unsigned count, uint16_t value)
{
    unsigned pairs = count / 2;
    uint32_t repeated = value | ((uint32_t)value << 16);
    __asm__ volatile("rep stosl" : "+D"(out), "+c"(pairs) : "a"(repeated) : "memory", "cc");
    if (count & 1) *out = (int16_t)value;
}

#ifdef AB3D_NATIVE_RGB565
/* Low word: ready-to-write RGB565. High word: the original 4-bit channels
 * in five-bit lanes, leaving one guard bit for each signed blend delta. */
static uint32_t water_brighten[22][256];
static int8_t water_blend_delta[128][32];
static RendererPixel water_rg[256],water_blue[16];
/* Brighten uses only the low eleven RGB565 green/blue bits. */
static uint8_t water_gb_index[2048];

void dos_raster_prepare_water(const uint8_t *brighten)
{
    for(unsigned row=0;row<22;row++)for(unsigned sample=0;sample<256;sample++) {
        unsigned i=(row*256+sample)*2;
        unsigned color=((unsigned)brighten[i]<<8)|brighten[i+1];
        water_brighten[row][sample]=((color&0xf00)<<18)|((color&0xf0)<<17)|
            ((color&15)<<16)|renderer_pixel_from_c12((uint16_t)color);
    }
    for(unsigned alpha=0;alpha<128;alpha++)for(int difference=-15;difference<=15;difference++) {
        /* floor((17*(255*bg + difference*alpha)/255 + 8)/17)
         * = bg + floor((difference*alpha + 120)/255).
         * C division truncates negative values, so adjust to floor. */
        int n=difference*(int)alpha+120;
        water_blend_delta[alpha][difference+16]=(int8_t)(n>=0?n/255:-((-n+254)/255));
    }
    for(unsigned i=0;i<2048;i++)water_gb_index[i]=(uint8_t)renderer_pixel_gb_index((uint16_t)i);
    for(unsigned rg=0;rg<256;rg++)water_rg[rg]=renderer_pixel_from_c12((uint16_t)(rg<<4));
    for(unsigned b=0;b<16;b++)water_blue[b]=renderer_pixel_from_c12((uint16_t)b);
}

RendererPixel dos_raster_water_color(unsigned row,unsigned sample)
{
    return (RendererPixel)water_brighten[row][sample];
}

/* A constant one_row selects an already resolved 64-byte phase row.
 * Both samplers share the original native Brighten/blend operations. */
__attribute__((always_inline)) static inline
void water_span(RendererPixel *out,unsigned count,
                const RendererPixel *background0,const RendererPixel *background1,
                const uint8_t *phase,const uint32_t (*brighten)[256],const int8_t *blend,
                unsigned offset,uint32_t u,uint32_t v,uint32_t du,uint32_t dv,int one_row)
{
    /* AB3DI.s texturedwater wraps this selected phase row every 64
     * texels. Keep its six integer bits at the top of the accumulator;
     * unsigned overflow then supplies the exact modulo-64 wrap. */
    if(one_row){u=(u+((uint32_t)offset<<16))<<10;du<<=10;}
    if(!background1) {
        for(unsigned x=0;x<count;x++,u+=du,v+=dv) {
            unsigned level;
            if(one_row)level=phase[u>>26];
            else {
                uint16_t uv=(uint16_t)(((v>>8)&0xff00)|((u>>16)&255));
                uv=(uint16_t)(uv+offset);
                level=phase[((uv>>2)&0xfc0)|(uv&63)];
            }
            out[x]=(RendererPixel)brighten[level][water_gb_index[background0[x]&2047]];
        }
        return;
    }
    for(unsigned x=0;x<count;x++,u+=du,v+=dv) {
        unsigned level;
        if(one_row)level=phase[u>>26];
        else {
            uint16_t uv=(uint16_t)(((v>>8)&0xff00)|((u>>16)&255));
            uv=(uint16_t)(uv+offset);
            level=phase[((uv>>2)&0xfc0)|(uv&63)];
        }
        uint32_t c0=brighten[level][water_gb_index[background0[x]&2047]];
        uint32_t c1=brighten[level][water_gb_index[background1[x]&2047]];
        if(c0==c1) {
            out[x]=(RendererPixel)c0;
            continue;
        }
        /* The three guard bits prevent a negative channel difference from
         * borrowing into the next lane. Each lookup still uses delta+16. */
        uint32_t difference=((c1>>16)|0x4210u)-(c0>>16);
        unsigned r=(c0>>26)+blend[(difference>>10)&31];
        unsigned g=((c0>>21)&15)+blend[(difference>>5)&31];
        unsigned b=((c0>>16)&15)+blend[difference&31];
        out[x]=water_rg[(r<<4)|g]|water_blue[b];
    }
}

/* Keep horizontal sampling state separate from byte-carry segmentation. */
__attribute__((noinline,optimize("O3,unroll-loops")))
static void water_span_row(RendererPixel *out,unsigned count,
                           const RendererPixel *background0,const RendererPixel *background1,
                           const uint8_t *phase,const uint32_t (*brighten)[256],const int8_t *blend,
                           unsigned offset,uint32_t u,uint32_t du)
{
    water_span(out,count,background0,background1,phase,brighten,blend,offset,u,0,du,0,1);
}

__attribute__((noinline,optimize("O3,unroll-loops")))
void dos_raster_water(RendererPixel *out,unsigned count,
                        const RendererPixel *background0,const RendererPixel *background1,
                        const uint8_t *phase,unsigned shade,unsigned offset,unsigned alpha,
                        uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    const uint32_t (*brighten)[256]=water_brighten+shade;
    /* Exact rounded interpolation is symmetric under swapping the samples. */
    if(background1 && alpha>127) {
        const RendererPixel *swap=background0;background0=background1;background1=swap;
        alpha=255-alpha;
    }
    /* At half-alpha 0..8 every possible 4-bit channel delta rounds to zero. */
    if(alpha<=8 || background0==background1)background1=NULL;
    const int8_t *blend=background1?water_blend_delta[alpha]:NULL;
    if(!dv && du && du<=65536u && count<=320) {
        /* AB3DI texturedwater adds wateroff to the packed 16-bit UV word.
         * With constant V, only the low U-byte carry can change its phase
         * row. Split at that carry or the U-byte wrap, preserving both. */
        unsigned bias=offset&255,base=((v>>16)+(offset>>8))&63;
        while(count) {
            unsigned low=u&0xffffffu,carry=((low>>16)+bias)>>8;
            unsigned limit=carry?0x1000000u:((256-bias)<<16);
            unsigned run=count;
            /* At most 319 one-texel steps keep this product in 32 bits.
             * A span ending before the boundary needs no division. */
            if((count-1)*du>=limit-low)run=(limit-1-low)/du+1;
            water_span_row(out,run,background0,background1,phase+(((base+carry)&63)*64),
                           brighten,blend,bias,u,du);
            out+=run;background0+=run;if(background1)background1+=run;
            u+=run*du;count-=run;
        }
    }
    else water_span(out,count,background0,background1,phase,brighten,blend,offset,u,v,du,dv,0);
}

#endif

/* BitMapObj writes are opaque except for source index zero. Within one
 * all-billboard object pass, traversing the original painter list backwards
 * lets the first opaque pixel own the result. Reset at each room/section. */
static DosSpriteBatch *sprite_batch;
static RendererPixel *sprite_batch_surface;
static DosSpriteHistory *sprite_history;

void dos_raster_sprite_batch_begin(RendererPixel *surface,DosSpriteBatch *scratch)
{
    sprite_batch=scratch;
    sprite_history=NULL;
    memset(sprite_batch->covered,0,sizeof(sprite_batch->covered));
    memset(sprite_batch->open_words,10,sizeof(sprite_batch->open_words));
    sprite_batch_surface=surface;
}
void dos_raster_sprite_batch_history_begin(DosSpriteHistory *scratch)
{
    scratch->count=0;scratch->next=0;sprite_history=scratch;
}
void dos_raster_sprite_batch_end(void) { sprite_batch_surface=NULL;sprite_batch=NULL;sprite_history=NULL; }

/* Masks and sampled row indices need equality only. Both callers provide
 * at least one 32-bit word, avoiding the C library's byte comparison loop. */
static inline int sprite_words_equal(const void *a,const void *b,unsigned words)
{
    unsigned char equal;
    __asm__ volatile("repe cmpsl; sete %0" : "=qm"(equal),"+S"(a),"+D"(b),"+c"(words) : : "memory","cc");
    return equal;
}

static int sprite_history_match(const DosSpriteHistoryEntry *entry,const RendererPixel *out,
                                  unsigned width,unsigned height,unsigned pitch,const DosSpriteRows *sprite)
{
    _Static_assert(sizeof(int)==sizeof(uint32_t),"486 sprite row word layout");
    /* Palette changes cannot expose an already claimed opaque texel. Rows
     * need their own copy because the renderer reuses its sampling array. */
    const DosSpriteRows *old=&entry->sprite;
    return entry->out==out && entry->width==width && entry->height==height && entry->pitch==pitch &&
        old->wad==sprite->wad && old->ptr==sprite->ptr &&
        old->wad_bytes==sprite->wad_bytes && old->ptr_bytes==sprite->ptr_bytes &&
        old->ptr_offset==sprite->ptr_offset && old->u==sprite->u && old->du==sprite->du &&
        old->columns==sprite->columns && old->down_strip==sprite->down_strip &&
        (sprite->rows ? old->rows && sprite_words_equal(entry->rows,sprite->rows,height) :
         !old->rows && old->row_v==sprite->row_v && old->row_dv==sprite->row_dv &&
         old->row_last==sprite->row_last);
}

static void sprite_history_note(RendererPixel *out,unsigned width,unsigned height,unsigned pitch,
                                 const DosSpriteRows *sprite,unsigned visible)
{
    DosSpriteHistoryEntry *entry=sprite_history->entries+sprite_history->next;
    entry->out=out;entry->width=width;entry->height=height;entry->pitch=pitch;
    entry->sprite=*sprite;entry->visible=visible;
    if(sprite->rows)memcpy(entry->rows,sprite->rows,height*sizeof(*entry->rows));
    if(sprite_history->count<4)sprite_history->count++;
    sprite_history->next=(sprite_history->next+1)&3;
}

static void sprite_copy(RendererPixel *out,const RendererPixel *source,unsigned count)
{
#ifdef AB3D_DOS_INDEXED
    unsigned quads=count/4;
    __asm__ volatile("rep movsl" : "+S"(source), "+D"(out), "+c"(quads) : : "memory","cc");
    if(count&2)__asm__ volatile("movsw" : "+S"(source), "+D"(out) : : "memory","cc");
    if(count&1)*out=*source;
#else
    /* MOVS preserves SHR's carry, which identifies the final odd word. */
    __asm__ volatile("shrl $1,%%ecx; rep movsl; jnc 1f; movsw; 1:"
                     : "+S"(source), "+D"(out), "+c"(count) : : "memory","cc");
#endif
}

typedef struct { const RendererPixel *source;unsigned count; } DosSpriteSpan;

#ifdef AB3D_NATIVE_RGB565
/* The caller already proved this range list nonempty. Keep the whole
 * row copy in registers; each span's source offset selects its target. */
static unsigned sprite_copy_ranges(RendererPixel *out,const RendererPixel *row,
                                    const DosSpriteSpan *span,const DosSpriteSpan *end)
{
    unsigned written;
    uintptr_t delta=(uintptr_t)out-(uintptr_t)row;
    __asm__ volatile(
        "xorl %%eax,%%eax\n\t"
        "1: movl (%%ebx),%%esi\n\t"
        "leal (%%esi,%%edx),%%edi\n\t"
        "movl 4(%%ebx),%%ecx\n\t"
        "addl %%ecx,%%eax\n\t"
        "shrl $1,%%ecx\n\t"
        "rep movsl\n\t"
        "jnc 2f\n\t"
        "movsw\n\t"
        "2: addl $8,%%ebx\n\t"
        "cmpl %[end],%%ebx\n\t"
        "jne 1b"
        : "+&b"(span),"=&a"(written)
        : "d"(delta),[end]"r"(end)
        : "ecx","esi","edi","memory","cc");
    return written;
}
#endif

static inline void sprite_copy_record(RendererPixel *out,const RendererPixel *source,
                                      unsigned count,DosSpriteSpan **next)
{
    if(next){(*next)->source=source;(*next)->count=count;(*next)++;}
    sprite_copy(out,source,count);
}

static unsigned sprite_batch_copy(RendererPixel *out,const RendererPixel *source,
                                   unsigned count,DosSpriteSpan **next)
{
    unsigned at=(unsigned)(out-sprite_batch_surface),y=at/320,x=at%320;
    if(!sprite_batch->open_words[y])return 0;
    unsigned end=x+count,origin=x,run=0,written=0;
    while(x<end) {
        unsigned word=x/32,bit=x%32,limit=(word+1)*32;
        if(limit>end)limit=end;
        uint32_t range=UINT32_MAX<<bit;
        if(limit%32)range&=(1u<<(limit%32))-1;
        uint32_t old=sprite_batch->covered[y][word],fresh=range&~old;
        uint32_t updated=old|range;sprite_batch->covered[y][word]=updated;
        if(updated==UINT32_MAX && old!=UINT32_MAX)sprite_batch->open_words[y]--;
        if(fresh==range) {
            if(!run)run=x-origin+1;
        } else {
            if(run) {
                unsigned first=run-1,length=x-origin-first;
                sprite_copy_record(out+first,source+first,length,next);written+=length;run=0;
            }
            while(fresh) {
                unsigned first=(unsigned)__builtin_ctz(fresh);
                /* Adding the lowest set bit clears this run and carries to
                 * its end. A zero carry means the run reaches bit 32. */
                uint32_t carry=fresh+(fresh&-fresh);
                unsigned length=carry?(unsigned)__builtin_ctz(carry)-first:32-first;
                unsigned offset=word*32+first-origin;
                sprite_copy_record(out+offset,source+offset,length,next);written+=length;
                fresh&=carry;
            }
        }
        x=limit;
    }
    if(run) {
        unsigned first=run-1,length=count-first;
        sprite_copy_record(out+first,source+first,length,next);written+=length;
    }
    return written;
}

static inline void sprite_mask_copy(uint32_t *out,const uint32_t *source,unsigned words)
{
    __asm__ volatile("rep movsl" : "+D"(out),"+S"(source),"+c"(words) : : "memory","cc");
}

/* BitMapObj's index-zero cutouts form the same opaque mask throughout a
 * magnified row band. Build it once, then subtract each row's prior coverage
 * and copy its fresh runs, joining runs across mask-word boundaries.
 * Matching prior words can reuse those ranges and exact mask updates. */
static unsigned sprite_batch_rows(RendererPixel *out,unsigned height,
                                   unsigned pitch,const RendererPixel *row,
                                   const DosSpriteSpan *spans,unsigned span_count)
{
    if(!span_count)return 0;
    unsigned at=(unsigned)(out-sprite_batch_surface),y=at/320,x=at%320;
    unsigned left=x+(unsigned)(spans[0].source-row);
    const DosSpriteSpan *last=spans+span_count-1;
    unsigned right=x+(unsigned)(last->source-row)+last->count;
    unsigned first=left/32,words=(right-1)/32-first+1,total=0;
    /* Only the words between the first and last opaque texel are used. */
    uint32_t opaque[10]={0};
    for(unsigned i=0;i<span_count;i++) {
        unsigned start=x+(unsigned)(spans[i].source-row),end=start+spans[i].count;
        unsigned a=start/32,b=(end-1)/32;
        uint32_t left=UINT32_MAX<<(start%32),right=UINT32_MAX>>(31-((end-1)%32));
        if(a==b)opaque[a-first]|=left&right;
        else {
            opaque[a-first]|=left;opaque[b-first]|=right;
            for(unsigned word=a+1;word<b;word++)opaque[word-first]=UINT32_MAX;
        }
    }

    uint32_t before[10];
    DosSpriteSpan ranges[160]; /* At most 16 fresh runs per 32-bit word. */
    for(unsigned dy=0;dy<height;) {
        uint32_t *covered=sprite_batch->covered[y+dy]+first;
        sprite_mask_copy(before,covered,words);
        unsigned open=sprite_batch->open_words[y+dy],run=0,length=0;
        DosSpriteSpan *end=ranges;
        RendererPixel *target=out+dy*pitch;
#ifdef AB3D_NATIVE_RGB565
        /* Without any previously claimed opaque bit, BitMapObj's original
         * spans already describe the exact fresh ranges. Keep outside bits
         * and open-word counts while avoiding range reconstruction. */
        unsigned overlap=0;
        for(unsigned word=0;word<words;word++)if(before[word]&opaque[word]) {
            overlap=1;break;
        }
        if(!overlap) {
            for(unsigned word=0;word<words;word++) {
                uint32_t old=covered[word],updated=old|opaque[word];
                covered[word]=updated;
                if(updated==UINT32_MAX && old!=UINT32_MAX)sprite_batch->open_words[y+dy]--;
            }
            unsigned closed=open-sprite_batch->open_words[y+dy],next=dy;
            for(;next<height;next++) {
                if(next!=dy) {
                    uint32_t *other=sprite_batch->covered[y+next]+first;
                    if(!sprite_words_equal(before,other,words))break;
                    sprite_mask_copy(other,covered,words);
                    sprite_batch->open_words[y+next]-=(uint8_t)closed;
                }
                total+=sprite_copy_ranges(out+next*pitch,row,spans,spans+span_count);
            }
            dy=next;continue;
        }
#endif
        for(unsigned word=0;word<words;word++) {
            uint32_t old=covered[word],fresh=opaque[word]&~old;
            if(!fresh)continue;
            uint32_t updated=old|opaque[word];covered[word]=updated;
            if(updated==UINT32_MAX && old!=UINT32_MAX)sprite_batch->open_words[y+dy]--;
            while(fresh) {
                unsigned bit=(unsigned)__builtin_ctz(fresh);
                uint32_t carry=fresh+(fresh&-fresh);
                unsigned count=carry?(unsigned)__builtin_ctz(carry)-bit:32-bit;
                unsigned offset=(first+word)*32+bit-x;
                if(length && run+length!=offset) {
                    sprite_copy_record(target+run,row+run,length,&end);total+=length;length=0;
                }
                if(!length)run=offset;
                length+=count;
                fresh&=carry;
            }
        }
        if(length){sprite_copy_record(target+run,row+run,length,&end);total+=length;}
        unsigned closed=open-sprite_batch->open_words[y+dy],next=dy+1;
        if(end==ranges) {
            /* No fresh bit means no mask or open-word count changed. */
            for(;next<height;next++)
                if(!sprite_words_equal(before,sprite_batch->covered[y+next]+first,words))break;
            dy=next;continue;
        }
        for(;next<height;next++) {
            uint32_t *other=sprite_batch->covered[y+next]+first;
            if(!sprite_words_equal(before,other,words))break;
            target=out+next*pitch;
#ifdef AB3D_NATIVE_RGB565
            total+=sprite_copy_ranges(target,row,ranges,end);
#else
            for(const DosSpriteSpan *span=ranges;span<end;span++) {
                sprite_copy(target+(span->source-row),span->source,span->count);
                total+=span->count;
            }
#endif
            sprite_mask_copy(other,covered,words);
            sprite_batch->open_words[y+next]-=(uint8_t)closed;
        }
        dy=next;
    }
    return total;
}

static int sprite_batch_hidden(const RendererPixel *out,unsigned width,unsigned height)
{
    unsigned at=(unsigned)(out-sprite_batch_surface),y=at/320,x=at%320;
    unsigned first=x/32,last=(x+width-1)/32;
    uint32_t left=UINT32_MAX<<(x%32),right=UINT32_MAX>>(31-((x+width-1)%32));
    if(first==last)left&=right;
    /* Narrow billboards need only their boundary words. Closed rows already
     * have those bits set, so they need no separate open-word test. */
    if(first==last) {
        for(unsigned row=y;row<y+height;row++)
            if((sprite_batch->covered[row][first]&left)!=left)return 0;
        return 1;
    }
    if(last==first+1) {
        for(unsigned row=y;row<y+height;row++) {
            const uint32_t *covered=sprite_batch->covered[row];
            if((covered[first]&left)!=left || (covered[last]&right)!=right)return 0;
        }
        return 1;
    }
    for(unsigned row=y;row<y+height;row++) {
        if(!sprite_batch->open_words[row])continue;
        const uint32_t *covered=sprite_batch->covered[row];
        if((covered[first]&left)!=left)return 0;
        for(unsigned word=first+1;word<last;word++)if(covered[word]!=UINT32_MAX)return 0;
        if(last!=first && (covered[last]&right)!=right)return 0;
    }
    return 1;
}

static unsigned sprite_batch_column(RendererPixel *out,unsigned count,unsigned stride,
                                     const uint8_t *source,const RendererPixel *palette,
                                     const int *rows,unsigned limit,unsigned shift,int decoded)
{
    unsigned at=(unsigned)(out-sprite_batch_surface),y=at/320,x=at%320,drawn=0;
    unsigned word=x/32;uint32_t bit=1u<<(x%32);
    for(unsigned i=0;i<count;i++,y++,out+=stride) {
        if(sprite_batch->covered[y][word]&bit || (unsigned)rows[i]>limit)continue;
        unsigned row=(unsigned)rows[i];
        unsigned texel=decoded?source[row]:(sprite_word_read(source+row*2)>>shift)&31;
        if(!texel)continue;
        uint32_t bits=sprite_batch->covered[y][word]|bit;sprite_batch->covered[y][word]=bits;
        if(bits==UINT32_MAX)sprite_batch->open_words[y]--;
        *out=palette[texel];drawn++;
    }
    return drawn;
}

/* All seven 486 registers keep the clipped column sampler in one loop.
 * Bounds are unsigned so an invalid negative row is rejected too. */
#ifdef AB3D_DOS_INDEXED
#define SPRITE_COLUMN_STORE \
            "movb (%[palette],%%eax),%%al\n\t" \
            "movb %%al,(%%edi)\n\t"
#else
#define SPRITE_COLUMN_STORE \
            "movw (%[palette],%%eax,2),%%ax\n\t" \
            "movw %%ax,(%%edi)\n\t"
#endif
#define SPRITE_COLUMN_SAMPLE(SAMPLE_) \
    do { \
        if(!count)return 0; \
        const int *end=rows+count; \
        unsigned bytes=stride*sizeof(*out),drawn; \
        __asm__ volatile( \
            "xorl %%edx,%%edx\n\t" \
            "1: movl (%%esi),%%eax\n\t" \
            "addl $4,%%esi\n\t" \
            "cmpl %[limit],%%eax\n\t" \
            "ja 2f\n\t" \
            SAMPLE_ \
            "testl %%eax,%%eax\n\t" \
            "jz 2f\n\t" \
            SPRITE_COLUMN_STORE \
            "incl %%edx\n\t" \
            "2: addl %[stride],%%edi\n\t" \
            "cmpl %[end],%%esi\n\t" \
            "jne 1b" \
            : "+&D"(out), "+&S"(rows), "=&d"(drawn) \
            : "b"(source), [palette]"r"(palette), "c"(shift), \
              [limit]"m"(limit), [stride]"m"(bytes), [end]"m"(end) \
            : "eax","memory","cc"); \
        return drawn; \
    } while(0)

unsigned dos_raster_sprite_column(RendererPixel *out,unsigned count,unsigned stride,
                                   const uint8_t *source,const RendererPixel *palette,
                                   const int *rows,unsigned limit,unsigned shift)
{
    if(sprite_batch_surface)return sprite_batch_column(out,count,stride,source,palette,rows,limit,shift,0);
    SPRITE_COLUMN_SAMPLE("movzwl (%%ebx,%%eax,2),%%eax\n\t"
                         "shrl %%cl,%%eax\n\t"
                         "andl $31,%%eax\n\t");
}

unsigned dos_raster_sprite_texel_column(RendererPixel *out,unsigned count,unsigned stride,
                                         const uint8_t *source,const RendererPixel *palette,
                                         const int *rows,unsigned limit)
{
    unsigned shift=0;
    if(sprite_batch_surface)return sprite_batch_column(out,count,stride,source,palette,rows,limit,0,1);
    SPRITE_COLUMN_SAMPLE("movzbl (%%ebx,%%eax),%%eax\n\t");
}
#undef SPRITE_COLUMN_SAMPLE
#undef SPRITE_COLUMN_STORE

typedef struct { const uint8_t *source; unsigned limit,shift,width; } DosSpriteGroup;

#ifdef AB3D_NATIVE_RGB565
static inline void sprite_group_source(DosSpriteGroup *g,const DosSpriteRows *sprite,
                                       unsigned column,unsigned width,unsigned x,
                                       const uint8_t *source_base,unsigned last_offset)
#else
static inline void sprite_group_source(DosSpriteGroup *g,const DosSpriteRows *sprite,
                                       unsigned column,unsigned width)
#endif
{
    g->source=NULL;g->width=width;
#ifdef AB3D_NATIVE_RGB565
    /* Widths and destination starts fit in 16 bits at the 320-pixel cap.
     * Keep the original 16-byte descriptor without another scratch array. */
    g->width|=x<<16;
#endif
    const uint8_t *entry=sprite->ptr+sprite->ptr_offset+column*4u;
    /* ObjDraw3 PTR entries retain their original big-endian layout. One
     * unaligned-safe word load and 486 BSWAP replaces four byte loads. */
    uint32_t descriptor;
    memcpy(&descriptor,entry,sizeof(descriptor));
    descriptor=__builtin_bswap32(descriptor);
    unsigned mode=descriptor>>24;
    uint32_t offset=descriptor&0xffffffu;
#ifdef AB3D_NATIVE_RGB565
    if(!descriptor || offset>last_offset)return;
    g->source=source_base+offset;
    g->limit=(last_offset-offset)/2u;
#else
    if((!mode && !offset) || offset>=sprite->wad_bytes)return;
    unsigned words=(sprite->wad_bytes-offset)/2u;
    if(words<=sprite->down_strip)return;
    g->source=sprite->wad+offset+sprite->down_strip*2u;
    g->limit=words-1-sprite->down_strip;
#endif
    g->shift=!mode?0:mode==1?5:10;
}

__attribute__((always_inline)) static inline
DosSpriteCoverage sprite_rows(RendererPixel *out,unsigned width,
                                        unsigned height,unsigned pitch,
                                        const DosSpriteRows *sprite,int batch)
{
    enum { MAX_WIDTH=320 };
    typedef DosSpriteGroup Group;
    Group groups[MAX_WIDTH];
    RendererPixel row[MAX_WIDTH];
    typedef DosSpriteSpan Span;
    Span spans[MAX_WIDTH/2];
    _Static_assert(sizeof(Group)==16 && sizeof(Span)==8,"486 sprite descriptor layout");
    DosSpriteCoverage result={0,0};
    if(!width || width>MAX_WIDTH || !height || !sprite->columns ||
       sprite->ptr_bytes<=sprite->ptr_offset)return result;
    if(batch && sprite_batch_hidden(out,width,height))return result;
    if(batch && sprite_history && height<=240) {
        for(unsigned i=0;i<sprite_history->count;i++) {
            const DosSpriteHistoryEntry *entry=sprite_history->entries+i;
            if(sprite_history_match(entry,out,width,height,pitch,sprite)) {
                result.visible=entry->visible;
                return result;
            }
        }
    }
    unsigned available=(sprite->ptr_bytes-sprite->ptr_offset)/4u;
    if(!available)return result;
#ifdef AB3D_NATIVE_RGB565
    /* A valid first word needs offset+2*(down_strip+1)<=wad_bytes.
     * Hoist the draw-constant part, preserving odd byte offsets and lengths. */
    if(sprite->wad_bytes<2 || sprite->down_strip>(sprite->wad_bytes-2)/2u) {
        if(batch && sprite_history && height<=240)
            sprite_history_note(out,width,height,pitch,sprite,0);
        return result;
    }
    const uint8_t *source_base=sprite->wad+sprite->down_strip*2u;
    unsigned last_offset=sprite->wad_bytes-sprite->down_strip*2u-2u;
    const uint8_t (*row_bounds)[2];
    const uint32_t *nonempty_rows=dos_sprite_alpha_source(sprite,&row_bounds);
#endif
    unsigned count=0,previous=UINT32_MAX;
    uint32_t u=sprite->u;
    if(sprite->du && sprite->du<65536u) {
        /* A magnified source column repeats whole pixels. After the first
         * boundary, each run is floor(65536/du) or one pixel longer; its
         * remainder advances exactly like the original unsigned DDA. */
        unsigned whole=65536u/sprite->du,remainder=65536u%sprite->du;
        unsigned column=u>>16,run=(65535u-(u&65535u))/sprite->du+1u;
        unsigned fraction=(u&65535u)+run*sprite->du-65536u;
        for(unsigned x=0;x<width;) {
            unsigned length=run,sample=column;
            if(length>width-x)length=width-x;
            if(sample>=sprite->columns)sample=sprite->columns-1;
            if(sample>=available)sample=available-1;
            if(sample==previous)groups[count-1].width+=length;
            else {
                previous=sample;
#ifdef AB3D_NATIVE_RGB565
                sprite_group_source(&groups[count++],sprite,sample,length,x,source_base,last_offset);
#else
                sprite_group_source(&groups[count++],sprite,sample,length);
#endif
            }
            x+=length;
            column=(column+1u)&65535u;
            run=whole+(fraction<remainder);
            fraction=fraction>=remainder?fraction-remainder:fraction+sprite->du-remainder;
        }
    } else for(unsigned x=0;x<width;x++,u+=sprite->du) {
        unsigned column=u>>16;
        if(column>=sprite->columns)column=sprite->columns-1;
        if(column>=available)column=available-1;
        if(column==previous){groups[count-1].width++;continue;}
        previous=column;
#ifdef AB3D_NATIVE_RGB565
        sprite_group_source(&groups[count++],sprite,column,1,x,source_base,last_offset);
#else
        sprite_group_source(&groups[count++],sprite,column,1);
#endif
    }
    unsigned minimum_limit=UINT32_MAX;
    int all_sources=1;
#ifdef AB3D_NATIVE_RGB565
    unsigned first_column=sprite->u>>16;
    /* Magnification visits consecutive source columns until the final clamp.
     * Exclude a wrapped U walk or a truncated PTR before indexing by column. */
    if(!sprite->du || sprite->du>=65536u || first_column>=sprite->columns ||
       available<sprite->columns || sprite->u>UINT32_MAX-(width-1)*sprite->du)row_bounds=NULL;
#endif
    for(unsigned g=0;g<count;g++)if(groups[g].source) {
#ifdef AB3D_NATIVE_RGB565
        result.visible+=groups[g].width&65535u;
#else
        result.visible+=groups[g].width*height;
#endif
        if(groups[g].limit<minimum_limit)minimum_limit=groups[g].limit;
    } else all_sources=0;
#ifdef AB3D_NATIVE_RGB565
    /* Sum valid destination widths before the common height multiplication. */
    result.visible*=height;
#endif
    for(unsigned y=0;y<height;) {
        int source_row;
        unsigned end_y=y+1;
        if(sprite->rows) {
            source_row=sprite->rows[y];
            while(end_y<height && sprite->rows[end_y]==source_row)end_y++;
        } else {
            uint32_t v=sprite->row_v+y*sprite->row_dv;
            unsigned sample=v>>16;
            source_row=(int)(sample>sprite->row_last?sprite->row_last:sample);
            if(!sprite->row_dv || sample>=sprite->row_last)end_y=height;
            else if(sprite->row_dv<65536u) {
                unsigned run=(65535u-(v&65535u))/sprite->row_dv+1u;
                end_y=run>height-y?height:y+run;
            }
        }
#ifdef AB3D_NATIVE_RGB565
        if(nonempty_rows && (unsigned)source_row<DOS_SPRITE_ALPHA_ROWS &&
           !(nonempty_rows[(unsigned)source_row>>5]&(1u<<((unsigned)source_row&31)))) {
            y=end_y;continue;
        }
#endif
        if(batch && sprite_batch_hidden(out+y*pitch,width,end_y-y)) {
            y=end_y;continue;
        }
        Group *group=groups;
        const Group *groups_end=groups+count;
#ifdef AB3D_NATIVE_RGB565
        if(row_bounds && (unsigned)source_row<DOS_SPRITE_ALPHA_ROWS) {
            unsigned first=row_bounds[source_row][0],last=row_bounds[source_row][1];
            first=first>first_column?first-first_column:0;
            last=last>first_column?last-first_column:0;
            if(last>count)last=count;
            if(first>=last){y=end_y;continue;}
            group=groups+first;groups_end=groups+last;
        }
#endif
        /* Ordinary painter draws can decode their first row directly into
         * the framebuffer. Repeated rows still copy only its opaque spans. */
        RendererPixel *decoded_row=batch?row:out+y*pitch;
#ifdef AB3D_NATIVE_RGB565
        RendererPixel *destination=decoded_row+(group->width>>16);
#else
        RendererPixel *destination=decoded_row;
#endif
        Span *span=spans;
        /* BitMapObj uses source index zero for cutouts. Separate transparent
         * and opaque loops retain that state in control flow, so sampling a
         * column does not reload a flag from the stack. */
#define SPRITE_ROW_CHECK(REJECT_) \
            "cmpl 4(%%esi),%%edx\n\t" \
            "ja " REJECT_ "\n\t"
#define SPRITE_ROW_UNCHECKED(REJECT_) ""
#define SPRITE_ROW_SOURCE_CHECK(REJECT_) \
            "testl %%eax,%%eax\n\t" \
            "jz " REJECT_ "\n\t"
#define SPRITE_ROW_SAMPLE(REJECT_,SOURCE_,BOUNDS_) \
            "movl (%%esi),%%eax\n\t" \
            SOURCE_(REJECT_) \
            BOUNDS_(REJECT_) \
            "movzwl (%%eax,%%edx,2),%%eax\n\t" \
            "movl 8(%%esi),%%ecx\n\t" \
            "shrl %%cl,%%eax\n\t" \
            "andl $31,%%eax\n\t" \
            "jz " REJECT_ "\n\t"
#ifdef AB3D_DOS_INDEXED
#define SPRITE_ROW_SKIP "3: movl 12(%%esi),%%eax\n\t"
#define SPRITE_ROW_ADVANCE "leal (%%edi,%%eax),%%edi\n\t"
#define SPRITE_ROW_PIXEL_COUNT ""
#define SPRITE_ROW_FILL \
            "movb (%%ebx,%%eax),%%al\n\t" \
            "movl 12(%%esi),%%ecx\n\t" \
            "rep stosb\n\t"
#else
#define SPRITE_ROW_SKIP "3: movzwl 12(%%esi),%%eax\n\t"
#define SPRITE_ROW_ADVANCE "leal (%%edi,%%eax,2),%%edi\n\t"
#define SPRITE_ROW_PIXEL_COUNT "shrl $1,%%eax\n\t"
#define SPRITE_ROW_FILL \
            "movw (%%ebx,%%eax,2),%%ax\n\t" \
            "movzwl 12(%%esi),%%ecx\n\t" \
            "rep stosw\n\t"
#endif
#define SPRITE_ROW_CLOSE \
            "movl %%edi,%%eax\n\t" \
            "subl (%[span]),%%eax\n\t" \
            SPRITE_ROW_PIXEL_COUNT \
            "movl %%eax,4(%[span])\n\t" \
            "addl $8,%[span]\n\t"
#define SPRITE_ROW_DECODE(SOURCE_,BOUNDS_) \
        __asm__ volatile( \
            "movl %[y],%%edx\n\t" \
            "1: " SPRITE_ROW_SAMPLE("3f",SOURCE_,BOUNDS_) \
            "movl %%edi,(%[span])\n\t" \
            "jmp 2f\n\t" \
            /* Transparent run: skip the destination, then sample again. */ \
            SPRITE_ROW_SKIP \
            SPRITE_ROW_ADVANCE \
            "addl $16,%%esi\n\t" \
            "cmpl %[end],%%esi\n\t" \
            "jne 1b\n\t" \
            "jmp 9f\n\t" \
            /* Opaque run: the span is already open until a cutout. */ \
            "4: " SPRITE_ROW_SAMPLE("6f",SOURCE_,BOUNDS_) \
            "2: " SPRITE_ROW_FILL \
            "addl $16,%%esi\n\t" \
            "cmpl %[end],%%esi\n\t" \
            "jne 4b\n\t" \
            SPRITE_ROW_CLOSE \
            "jmp 9f\n\t" \
            "6: " SPRITE_ROW_CLOSE \
            "jmp 3b\n\t" \
            "9:" \
            : "+&S"(group), "+&D"(destination), [span]"+&r"(span) \
            : "b"(sprite->palette), [y]"m"(source_row), [end]"m"(groups_end) \
            : "eax","ecx","edx","cc","memory")
        /* Most rows are valid in every nonempty source column. Prove that
         * once per band instead of loading and checking each column's limit.
         * Descriptor setup can also prove every source column is nonempty.
         * Truncated data and negative rows retain the unsigned bounds checks. */
        if((unsigned)source_row<=minimum_limit) {
            if(all_sources)
                SPRITE_ROW_DECODE(SPRITE_ROW_UNCHECKED,SPRITE_ROW_UNCHECKED);
            else
                SPRITE_ROW_DECODE(SPRITE_ROW_SOURCE_CHECK,SPRITE_ROW_UNCHECKED);
        } else
            SPRITE_ROW_DECODE(SPRITE_ROW_SOURCE_CHECK,SPRITE_ROW_CHECK);
#undef SPRITE_ROW_DECODE
#undef SPRITE_ROW_SKIP
#undef SPRITE_ROW_FILL
#undef SPRITE_ROW_PIXEL_COUNT
#undef SPRITE_ROW_ADVANCE
#undef SPRITE_ROW_CLOSE
#undef SPRITE_ROW_SAMPLE
#undef SPRITE_ROW_SOURCE_CHECK
#undef SPRITE_ROW_UNCHECKED
#undef SPRITE_ROW_CHECK
        if(batch && end_y-y>1) {
            result.drawn+=sprite_batch_rows(out+y*pitch,end_y-y,pitch,row,
                                             spans,(unsigned)(span-spans));
            y=end_y;continue;
        }
        unsigned opaque=0;
        for(Span *r=spans;r<span;r++) {
            opaque+=r->count;
            unsigned first_copy=batch?y:y+1;
            RendererPixel *target=out+first_copy*pitch+(r->source-decoded_row);
            for(unsigned dy=first_copy;dy<end_y;dy++,target+=pitch) {
                if(batch) {
                    result.drawn+=sprite_batch_copy(target,r->source,r->count,NULL);
                    continue;
                }
                sprite_copy(target,r->source,r->count);
            }
        }
        if(!batch)result.drawn+=opaque*(end_y-y);
        y=end_y;
    }
    if(batch && sprite_history && height<=240)
        sprite_history_note(out,width,height,pitch,sprite,result.visible);
    return result;
}

DosSpriteCoverage dos_raster_sprite_rows(RendererPixel *out,unsigned width,
                                        unsigned height,unsigned pitch,
                                        const DosSpriteRows *sprite)
{
    if(sprite_batch_surface)return sprite_rows(out,width,height,pitch,sprite,1);
    return sprite_rows(out,width,height,pitch,sprite,0);
}

#ifdef AB3D_NATIVE_RGB565
/* Two 1024-bit maps classify every interleaved horizontal/vertical tile row
 * at the authored darkest shade. No pre-shaded texture bank is retained. */
typedef struct {
    const uint8_t *texture;
    const RendererPixel *palette;
    int whole_solid;
    uint8_t solid[2][128];
} DosFloorSolid;
static DosFloorSolid floor_solid[3];

void dos_raster_prepare_floor(unsigned slot,const uint8_t *texture,const uint8_t *palettes)
{
    if(slot>=3)return;
    DosFloorSolid *e=&floor_solid[slot];
    memset(e,0,sizeof(*e));
    if(!texture || !palettes)return;
    e->texture=texture;e->palette=(const RendererPixel *)(const void *)palettes+14*256;
    e->whole_solid=1;
    for(unsigned i=1;i<256;i++)if(e->palette[i]!=e->palette[0]){e->whole_solid=0;break;}
    if(e->whole_solid){memset(e->solid,255,sizeof(e->solid));return;}
    for(unsigned axis=0;axis<2;axis++)for(unsigned key=0;key<1024;key++) {
        unsigned start=axis?(key>>2)+(key&3)*256:(key>>2)*256+(key&3);
        unsigned stride=axis?1024:4,i;
        RendererPixel color=e->palette[texture[start]];
        for(i=1;i<64;i++)if(e->palette[texture[start+i*stride]]!=color)break;
        if(i==64)e->solid[axis][key>>3]|=(uint8_t)(1u<<(key&7));
    }
}

__attribute__((always_inline)) static inline
int dos_raster_solid_floor(RendererPixel *out,unsigned count,const uint8_t *texture,
                            const RendererPixel *palette,uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    for(unsigned i=0;i<3;i++) {
        const DosFloorSolid *e=&floor_solid[i];
        if(palette!=e->palette)continue;
        if(e->whole_solid){fill_words((int16_t *)out,count,palette[0]);return 1;}
        if(du && dv)continue;
        unsigned axis=dv!=0;
        const uint8_t *source=texture+(axis?((u>>14)&0xfc):((v>>6)&0xfc00));
        uintptr_t offset=(uintptr_t)source-(uintptr_t)e->texture;
        if(offset>=65536 || (axis?(offset>>8)>3:(offset&255)>3))continue;
        unsigned key=axis?((offset&255)<<2)+(offset>>8):((offset>>8)<<2)+(offset&3);
        if(e->solid[axis][key>>3]&(1u<<(key&7))) {
            fill_words((int16_t *)out,count,palette[source[0]]);return 1;
        }
    }
    return 0;
}
#endif

void dos_raster_floor_edges_init(int16_t *left, int16_t *right,
                                  unsigned count, int16_t empty_left)
{
    fill_words(left, count, empty_left);
    fill_words(right, count, (uint16_t)-1);
    /* A visible endpoint always replaces its empty bound and writes its
     * brightness at the same time. Invalid rows never consume brightness. */
}

__attribute__((noinline, optimize("O2")))
void dos_raster_floor_edge(int16_t *restrict left, int16_t *restrict right,
                            int16_t *restrict left_bright, int16_t *restrict right_bright,
                            unsigned count, uint32_t x, uint32_t dx, uint32_t bright, uint32_t dbright)
{
    for (unsigned i = 0; i < count; i++) {
        int l = (int32_t)x >> 16;
        int r = l + ((int32_t)x > 0 && (x & 65535) != 0);
        if (l < left[i]) {
            left[i] = (int16_t)l;
            if (left_bright) left_bright[i] = (int16_t)(bright >> 16);
        }
        if (r > right[i]) {
            right[i] = (int16_t)(r > INT16_MAX ? INT16_MAX : r);
            if (right_bright) right_bright[i] = (int16_t)(bright >> 16);
        }
        x += dx;
        bright += dbright;
    }
}

/* WallRoutine3.ChipMem.s: three five-bit columns share each big-endian
 * word. Keep word order within each plane: levels can interpret one texture
 * at different VALSHIFT values. A four-byte prefix records the plane stride. */
uint8_t *dos_load_wall(uint8_t *data, size_t bytes, size_t *source_bytes)
{
    if(bytes<DOS_WALL_HEADER_BYTES+8 || memcmp(data,"ABW1",4)) {
        fprintf(stderr,"Invalid preconverted DOS wall; rebuild the complete DOS package\n");
        exit(1);
    }
    uint32_t native=*(const uint32_t *)(const void *)(data+4);
#ifdef AB3D_NATIVE_RGB565
    const uint32_t expected_native=1;
#else
    const uint32_t expected_native=0;
#endif
    uint8_t *palette=data+8;
    uint32_t words=*(const uint32_t *)(const void *)(palette+DOS_WALL_PALETTE_BYTES);
    size_t payload=bytes-8-DOS_WALL_HEADER_BYTES;
    if(native!=expected_native || !words || payload%3 || payload/3!=words) {
        fprintf(stderr,"Invalid DOS wall format/plane size; rebuild the matching DOS package\n");exit(1);
    }
    for(size_t i=0;i<payload;i++)if(palette[DOS_WALL_HEADER_BYTES+i]>31) {
        fprintf(stderr,"Invalid DOS wall texel at %lu\n",(unsigned long)i);exit(1);
    }
    *source_bytes=DOS_WALL_PALETTE_BYTES+(size_t)words*2;
    return palette;
}

__attribute__((noinline, optimize("O2")))
void dos_raster_wall_rows(RendererPixel *restrict out, unsigned width, unsigned height, unsigned pitch,
                           const DosWallColumn *restrict columns, const RendererPixel *shared_palette,
                           uint32_t y, uint32_t dy, unsigned mask)
{
#ifdef AB3D_NATIVE_RGB565
    /* A caller with constant brightness already knows the palette. Its column
     * palette fields are unused; varying spans retain the exact classification. */
    if (!shared_palette) {
        shared_palette = width ? columns[0].palette : NULL;
        for (unsigned x = 1; x < width; x++) {
            if (columns[x].palette != shared_palette) { shared_palette = NULL; break; }
        }
    }
#else
    (void)shared_palette;
#endif
    unsigned previous_sample = ~0u;
    for (unsigned row = 0; row < height; row++) {
        unsigned sample = (y >> 16) & mask;
        if (sample == previous_sample)
            memcpy(out, out - pitch, width * sizeof(*out));
        else {
#ifdef AB3D_NATIVE_RGB565
            if (width) {
                RendererPixel *dest = out;
                const DosWallColumn *source = columns;
                unsigned count = width;
                if (shared_palette) {
#define WALL_ROW_PIXEL(COLUMN, PIXEL) \
                    "movl " #COLUMN "(%%esi), %%eax\n\t" \
                    "movzbl (%%eax,%%ebx), %%eax\n\t" \
                    "movw (%[pal],%%eax,2), %%ax\n\t" \
                    "movw %%ax, " #PIXEL "(%%edi)\n\t"
                    unsigned groups = count / 8;
                    if (groups) __asm__ volatile(
                        "1:\n\t"
                        WALL_ROW_PIXEL(0, 0)
                        WALL_ROW_PIXEL(8, 2)
                        WALL_ROW_PIXEL(16, 4)
                        WALL_ROW_PIXEL(24, 6)
                        WALL_ROW_PIXEL(32, 8)
                        WALL_ROW_PIXEL(40, 10)
                        WALL_ROW_PIXEL(48, 12)
                        WALL_ROW_PIXEL(56, 14)
                        "addl $64, %%esi\n\t"
                        "addl $16, %%edi\n\t"
                        "decl %%ecx\n\t"
                        "jnz 1b"
                        : "+&D"(dest), "+&S"(source), "+&c"(groups)
                        : "b"(sample), [pal] "r"(shared_palette) : "eax", "cc", "memory");
                    count &= 7;
                    if (count) __asm__ volatile(
                        "1:\n\t"
                        WALL_ROW_PIXEL(0, 0)
                        "addl $8, %%esi\n\t"
                        "addl $2, %%edi\n\t"
                        "decl %%ecx\n\t"
                        "jnz 1b"
                        : "+&D"(dest), "+&S"(source), "+&c"(count)
                        : "b"(sample), [pal] "r"(shared_palette) : "eax", "cc", "memory");
#undef WALL_ROW_PIXEL
                } else {
                    __asm__ volatile(
                    "1: movl (%%esi), %%eax\n\t"
                    "movl 4(%%esi), %%edx\n\t"
                    "movzbl (%%eax,%%ebx), %%eax\n\t"
                    "movw (%%edx,%%eax,2), %%ax\n\t"
                    "movw %%ax, (%%edi)\n\t"
                    "addl $8, %%esi\n\t"
                    "addl $2, %%edi\n\t"
                    "decl %%ecx\n\t"
                    "jnz 1b"
                    : "+&D"(dest), "+&S"(source), "+&c"(count)
                    : "b"(sample) : "eax", "edx", "cc", "memory");
                }
            }
#else
            for (unsigned x = 0; x < width; x++)
                out[x] = columns[x].palette[columns[x].texels[sample]];
#endif
        }
        previous_sample = sample;
        out += pitch;
        y += dy;
    }
}

__attribute__((noinline, optimize("O2")))
void dos_raster_wall(RendererPixel *restrict out, unsigned count, unsigned stride,
                      const uint8_t *restrict column, const RendererPixel *restrict palette,
                      uint32_t y, uint32_t dy, unsigned mask)
{
#ifdef AB3D_NATIVE_RGB565
    if (!count) return;
    /* Native 320-pixel modes can address eight rows directly. Keep every
     * original fixed-point sample, while stepping the destination once. */
    if (stride == 320 && count >= 8) {
#define WALL_COLUMN_PIXEL(OFFSET, SAMPLE_SHIFT, MASK_OP) \
        "movl %%eax, %%edx\n\t" \
        "shrl $" #SAMPLE_SHIFT ", %%edx\n\t" \
        MASK_OP \
        "movzbl (%%esi,%%edx), %%edx\n\t" \
        "movw (%[pal],%%edx,2), %%dx\n\t" \
        "movw %%dx, " #OFFSET "(%%edi)\n\t" \
        "addl %%ebx, %%eax\n\t"
#define WALL_COLUMN_GROUPS(COORD_SHIFT, SAMPLE_SHIFT, MASK_OP) do { \
        unsigned groups = count / 8, tail = count & 7; \
        uint32_t coord = y << COORD_SHIFT, delta = dy << COORD_SHIFT; \
        __asm__ volatile( \
            "1:\n\t" \
            WALL_COLUMN_PIXEL(0, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(640, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(1280, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(1920, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(2560, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(3200, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(3840, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(4480, SAMPLE_SHIFT, MASK_OP) \
            "addl $5120, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coord), "+&D"(out), "+&c"(groups) \
            : "S"(column), "b"(delta), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        if (tail) __asm__ volatile( \
            "1:\n\t" \
            WALL_COLUMN_PIXEL(0, SAMPLE_SHIFT, MASK_OP) \
            "addl $640, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coord), "+&D"(out), "+&c"(tail) \
            : "S"(column), "b"(delta), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        return; \
        } while (0)
        /* Shift the authored wrapping bits to the top of the accumulator.
         * Unsigned addition commutes with that shift, so a single right
         * shift gives the exact sample without loading a mask per pixel. */
        switch (mask) {
            case 15:  WALL_COLUMN_GROUPS(12, 28, ""); break;
            case 31:  WALL_COLUMN_GROUPS(11, 27, ""); break;
            case 63:  WALL_COLUMN_GROUPS(10, 26, ""); break;
            case 127: WALL_COLUMN_GROUPS(9, 25, ""); break;
            case 255: WALL_COLUMN_GROUPS(8, 24, ""); break;
            default:  WALL_COLUMN_GROUPS(0, 16, "andl %[mask], %%edx\n\t"); break;
        }
#undef WALL_COLUMN_GROUPS
#undef WALL_COLUMN_PIXEL
    }
    unsigned stride_bytes = stride * sizeof(*out);
    __asm__ volatile(
        "1: movl %%eax, %%edx\n\t"
        "shrl $16, %%edx\n\t"
        "andl %[mask], %%edx\n\t"
        "movzbl (%%esi,%%edx), %%edx\n\t"
        "movw (%[pal],%%edx,2), %%dx\n\t"
        "movw %%dx, (%%edi)\n\t"
        "addl %%ebx, %%eax\n\t"
        "addl %[stride], %%edi\n\t"
        "decl %%ecx\n\t"
        "jnz 1b"
        : "+&a"(y), "+&D"(out), "+&c"(count)
        : "S"(column), "b"(dy), [pal] "r"(palette), [mask] "m"(mask), [stride] "m"(stride_bytes)
        : "edx", "cc", "memory");
#else
    for (unsigned i = 0; i < count; i++) {
        *out = palette[column[(y >> 16) & mask]];
        out += stride;
        y += dy;
    }
#endif
}

/* Side extensions use the completed core's exact sample and shade. Store all
 * two/three adjacent words while that color is still in a register. */
__attribute__((noinline, optimize("O2")))
void dos_raster_wall_edges(RendererPixel *restrict out, unsigned count, unsigned stride,
                            const uint8_t *restrict column, const RendererPixel *restrict palette,
                            uint32_t y, uint32_t dy, unsigned mask, unsigned copies)
{
#ifdef AB3D_NATIVE_RGB565
    if (stride == 320) {
#define EDGE_STORE_TWO(OFFSET) "movw %%dx, " #OFFSET "+2(%%edi)\n\t"
#define EDGE_STORE_THREE(OFFSET) EDGE_STORE_TWO(OFFSET) "movw %%dx, " #OFFSET "+4(%%edi)\n\t"
#define EDGE_PIXEL(OFFSET, EXTRA) \
        "movl %%eax, %%edx\n\t" \
        "shrl $16, %%edx\n\t" \
        "andl %[mask], %%edx\n\t" \
        "movzbl (%%esi,%%edx), %%edx\n\t" \
        "movw (%[pal],%%edx,2), %%dx\n\t" \
        "movw %%dx, " #OFFSET "(%%edi)\n\t" \
        EXTRA(OFFSET) \
        "addl %%ebx, %%eax\n\t"
#define EDGE_COLUMNS(EXTRA) do { \
        unsigned groups = count / 8; \
        if (groups) __asm__ volatile( \
            "1:\n\t" \
            EDGE_PIXEL(0, EXTRA) \
            EDGE_PIXEL(640, EXTRA) \
            EDGE_PIXEL(1280, EXTRA) \
            EDGE_PIXEL(1920, EXTRA) \
            EDGE_PIXEL(2560, EXTRA) \
            EDGE_PIXEL(3200, EXTRA) \
            EDGE_PIXEL(3840, EXTRA) \
            EDGE_PIXEL(4480, EXTRA) \
            "addl $5120, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(y), "+&D"(out), "+&c"(groups) \
            : "S"(column), "b"(dy), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        count &= 7; \
        if (count) __asm__ volatile( \
            "1:\n\t" \
            EDGE_PIXEL(0, EXTRA) \
            "addl $640, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(y), "+&D"(out), "+&c"(count) \
            : "S"(column), "b"(dy), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        } while (0)
        if (copies == 3) EDGE_COLUMNS(EDGE_STORE_THREE);
        else EDGE_COLUMNS(EDGE_STORE_TWO);
#undef EDGE_COLUMNS
#undef EDGE_PIXEL
#undef EDGE_STORE_THREE
#undef EDGE_STORE_TWO
        return;
    }
#endif
    for (unsigned row = 0; row < count; row++, out += stride, y += dy) {
        RendererPixel color = palette[column[(y >> 16) & mask]];
        out[0] = out[1] = color;
        if (copies == 3) out[2] = color;
    }
}

#ifdef AB3D_NATIVE_RGB565
__attribute__((noinline, optimize("O2")))
void dos_raster_floor_dark(RendererPixel *out,unsigned count,const uint8_t *texture,
                            const RendererPixel *palette,uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    if(count>=16 && (du || dv) && dos_raster_solid_floor(out,count,texture,palette,u,v,du,dv))return;
    dos_raster_floor(out,count,texture,palette,u,v,du,dv);
}
#endif

/* Separate small loops keep the water/projection state out of 486 registers.
 * Unsigned additions retain the renderer's exact 32-bit UV wrapping. */
__attribute__((noinline, optimize("O2")))
void dos_raster_floor(RendererPixel *restrict out, unsigned count, const uint8_t *restrict texture,
                       const RendererPixel *restrict palette, uint32_t u, uint32_t v,
                       uint32_t du, uint32_t dv)
{
    if (!count) return;
#ifdef AB3D_NATIVE_RGB565
    /* The 486 has seven general registers. Keep UV, source, destination and
     * palette in them; only the two steps/end pointer need stack operands. */
    if (!dv || !du) {
        uint32_t coordinate, step;
        if (!dv) {
            texture += (v >> 6) & 0xfc00;
            coordinate = u; step = du;
        } else {
            texture += (u >> 14) & 0xfc;
            coordinate = v; step = dv;
        }
        uint32_t magnitude=(int32_t)step<0?0u-step:step;
        if(magnitude<=32768) {
            /* AB3DI sideloop: at most half a texel per pixel. Walk the exact
             * 16-bit fraction and fetch a new color only on carry/borrow. */
#define FLOOR_CACHED_PIXEL(SHIFT, MASK, OP, OFFSET, LABEL) \
            "movw %%dx, " #OFFSET "(%%edi)\n\t" \
            #OP "w %%bx, %%ax\n\t" \
            "jnc " #LABEL "f\n\t" \
            #OP "l $65536, %%eax\n\t" \
            "movl %%eax, %%edx\n\t" \
            "shrl $" #SHIFT ", %%edx\n\t" \
            "andl $" #MASK ", %%edx\n\t" \
            "movzbl (%%esi,%%edx), %%edx\n\t" \
            "movw (%[pal],%%edx,2), %%dx\n\t" \
            #LABEL ":\n\t"
#define FLOOR_CACHED_AXIS(SHIFT, MASK, OP) do { \
            unsigned color=palette[texture[(coordinate>>SHIFT)&MASK]]; \
            if(!magnitude){fill_words((int16_t *)out,count,(uint16_t)color);return;} \
            unsigned groups=count/8; \
            if(groups)__asm__ volatile( \
                "1:\n\t" \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,0,2) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,2,3) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,4,4) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,6,5) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,8,6) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,10,7) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,12,8) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,14,9) \
                "addl $16, %%edi\n\t" \
                "decl %%ecx\n\t" \
                "jnz 1b" \
                : "+&a"(coordinate), "+&D"(out), "+&c"(groups), "+&d"(color) \
                : "S"(texture), "b"(magnitude), [pal] "r"(palette) : "cc", "memory"); \
            count&=7; \
            if(count)__asm__ volatile( \
                "1:\n\t" \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,0,2) \
                "addl $2, %%edi\n\t" \
                "decl %%ecx\n\t" \
                "jnz 1b" \
                : "+&a"(coordinate), "+&D"(out), "+&c"(count), "+&d"(color) \
                : "S"(texture), "b"(magnitude), [pal] "r"(palette) : "cc", "memory"); \
            } while(0)
            if(!dv) {
                if((int32_t)step<0)FLOOR_CACHED_AXIS(14,0xfc,sub);
                else FLOOR_CACHED_AXIS(14,0xfc,add);
            } else {
                if((int32_t)step<0)FLOOR_CACHED_AXIS(6,0xfc00,sub);
                else FLOOR_CACHED_AXIS(6,0xfc00,add);
            }
#undef FLOOR_CACHED_AXIS
#undef FLOOR_CACHED_PIXEL
            return;
        }
#define FLOOR_AXIS_PIXEL(SHIFT, MASK, OFFSET) \
            "movl %%eax, %%edx\n\t" \
            "shrl $" #SHIFT ", %%edx\n\t" \
            "andl $" #MASK ", %%edx\n\t" \
            "movzbl (%%esi,%%edx), %%edx\n\t" \
            "movw (%[pal],%%edx,2), %%dx\n\t" \
            "movw %%dx, " #OFFSET "(%%edi)\n\t" \
            "addl %%ebx, %%eax\n\t"
#define FLOOR_AXIS(SHIFT, MASK) do { \
        unsigned groups = count / 8; \
        if (groups) __asm__ volatile( \
            "1:\n\t" \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 0) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 2) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 4) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 6) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 8) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 10) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 12) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 14) \
            "addl $16, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coordinate), "+&D"(out), "+&c"(groups) \
            : "S"(texture), "b"(step), [pal] "r"(palette) \
            : "edx", "cc", "memory"); \
        count &= 7; \
        if (count) __asm__ volatile( \
            "1:\n\t" \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 0) \
            "addl $2, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coordinate), "+&D"(out), "+&c"(count) \
            : "S"(texture), "b"(step), [pal] "r"(palette) \
            : "edx", "cc", "memory"); \
        } while (0)
        if (!dv) FLOOR_AXIS(14, 0xfc);
        else FLOOR_AXIS(6, 0xfc00);
#undef FLOOR_AXIS
#undef FLOOR_AXIS_PIXEL
        return;
    }
#define FLOOR_UV_PIXEL(OFFSET) \
        "movl %%eax, %%ecx\n\t" \
        "shrl $14, %%ecx\n\t" \
        "andl $0xfc, %%ecx\n\t" \
        "movl %%ebx, %%edx\n\t" \
        "shrl $6, %%edx\n\t" \
        "andl $0xfc00, %%edx\n\t" \
        "orl %%edx, %%ecx\n\t" \
        "movzbl (%%esi,%%ecx), %%ecx\n\t" \
        "movw (%[pal],%%ecx,2), %%cx\n\t" \
        "movw %%cx, " #OFFSET "(%%edi)\n\t" \
        "addl %[du], %%eax\n\t" \
        "addl %[dv], %%ebx\n\t"
    if (count >= 8) {
        RendererPixel *groups_end = out + (count & ~7u);
        __asm__ volatile(
            "1:\n\t"
            FLOOR_UV_PIXEL(0)
            FLOOR_UV_PIXEL(2)
            FLOOR_UV_PIXEL(4)
            FLOOR_UV_PIXEL(6)
            FLOOR_UV_PIXEL(8)
            FLOOR_UV_PIXEL(10)
            FLOOR_UV_PIXEL(12)
            FLOOR_UV_PIXEL(14)
            "addl $16, %%edi\n\t"
            "cmpl %[end], %%edi\n\t"
            "jne 1b"
            : "+&a"(u), "+&b"(v), "+&D"(out)
            : "S"(texture), [pal] "r"(palette), [du] "m"(du), [dv] "m"(dv), [end] "m"(groups_end)
            : "ecx", "edx", "cc", "memory");
        count &= 7;
    }
    if (count) {
        RendererPixel *native_end = out + count;
        __asm__ volatile(
            "1:\n\t"
            FLOOR_UV_PIXEL(0)
            "addl $2, %%edi\n\t"
            "cmpl %[end], %%edi\n\t"
            "jne 1b"
            : "+&a"(u), "+&b"(v), "+&D"(out)
            : "S"(texture), [pal] "r"(palette), [du] "m"(du), [dv] "m"(dv), [end] "m"(native_end)
            : "ecx", "edx", "cc", "memory");
    }
#undef FLOOR_UV_PIXEL
    return;
#endif
    RendererPixel *end = out + count;
    if (!dv) {
        texture += (v >> 6) & 0xfc00;
        while (out != end) {
            *out++ = palette[texture[(u >> 14) & 0xfc]];
            u += du;
        }
        return;
    }
    if (!du) {
        texture += (u >> 14) & 0xfc;
        while (out != end) {
            *out++ = palette[texture[(v >> 6) & 0xfc00]];
            v += dv;
        }
        return;
    }
    while (out != end) {
        *out++ = palette[texture[((u >> 14) & 0xfc) | ((v >> 6) & 0xfc00)]];
        u += du;
        v += dv;
    }
}

__attribute__((noinline, optimize("O2")))
void dos_raster_floor_gouraud(RendererPixel *restrict out, unsigned count, const uint8_t *restrict texture,
                               const RendererPixel (*restrict palettes)[256],
                               uint32_t u, uint32_t v, uint32_t du, uint32_t dv,
                               int32_t level, int32_t dlevel)
{
    if (!dlevel) {
        dos_raster_floor_shaded(out, count, texture, palettes, level >> 16, u, v, du, dv);
        return;
    }
    unsigned gradient = dlevel < 0 ? 0u - (uint32_t)dlevel : (uint32_t)dlevel;
    /* Long constant-shade runs keep Gouraud selection out of the pixel loop.
     * Run boundaries are exact for either sign of the 16.16 gradient. */
    if (gradient < 8192) {
        while (count) {
            unsigned fraction = (uint32_t)level & 65535;
            unsigned run = dlevel < 0 ? fraction / gradient + 1
                : (65536 - fraction + gradient - 1) / gradient;
            if (run > count) run = count;
            dos_raster_floor_shaded(out, run, texture, palettes, level >> 16, u, v, du, dv);
            out += run;
            u += du * run;
            v += dv * run;
            level += dlevel * (int32_t)run;
            count -= run;
        }
        return;
    }
    RendererPixel *end = out + count;
    if (!dv) {
        texture += (v >> 6) & 0xfc00;
        while (out != end) {
            *out++ = palettes[level >> 16][texture[(u >> 14) & 0xfc]];
            u += du;
            level += dlevel;
        }
        return;
    }
    if (!du) {
        texture += (u >> 14) & 0xfc;
        while (out != end) {
            *out++ = palettes[level >> 16][texture[(v >> 6) & 0xfc00]];
            v += dv;
            level += dlevel;
        }
        return;
    }
    while (out != end) {
        /* Both gradient endpoints are clamped to 0..14 by the caller. */
        *out++ = palettes[level >> 16][texture[((u >> 14) & 0xfc) | ((v >> 6) & 0xfc00)]];
        u += du;
        v += dv;
        level += dlevel;
    }
}
