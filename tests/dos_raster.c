/* Compare the DOS kernels with the existing WallRoutine3/AB3DI expressions. */
#include "raster_dos.h"
#include "renderer_fixed.h"
#include "objects_dos.h"
#include "sprite_palettes.h"
#include "sprite_format.h"
#include "water_format.h"
#include "level.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reference contains the original big-endian .pal bytes; the linked
 * sprite_palettes.c contains the actual native data used by the game. */
#define sprite_pal_embedded sprite_pal_source
#define sprite_pal_embedded_size sprite_pal_source_size
#include "sprite_palettes_source.h"
#undef sprite_pal_embedded
#undef sprite_pal_embedded_size

static uint32_t random_state = 17;
static uint32_t next_value(void)
{
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

static void check_water_transport(void)
{
    static uint8_t native[WATER_FILE_BYTES],original[WATER_FILE_BYTES];
    uint8_t header[WATER_NATIVE_HEADER_BYTES];
    FILE *file=fopen("WATER.BIN","rb");assert(file);
    assert(fread(header,1,sizeof(header),file)==sizeof(header));
    assert(water_native_payload_size(header,WATER_FILE_BYTES+WATER_NATIVE_HEADER_BYTES)==WATER_FILE_BYTES);
    assert(!water_native_payload_size(header,WATER_FILE_BYTES+WATER_NATIVE_HEADER_BYTES-1));
    assert(!water_native_payload_size(header,7));
    header[3]='2';assert(!water_native_payload_size(header,WATER_FILE_BYTES+WATER_NATIVE_HEADER_BYTES));header[3]='1';
    header[4]=1;assert(!water_native_payload_size(header,WATER_FILE_BYTES+WATER_NATIVE_HEADER_BYTES));header[4]=0;
    assert(fread(native,1,sizeof(native),file)==sizeof(native) && fgetc(file)==EOF);fclose(file);
    file=fopen("WATERRAW.BIN","rb");assert(file);
    assert(fread(original,1,sizeof(original),file)==sizeof(original) && fgetc(file)==EOF);fclose(file);
    for(unsigned offset=0;offset<WATER_FILE_BYTES;offset++)
        assert(water_source_byte(native,offset)==original[offset]);
    /* Independent AB3DI.s d5/phase address expressions verify the direct
     * views consumed by the actual water raster, including all eight phases. */
    static const unsigned offsets[8]={0,2,256,258,512,514,768,770};
    for(unsigned phase=0;phase<8;phase++)for(unsigned v=0;v<64;v++)for(unsigned u=0;u<64;u++) {
        unsigned source=(((v<<8)|u)<<2)+offsets[phase];
        assert(native[phase*4096+v*64+u]==original[source]);
    }
    puts("PASS: all 65536 transported water bytes, 32768 phase samples and invalid headers");
}

static RendererPixel reference_water_pixel(RendererPixel p0,RendererPixel p1,
                                           const uint8_t *brighten,unsigned row,unsigned alpha)
{
    unsigned s0=renderer_pixel_gb_index(p0),s1=renderer_pixel_gb_index(p1);
    const uint8_t *b0=brighten+(row*256+s0)*2,*b1=brighten+(row*256+s1)*2;
    unsigned c0=(b0[0]<<8)|b0[1],c1=(b1[0]<<8)|b1[1],result=0;
    for(unsigned shift=0;shift<=8;shift+=4) {
        unsigned a=((c0>>shift)&15)*17,b=((c1>>shift)&15)*17;
        unsigned byte=(a*(255-alpha)+b*alpha)/255;
        result|=((byte+8)/17)<<shift;
    }
    return renderer_pixel_from_c12((uint16_t)result);
}

static void check_water(void)
{
    enum { PITCH=324, ROWS=5 };
    static uint8_t brighten[22*512],phase[4096];
    static RendererPixel actual[PITCH*ROWS],expected[PITCH*ROWS];
    for(unsigned row=0;row<22;row++)for(unsigned sample=0;sample<256;sample++) {
        unsigned color=row?((row*197+sample*37)&4095):
            ((((sample*3)&15)<<8)|(((sample*7)&15)<<4)|(sample&15));
        brighten[(row*256+sample)*2]=(uint8_t)(color>>8);
        brighten[(row*256+sample)*2+1]=(uint8_t)color;
    }
    dos_raster_prepare_water(brighten);
    for(unsigned row=0;row<22;row++)for(unsigned sample=0;sample<256;sample++) {
        unsigned at=(row*256+sample)*2;
        assert(dos_raster_water_color(row,sample)==renderer_pixel_from_c12((uint16_t)(((unsigned)brighten[at]<<8)|brighten[at+1])));
    }
    for(unsigned pair=0;pair<256;pair++) {
        actual[pair]=renderer_pixel_from_c12((uint16_t)(pair>>4));
        actual[PITCH+pair]=renderer_pixel_from_c12((uint16_t)(pair&15));
    }
    memset(phase,0,sizeof(phase));
    for(unsigned alpha=0;alpha<256;alpha++) {
        dos_raster_water(actual+PITCH*2,256,actual,actual+PITCH,phase,0,0,alpha,0,0,0,0);
        for(unsigned pair=0;pair<256;pair++)
            assert(actual[PITCH*2+pair]==reference_water_pixel(actual[pair],actual[PITCH+pair],brighten,0,alpha));
    }
    /* Every native RGB565 word, including noncanonical channel values,
     * must retain the original low-byte Amiga green/blue selection. */
    for(unsigned block=0;block<256;block++) {
        for(unsigned x=0;x<256;x++)actual[x]=(RendererPixel)((block<<8)|x);
        dos_raster_water(actual+PITCH*2,256,actual,NULL,phase,1,0,231,0,0,0,0);
        for(unsigned x=0;x<256;x++)
            assert(actual[PITCH*2+x]==reference_water_pixel(actual[x],actual[x],brighten,1,0));
    }
    for(unsigned i=0;i<sizeof(phase);i++)phase[i]=(uint8_t)(next_value()%10);
    static const unsigned widths[]={0,1,2,3,7,17,159,319,320};
    for(unsigned test=0;test<768;test++) {
        for(unsigned i=0;i<PITCH*ROWS;i++)actual[i]=expected[i]=(RendererPixel)next_value();
        unsigned count=widths[test%9],shade=next_value()%13;
        unsigned offset=test%9==0?65535u:test%9==1?65536u:test%64,alpha=test%256;
        int single=test>=384;
        unsigned out=PITCH*2+2,src0=(test%5)*PITCH+2,src1=((test+1)%5)*PITCH+2;
        uint32_t u=next_value(),v=next_value(),du=next_value(),dv=next_value(),ru=u,rv=v;
        dos_raster_water(actual+out,count,actual+src0,single?NULL:actual+src1,phase,shade,offset,alpha,u,v,du,dv);
        for(unsigned x=0;x<count;x++,ru+=du,rv+=dv) {
            uint8_t ub=(uint8_t)(ru>>16),vb=(uint8_t)(rv>>16);
            uint16_t uv=(uint16_t)(((uint16_t)vb<<8)|ub);
            uv=(uint16_t)(uv+offset);uv&=0x3f3f;
            unsigned level=phase[((uv>>8)<<6)|(uv&63)];
            expected[out+x]=reference_water_pixel(expected[src0+x],single?expected[src0+x]:expected[src1+x],brighten,shade+level,single?0:alpha);
        }
        assert(!memcmp(actual,expected,sizeof(actual)));
    }
    printf("Water: 5632 shared colours, 65536 native inputs, 65536 alpha/colour pairs and 768 single/blended wrapping/alias/clip spans passed\n");
}

__attribute__((noinline)) static int64_t reference_divide(int64_t n, int64_t d) { return n / d; }

static void check_sprite_words(void)
{
    FILE *file=fopen("SWORDS.BIN","rb");assert(file);
    uint8_t header[8],samples[514];
    assert(fread(header,1,sizeof(header),file)==sizeof(header));
    assert(sprite_native_payload_size(header,131080)==131072);
    assert(!sprite_native_payload_size(header,131079));
    assert(!sprite_native_payload_size(header,7));
    header[3]='2';assert(!sprite_native_payload_size(header,131080));header[3]='1';
    uint8_t empty[8]={'A','B','S','1',0,0,0,0};
    assert(!sprite_native_payload_size(empty,8));
    empty[4]=1;assert(!sprite_native_payload_size(empty,9));
    for(unsigned start=0;start<65536;start+=256) {
        assert(fread(samples+1,1,512,file)==512);
        for(unsigned i=0;i<256;i++) {
            unsigned word=start+i;
            assert(sprite_word_read(samples+1+i*2)==word);
            uint8_t written[3]={0};sprite_word_write(written+1,(uint16_t)word);
            assert(written[1]==(word&255) && written[2]==(word>>8));
        }
    }
    assert(fgetc(file)==EOF);fclose(file);
    puts("PASS: all 65536 prepared sprite words, unaligned reads/writes and invalid transport headers");
}

static void check_sprite_columns(void)
{
    enum { CAPACITY=41*242+8 };
    static RendererPixel actual[CAPACITY],expected[CAPACITY],palette[32];
    static uint8_t packed[513],native[513],texels[256];
    static int rows[240];
    static const unsigned counts[]={0,1,2,3,7,31,127,239,240};
    static const unsigned strides[]={1,2,3,7,41};
    static const unsigned shifts[]={0,5,10};
    for(unsigned i=0;i<sizeof(packed);i++)packed[i]=(uint8_t)next_value();
    for(unsigned i=0;i<256;i++){native[1+i*2]=packed[2+i*2];native[2+i*2]=packed[1+i*2];}
    for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)(i==1?0:i*1973);
    assert(!dos_raster_sprite_column(NULL,0,0,NULL,NULL,NULL,0,0));
    assert(!dos_raster_sprite_texel_column(NULL,0,0,NULL,NULL,NULL,0));
    for(unsigned test=0;test<384;test++) {
        unsigned count=counts[test%9],stride=strides[test%5],shift=shifts[test%3];
        unsigned limit=test%256,drawn=0;
        const uint8_t *source=packed+1; /* Include unaligned packed words. */
        for(unsigned i=0;i<256;i++) {
            unsigned word=((unsigned)source[i*2]<<8)|source[i*2+1];
            texels[i]=(uint8_t)((word>>shift)&31);
        }
        for(unsigned y=0;y<240;y++) {
            rows[y]=(int)(next_value()%340);
            if(y%31==0)rows[y]=-1;
            if(y%47==0)rows[y]=INT_MAX;
        }
        for(unsigned i=0;i<CAPACITY;i++)actual[i]=expected[i]=(RendererPixel)next_value();
        for(unsigned y=0;y<count;y++) {
            if((unsigned)rows[y]>limit)continue;
            const uint8_t *p=source+rows[y]*2;
            unsigned word=((unsigned)p[0]<<8)|p[1],texel=(word>>shift)&31;
            if(texel){expected[3+y*stride]=palette[texel];drawn++;}
        }
        assert(dos_raster_sprite_column(actual+3,count,stride,native+1,palette,rows,limit,shift)==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        /* Both source representations must write exactly the same pixels.
         * Reset only the column so all surrounding guards remain checked. */
        for(unsigned y=0;y<count;y++)actual[3+y*stride]=0x6b5a;
        memcpy(expected,actual,sizeof(actual));
        for(unsigned y=0;y<count;y++)if((unsigned)rows[y]<=limit && texels[rows[y]])
            expected[3+y*stride]=palette[texels[rows[y]]];
        assert(dos_raster_sprite_texel_column(actual+3,count,stride,texels,palette,rows,limit)==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
    }
    puts("PASS: 768 packed/decoded sprite columns preserve clipping, modes, cutouts, opaque black, bounds and guards");
}

static void check_sprite_rows(void)
{
    static RendererPixel actual[324*242],expected[324*242];
    static uint8_t wad[4099],native[4099],ptr[131];
    static RendererPixel palette[32];
    static int rows[240];
    static const unsigned widths[]={0,1,2,3,7,31,127,319,320};
    uint32_t seed=0x13579bdf;unsigned checked=0;
    for(unsigned i=0;i<sizeof(wad);i++) {
        seed=seed*1664525u+1013904223u;wad[i]=(uint8_t)(seed>>24);
    }
    for(unsigned i=0;i+1<sizeof(wad);i+=2){native[i]=wad[i+1];native[i+1]=wad[i];}
    /* The isolated final column deliberately starts at an odd address. Its
     * native words do not overlap any other sampled column in this fixture. */
    native[4097]=wad[4098];native[4098]=wad[4097];
    for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)(i==1?0:i*1973u);
    for(unsigned test=0;test<540;test++) {
        unsigned width=widths[test%9],height=test==287?240:1+(test%61);
        for(unsigned i=1;i<32;i++) {
            /* Adjacent opaque samples may share a colour, including black.
             * Alpha still comes from the source texel rather than that colour. */
            palette[i]=(RendererPixel)(test%4==0?0:test%4==1?1:
                                       test%4==2?(i/8)*1973:i*1973);
        }
        unsigned pitch=width+4;
        for(unsigned i=0;i<32;i++) {
            unsigned offset=2+i*64;
            if(i==0 && test%3==0)offset=0;
            if(i==31)offset=4097;
            ptr[i*4]=(uint8_t)((i+test)%4);
            ptr[i*4+1]=(uint8_t)(offset>>16);ptr[i*4+2]=(uint8_t)(offset>>8);ptr[i*4+3]=(uint8_t)offset;
        }
        for(unsigned i=0;i<height;i++)rows[i]=(int)(i/(1+test%7))-(test%11==0?1:0);
        for(unsigned i=0;i<sizeof(actual)/sizeof(*actual);i++)actual[i]=expected[i]=0xa55a;
        DosSpriteRows sprite={native,ptr,sizeof(wad)-(test%5),sizeof(ptr)-(test%7),palette,rows,
                              test%4,(test%3)*65536u+333u,test%4==0?0:65536u/(1+test%8),
                              32,test%13==0?65535u:test%3};
        if(test>=288) {
            /* Compare the row renderer with the independent per-pixel DDA
             * across fractional boundaries, clamped columns and 32-bit wraps. */
            static const uint32_t steps[]={0,1,2,3,127,4095,4096,4097,8191,8192,8193,
                16383,16384,16385,32767,32768,32769,65535,65536,65537,UINT32_MAX};
            static const uint32_t origins[]={0,1,65534,65535,65536,65537,31*65536u,
                                             32*65536u-1u,32*65536u,UINT32_MAX-65535u,
                                             UINT32_MAX-1u,UINT32_MAX};
            unsigned n=test-288;
            sprite.du=steps[n%21];sprite.u=origins[n/21];
            if(n%7==0)sprite.columns=1;
        }
        DosSpriteCoverage coverage={0,0};
        unsigned available=(sprite.ptr_bytes-sprite.ptr_offset)/4;
        for(unsigned x=0;x<width;x++) {
            unsigned column=(sprite.u+x*sprite.du)>>16;
            if(column>=sprite.columns)column=sprite.columns-1;
            if(column>=available)column=available-1;
            const uint8_t *entry=ptr+sprite.ptr_offset+column*4;
            unsigned offset=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
            unsigned mode=entry[0];
            if((!mode && !offset) || (uint64_t)offset+(uint64_t)sprite.down_strip*2+2>sprite.wad_bytes)continue;
            coverage.visible+=height;
            for(unsigned y=0;y<height;y++) {
                if(rows[y]<0)continue;
                uint64_t address=(uint64_t)offset+((uint64_t)sprite.down_strip+(unsigned)rows[y])*2;
                if(address+2>sprite.wad_bytes)continue;
                /* Malformed PTR views can point at odd byte offsets. Their
                 * bounds and samples follow the native transport ABI, while
                 * valid authored columns retain the original word values. */
                unsigned word=(unsigned)native[address]|((unsigned)native[address+1]<<8);
                unsigned texel=(word>>(!mode?0:mode==1?5:10))&31;
                if(texel){expected[(y+1)*pitch+x+2]=palette[texel];coverage.drawn++;}
            }
        }
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+pitch+2,width,height,pitch,&sprite);
        if(memcmp(actual,expected,sizeof(actual))) {
            for(unsigned i=0;i<sizeof(actual)/sizeof(*actual);i++)if(actual[i]!=expected[i]) {
                fprintf(stderr,"Sprite row case %u, pixel %u, width=%u height=%u: actual=%04x expected=%04x\n",
                        test,i,width,height,(unsigned)actual[i],(unsigned)expected[i]);break;
            }
        }
        assert(got.visible==coverage.visible && got.drawn==coverage.drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        checked++;
    }
    /* Exercise both run states at their first/last pixel and the full
     * 160-span capacity. Texel zero alone is transparent, even with a
     * black palette; repeated and rejected rows must reset run state. */
    static uint8_t run_ptr[320*4],run_wad[320*2+2];
    static const unsigned run_widths[]={1,2,3,319,320};
    rows[0]=rows[2]=rows[3]=0;rows[1]=-1;
    for(unsigned pattern=0;pattern<4;pattern++)for(unsigned w=0;w<5;w++) {
        unsigned width=run_widths[w],pitch=width+4,drawn=0;
        palette[31]=(RendererPixel)(w&1?0:1);
        for(unsigned i=0;i<sizeof(actual)/sizeof(*actual);i++)actual[i]=expected[i]=0xa55a;
        for(unsigned x=0;x<320;x++) {
            unsigned offset=2+x*2;
            run_ptr[x*4]=run_ptr[x*4+1]=0;
            run_ptr[x*4+2]=(uint8_t)(offset>>8);run_ptr[x*4+3]=(uint8_t)offset;
            int opaque=pattern==1 || (pattern>=2 && (x&1)==pattern-2);
            run_wad[offset]=(uint8_t)(opaque?31:0);run_wad[offset+1]=0;
            if(x<width && opaque)for(unsigned y=0;y<4;y++)if(rows[y]>=0) {
                expected[(y+1)*pitch+x+2]=palette[31];drawn++;
            }
        }
        DosSpriteRows sprite={run_wad,run_ptr,sizeof(run_wad),sizeof(run_ptr),palette,rows,
                              0,0,65536u,320,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+pitch+2,width,4,pitch,&sprite);
        assert(got.visible==width*4 && got.drawn==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));checked++;
    }
    printf("PASS: %u sprite row rasters preserve packed modes, clipping origins, repeated rows, black/transparent texels, run boundaries/capacity, bounds and guards\n",checked);
}

static void check_sprite_batches(void)
{
    enum { PIXELS=320*240, LAYERS=12 };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16],palettes[LAYERS][32];
    static uint8_t wad[4098],native[4098],ptr[128],decoded[64];
    static int rows[LAYERS][240];
    struct Rect { unsigned x,y,w,h,u,du,down; } rect[LAYERS];
    DosSpriteBatch scratch;
    DosSpriteHistory history;
    uint32_t random=0x87451a9bu;
    for(unsigned test=0;test<32;test++) {
        for(unsigned i=0;i<sizeof(wad);i++) {
            random=random*1664525u+1013904223u;wad[i]=(uint8_t)(random>>24);
        }
        /* A fully opaque source also checks black ownership and complete
         * covered-row rejection, including reset between separate batches. */
        if(test%4==0)for(unsigned i=0;i<sizeof(wad);i+=2){wad[i]=4;wad[i+1]=33;}
        for(unsigned i=0;i<sizeof(wad);i+=2){native[i]=wad[i+1];native[i+1]=wad[i];}
        for(unsigned x=0;x<32;x++) {
            unsigned offset=2+x*128;
            ptr[x*4]=(uint8_t)((x+test)%3);ptr[x*4+1]=0;
            ptr[x*4+2]=(uint8_t)(offset>>8);ptr[x*4+3]=(uint8_t)offset;
        }
        for(unsigned layer=0;layer<LAYERS;layer++) {
            struct Rect *r=&rect[layer];
            random=random*1664525u+1013904223u;
            r->x=layer==LAYERS-1 && test%4==0?0:(random>>16)%320;
            r->w=1+(random>>8)%(320-r->x);
            r->y=(random>>24)%100;r->h=1+random%(240-r->y);
            if(layer==LAYERS-1 && test%4==0){r->w=320;r->y=0;r->h=240;}
            r->u=(layer%3)*65536u+333u;r->du=(32u*65536u)/r->w;
            r->down=layer%3;
            for(unsigned y=0;y<r->h;y++)rows[layer][y]=(int)(y*61/r->h)-(test%3==0?1:0);
            for(unsigned i=0;i<32;i++)palettes[layer][i]=(RendererPixel)(i==1?0:i==2?1:((i+layer*32)*1973u)&0x7fff);
        }
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=0xa55a;
        /* Independent original-order pixel sampler: later opaque pixels win. */
        for(unsigned layer=0;layer<LAYERS;layer++) {
            const struct Rect *r=&rect[layer];
            for(unsigned y=0;y<r->h;y++)for(unsigned x=0;x<r->w;x++) {
                int row=rows[layer][y];if(row<0)continue;
                unsigned column=(r->u+x*r->du)>>16;if(column>=32)column=31;
                const uint8_t *entry=ptr+column*4;
                unsigned offset=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
                unsigned address=offset+((unsigned)row+r->down)*2;
                if(address+2>sizeof(wad))continue;
                unsigned word=((unsigned)wad[address]<<8)|wad[address+1];
                unsigned texel=(word>>(entry[0]*5))&31;
                if(texel)expected[8+(r->y+y)*320+r->x+x]=palettes[layer][texel];
            }
        }
        dos_raster_sprite_batch_begin(actual+8,&scratch);
        if(test&1)dos_raster_sprite_batch_history_begin(&history);
        unsigned written=0;
        for(unsigned step=0;step<LAYERS;step++) {
            unsigned layer=LAYERS-1-step;const struct Rect *r=&rect[layer];
            RendererPixel *out=actual+8+r->y*320+r->x;
            if(layer%3!=0 || test%4==0) {
                DosSpriteRows sprite={native,ptr,sizeof(wad),sizeof(ptr),palettes[layer],rows[layer],0,r->u,r->du,32,r->down};
                written+=dos_raster_sprite_rows(out,r->w,r->h,320,&sprite).drawn;
            } else for(unsigned x=0;x<r->w;x++) {
                unsigned column=(r->u+x*r->du)>>16;if(column>=32)column=31;
                const uint8_t *entry=ptr+column*4;
                unsigned offset=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
                const uint8_t *source=wad+offset+r->down*2;
                unsigned limit=(sizeof(wad)-offset)/2-1-r->down,shift=entry[0]*5;
                if(x&1)written+=dos_raster_sprite_column(out+x,r->h,320,native+offset+r->down*2,palettes[layer],rows[layer],limit,shift);
                else {
                    for(unsigned y=0;y<64 && y<=limit;y++)decoded[y]=(uint8_t)((((unsigned)source[y*2]<<8)|source[y*2+1])>>shift)&31;
                    written+=dos_raster_sprite_texel_column(out+x,r->h,320,decoded,palettes[layer],rows[layer],limit<63?limit:63);
                }
            }
        }
        dos_raster_sprite_batch_end();
        assert(!memcmp(actual,expected,sizeof(actual)));
        unsigned changed=0;for(unsigned i=0;i<PIXELS;i++)changed+=expected[i+8]!=0xa55a;
        assert(written==changed);
        /* Ending a pass restores ordinary writes at previously covered pixels. */
        int zero=0;uint8_t one=1;RendererPixel palette[2]={0,0x1234};
        assert(dos_raster_sprite_texel_column(actual+8,1,320,&one,palette,&zero,0)==1);
        assert(actual[8]==0x1234);
    }
    puts("PASS: 32 reverse billboard batches match independent painter order, row/column cutouts, black, mask edges, complete coverage, reset and guards");
}

static void check_sprite_batch_repeated_rows(void)
{
    enum { HEIGHT=8, PIXELS=320*HEIGHT };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16];
    uint8_t wad[4]={0,0,31,0},ptr[128]={0},back_ptr[128]={0},texels[2]={0,1};
    RendererPixel front[2]={0,0x1357},middle[32]={0},back[32]={0};
    int rows[HEIGHT],constant[HEIGHT]={0};
    static const unsigned clips[3][2]={{0,320},{1,318},{33,254}};
    DosSpriteBatch scratch;
    for(unsigned x=0;x<32;x++)back_ptr[x*4+3]=2;
    for(unsigned stencil=0;stencil<3;stencil++)
    for(unsigned clip=0;clip<3;clip++)for(unsigned pattern=0;pattern<6;pattern++) {
        unsigned left=clips[clip][0],width=clips[clip][1],written=0;
        for(unsigned x=0;x<32;x++) {
            int opaque=stencil==0 || (stencil==1?x>=8 && x<24:
                                      (x>=2 && x<6) || (x>=24 && x<30));
            ptr[x*4+3]=opaque?2:0;
        }
        middle[31]=(RendererPixel)(pattern&1?0:1);back[31]=0x2468;
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=0xa55a;
        dos_raster_sprite_batch_begin(actual+8,&scratch);
        for(unsigned x=0;x<320;x++) {
            for(unsigned y=0;y<HEIGHT;y++) {
                int opaque;
                switch(pattern) {
                    case 0:opaque=x&1;break; /* Maximum 160 fresh copy ranges. */
                    case 1:opaque=(x+y)&1;break;
                    case 2:opaque=(x&1)^(x==31 && y>=4);break;
                    case 3:opaque=x!=(31+y);break;
                    case 4:opaque=(x<32 || x>=288) && (y&1);break;
                    default:opaque=y&1;break;
                }
                rows[y]=opaque;
                unsigned column=x>=left?(x-left)/8:0;
                if(column>31)column=31;
                int middle_opaque=x>=left && x<left+width && ptr[column*4+3]!=0;
                expected[8+y*320+x]=opaque?front[1]:middle_opaque?middle[31]:back[31];
            }
            written+=dos_raster_sprite_texel_column(actual+8+x,HEIGHT,320,
                                                      texels,front,rows,1);
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),middle,constant,
                              0,0,8192u,32,0};
        written+=dos_raster_sprite_rows(actual+8+left,width,HEIGHT,320,&sprite).drawn;
        sprite.palette=back;
        sprite.ptr=back_ptr;
        written+=dos_raster_sprite_rows(actual+8,320,HEIGHT,320,&sprite).drawn;
        for(unsigned y=0;y<HEIGHT;y++)assert(!scratch.open_words[y]);
        dos_raster_sprite_batch_end();
        assert(written==PIXELS && !memcmp(actual,expected,sizeof(actual)));
    }
    puts("PASS: 54 repeated billboard bands preserve 160 copy ranges, changing masks, transparent margins and gaps, boundary words, outside clips, closed-word counts, black and guards");
}

static void check_sprite_narrow_coverage(void)
{
    enum { HEIGHT=8, PIXELS=320*HEIGHT };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16];
    static const unsigned widths[]={1,2,31,32,33,63,64,65};
    uint8_t wad[4]={0,0,31,0},ptr[4]={0,0,0,2};
    RendererPixel palette[32]={0};int rows[HEIGHT]={0};
    DosSpriteBatch scratch,expected_mask;unsigned checked=0;
    for(unsigned bit=0;bit<32;bit++)for(unsigned w=0;w<8;w++)
    for(unsigned pattern=0;pattern<4;pattern++) {
        unsigned left=224+bit,width=widths[w],drawn=0;
        palette[31]=(RendererPixel)(w&1?0:0x2468);
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=0xa55a;
        dos_raster_sprite_batch_begin(actual+8,&scratch);
        /* Independent pixel coverage: fully closed rows alternate with
         * outside holes, while the rectangle is covered except for selected
         * first/last pixels in its first, last or interior rows. */
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=0;x<320;x++) {
            int inside=x>=left && x<left+width;
            int covered=inside || !(y&1) || (x&1);
            if((pattern==1 && y==0 && x==left) ||
               (pattern==2 && y==HEIGHT-1 && x==left+width-1) ||
               (pattern==3 && ((y==1 && x==left) ||
                               (y==HEIGHT-2 && x==left+width-1))))covered=0;
            if(covered)scratch.covered[y][x/32]|=1u<<(x%32);
            actual[8+y*320+x]=expected[8+y*320+x]=covered?0x1357:0xa55a;
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=scratch.covered[y][word]!=UINT32_MAX;
            scratch.open_words[y]=(uint8_t)open;
        }
        expected_mask=scratch;
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=left;x<left+width;x++) {
            uint32_t mask=1u<<(x%32);
            if(!(scratch.covered[y][x/32]&mask)) {
                expected[8+y*320+x]=palette[31];drawn++;
                expected_mask.covered[y][x/32]|=mask;
            }
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=expected_mask.covered[y][word]!=UINT32_MAX;
            expected_mask.open_words[y]=(uint8_t)open;
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,rows,
                              0,0,65536u,1,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+8+left,width,HEIGHT,320,&sprite);
        dos_raster_sprite_batch_end();
        assert(got.visible==(pattern?width*HEIGHT:0) && got.drawn==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        assert(!memcmp(&scratch,&expected_mask,sizeof(scratch)));checked++;
    }
    printf("PASS: %u narrow billboard coverage cases preserve every bit alignment, one/two/three-word clips, first/last/interior gaps, closed rows, black, counts and guards\n",checked);
}

static void check_sprite_history(void)
{
    enum { PIXELS=320*240, STEPS=12 };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16],palette[2][32];
    static uint8_t wad[2][16*32*2+2],ptr[2][16*4];
    int rows[240];
    DosSpriteBatch scratch,expected_mask;
    DosSpriteHistory history;
    DosSpriteCoverage reference[STEPS];
    for(unsigned source=0;source<2;source++)for(unsigned x=0;x<16;x++) {
        unsigned offset=2+x*64,ptr_offset=2+((x+source)%16)*64;
        ptr[source][x*4]=(uint8_t)(x%3);
        ptr[source][x*4+2]=(uint8_t)(ptr_offset>>8);
        ptr[source][x*4+3]=(uint8_t)ptr_offset;
        for(unsigned y=0;y<32;y++) {
            unsigned texel=(x+y+source)%3?0:1+(x+y)%31;
            sprite_word_write(wad[source]+offset+y*2,(uint16_t)(texel|(texel<<5)|(texel<<10)));
        }
    }
    /* Compare the optional cache with the same uncached mask kernel, including
     * every sampling input, palette-only changes, reused row storage and reset.
     * Transparent holes keep repeated rectangles out of the whole-box reject. */
    for(unsigned variant=0;variant<25;variant++)for(unsigned cached=0;cached<2;cached++) {
        RendererPixel *surface=cached?actual:expected;
        for(unsigned i=0;i<PIXELS+16;i++)surface[i]=0xa55a;
        dos_raster_sprite_batch_begin(surface+8,&scratch);
        if(cached)dos_raster_sprite_batch_history_begin(&history);
        for(unsigned step=0;step<STEPS;step++) {
            for(unsigned y=0;y<240;y++)rows[y]=(int)(y/4)%24;
            for(unsigned i=0;i<32;i++) {
                palette[0][i]=(RendererPixel)(i==1?0:i*1973u);
                palette[1][i]=(RendererPixel)(i*917u);
            }
            unsigned width=310,height=239,pitch=320,offset=8;
            if(variant>=23)height=variant==23?1:240;
            unsigned changed_row=(variant==18 || variant==22 || variant==23)?0:
                                  (variant==19 || variant==21)?height/2:height-1;
            if(variant>=18) {
                /* Begin with a rejected row, then expose its valid low-byte
                 * sample without changing the reused row-array address.
                 * Differences in any byte must invalidate the footprint. */
                uint32_t high=variant==21?65536u:variant==22?0x80000000u:256u;
                rows[changed_row]=(int)((uint32_t)rows[changed_row]|high);
            }
            DosSpriteRows sprite={wad[0],ptr[0],sizeof(wad[0]),sizeof(ptr[0]),
                                  palette[0],rows,0,0,65536u/20,16,0};
            if(step%4==1 || step%4==2)switch(variant) {
                case 0:sprite.palette=palette[1];break;
                case 1:for(unsigned y=0;y<240;y++)rows[y]++;break;
                case 2:sprite.wad=wad[1];break;
                case 3:sprite.ptr=ptr[1];break;
                case 4:sprite.wad_bytes-=128;break;
                case 5:sprite.ptr_bytes=16;break;
                case 6:sprite.ptr_offset=4;break;
                case 7:sprite.u=65536;break;
                case 8:sprite.du=65536u/10;break;
                case 9:sprite.columns=8;break;
                case 10:sprite.down_strip=2;break;
                case 11:width--;break;
                case 12:height++;break;
                case 13:offset++;break;
                case 14:offset+=320;break;
                case 15:width=32;height=1;break;
                case 16:for(unsigned i=1;i<32;i++)palette[0][i]=0;break;
                case 17:for(unsigned y=0;y<240;y++)rows[y]=(y&1)?-1:INT_MAX;break;
                case 18:case 19:case 20:case 21:case 22:case 23:case 24:
                    rows[changed_row]=(int)((changed_row/4)%24);break;
            }
            /* More than four distinct footprints exercise eviction, then
             * revisiting the original verifies that missing entries are safe. */
            if(step>=4 && step<10)offset+=step;
            DosSpriteCoverage got=dos_raster_sprite_rows(surface+offset,width,height,pitch,&sprite);
            if(!cached)reference[step]=got;
            else assert(got.visible==reference[step].visible && got.drawn==reference[step].drawn);
        }
        dos_raster_sprite_batch_end();
        if(!cached)expected_mask=scratch;
        else {
            assert(!memcmp(actual,expected,sizeof(actual)));
            assert(!memcmp(&scratch,&expected_mask,sizeof(scratch)));
        }
    }
    puts("PASS: 300 cached billboard draws retain uncached pixels, coverage, sampling keys, palette changes, row storage reuse, high-byte changes, first/last rows, one/full-height clips, eviction, reset and guards");
}

static void check_word_offset_bounds(void)
{
    static const uint32_t values[]={0,1,2,3,4,15,31,255,256,65535,65536,UINT32_MAX-2,UINT32_MAX-1,UINT32_MAX};
    unsigned checked=0;
    for(unsigned a=0;a<sizeof(values)/sizeof(*values);a++)
        for(unsigned b=0;b<sizeof(values)/sizeof(*values);b++)
            for(unsigned c=0;c<sizeof(values)/sizeof(*values);c++) {
                uint32_t base=values[a],offset=values[b],bytes=values[c];
                int expected=(uint64_t)base+offset+2<=bytes;
                assert(renderer_word_offsets_fit_u32(base,offset,bytes)==expected);
                checked++;
            }
    uint32_t seed=0xa5a55a5a;
    for(unsigned i=0;i<32768;i++) {
        seed=seed*1664525u+1013904223u;uint32_t base=seed;
        seed=seed*1664525u+1013904223u;uint32_t maximum=seed;
        seed=seed*1664525u+1013904223u;uint32_t bytes=seed;
        assert(renderer_word_offsets_fit_u32(base,maximum,bytes)==((uint64_t)base+maximum+2<=bytes));
        checked++;
    }
    printf("PASS: %u word-span bounds retain short inputs and unsigned overflow guards\n",checked);
}

static void check_sky_step_cache(void)
{
    static const int32_t widths[]={1,2,3,7,127,128,255,256,319,320,321,640,1024,16384,INT32_MAX};
    static const int32_t columns[]={1,2,3,31,64,127,128,511,512,1024,32767,65536,INT32_MAX};
    RendererSkyStepCache cache={0};unsigned checked=0;
    for(unsigned pass=0;pass<4;pass++)
        for(unsigned w=0;w<sizeof(widths)/sizeof(*widths);w++)
            for(unsigned c=0;c<sizeof(columns)/sizeof(*columns);c++) {
                int32_t width=widths[pass&1?(sizeof(widths)/sizeof(*widths)-1-w):w];
                int32_t count=columns[pass&2?(sizeof(columns)/sizeof(*columns)-1-c):c];
                int32_t expected=(int32_t)reference_divide((int64_t)count*65536,width);
                assert(renderer_sky_step_cached(&cache,count,width)==expected);
                assert(renderer_sky_step_cached(&cache,count,width)==expected);
                int32_t changed_width=width==1?2:width-1;
                int32_t changed=(int32_t)reference_divide((int64_t)count*65536,changed_width);
                assert(renderer_sky_step_cached(&cache,count,changed_width)==changed);
                assert(renderer_sky_step_cached(&cache,count,width)==expected);
                checked+=4;
            }
    printf("PASS: %u cached sky quotients preserve changed widths/columns, repeats and wide results\n",checked);
}

static void check_billboard_projection(void)
{
    static const int32_t sizes[]={1,2,3,31,127,128,255,1024,32767,INT32_MAX};
    static const int64_t scales[]={65536,65537,98304,131072,33554432,
        INT64_C(4294967296),INT64_C(4294967297),INT64_C(1)<<47,
        INT64_MAX-65535,INT64_MAX};
    static const int32_t depths[]={1,2,3,255,32767,32768,65535,16777216,INT32_MAX};
    unsigned checked=0;
    for(unsigned s=0;s<sizeof(sizes)/sizeof(*sizes);s++)
        for(unsigned k=0;k<sizeof(scales)/sizeof(*scales);k++) {
            if(scales[k]>INT64_MAX/sizes[s])continue;
            int64_t n=(int64_t)sizes[s]*scales[k];
            for(unsigned d=0;d<sizeof(depths)/sizeof(*depths);d++) {
                int64_t denominator=(int64_t)depths[d]*65536;
                assert(renderer_project_size_fp16(sizes[s],scales[k],depths[d])==reference_divide(n,denominator));
                checked++;
            }
        }
    uint32_t seed=0x9e3779b9;
    for(unsigned i=0;i<65536;i++) {
        seed=seed*1664525u+1013904223u;int32_t size=1+(seed&65535);
        seed=seed*1664525u+1013904223u;int64_t scale=(int64_t)(1+(seed&32767))*65536;
        if(i&1)scale+=i&65535;
        seed=seed*1664525u+1013904223u;int32_t depth=(int32_t)(seed&INT32_MAX);
        if(!depth)depth=1;
        int64_t expected=reference_divide((int64_t)size*scale,(int64_t)depth*65536);
        assert(renderer_project_size_fp16(size,scale,depth)==expected);
        checked++;
    }
    printf("PASS: %u billboard projections preserve whole/fractional scales, wide depths and quotients\n",checked);
}

static void check_floor_step_rounding(void)
{
    static const int32_t values[]={INT32_MIN,INT32_MIN+1,-65537,-65536,-257,-256,-255,
        -129,-128,-127,-2,-1,0,1,2,127,128,129,255,256,257,65535,65536,INT32_MAX-1,INT32_MAX};
    static const int32_t widths[]={1,3,7,160,320,511,640,1024,1048575};
    unsigned checked=0;
    for(unsigned shift=0;shift<32;shift++)for(unsigned i=0;i<sizeof(values)/sizeof(*values);i++) {
        int64_t d=INT64_C(1)<<shift,n=values[i],half=d/2;
        int32_t expected=(int32_t)reference_divide(n>=0?n+half:n-half,d);
        assert(renderer_round_shift_away_i32(values[i],shift)==expected);
        checked++;
    }
    uint32_t seed=0x6d2b79f5;
    for(unsigned i=0;i<131072;i++) {
        seed=seed*1664525u+1013904223u;
        int32_t value=(int32_t)seed;
        unsigned shift=1+(i%31);
        int64_t d=INT64_C(1)<<shift,n=value,half=d/2;
        assert(renderer_round_shift_away_i32(value,shift)==reference_divide(n>=0?n+half:n-half,d));
        checked++;
    }
    /* Exhaust every discarded byte on both sides of selected quotient and
     * overflow boundaries against the original width-scaled wide expression. */
    static const int32_t quotients[]={-8388608,-8388607,-65536,-129,-128,-127,-2,-1,0,1,2,127,128,129,65535,8388606,8388607};
    for(unsigned w=0;w<sizeof(widths)/sizeof(*widths);w++)
        for(unsigned q=0;q<sizeof(quotients)/sizeof(*quotients);q++)for(unsigned rem=0;rem<256;rem++) {
            int64_t value=(int64_t)quotients[q]*256+rem;
            if(value<INT32_MIN || value>INT32_MAX)continue;
            int64_t denominator=(int64_t)widths[w]*512;
            int64_t numerator=value*(widths[w]*2);
            int32_t expected=(int32_t)reference_divide(numerator>=0?numerator+denominator/2:numerator-denominator/2,denominator);
            assert(renderer_round_shift_away_i32((int32_t)value,8)==expected);
            checked++;
        }
    printf("PASS: %u floor step roundings preserve signed half ties, width cancellation and int32 extremes\n",checked);
}

static void check_division(void)
{
    for (int64_t n = -65537; n <= 65537; n += 127) {
        static const int offsets[] = {-8388609, -8388608, -1, 0, 1, 8388607, 8388608};
        for (unsigned i = 0; i < sizeof(offsets) / sizeof(*offsets); i++) {
            int64_t value = n * 16777216 + offsets[i];
            int32_t rounded = (int32_t)((value < 0 ? value - 8388608 : value + 8388608) / 16777216);
            assert(renderer_project_round_24(value) == rounded);
        }
    }
    static const int64_t numerators[] = {INT64_MIN, INT64_MIN + 1, -4294967297LL,
        INT32_MIN, -1, 0, 1, INT32_MAX, 4294967297LL, INT64_MAX};
    static const int32_t divisors[] = {INT32_MIN, -16777216, -65536, -255, -3, -1,
        1, 3, 255, 65536, 16777216, INT32_MAX};
    unsigned checked = 0;
    for (unsigned n = 0; n < sizeof(numerators) / sizeof(*numerators); n++) {
        for (unsigned d = 0; d < sizeof(divisors) / sizeof(*divisors); d++) {
            if (numerators[n] == INT64_MIN && divisors[d] == -1) continue;
            assert(renderer_divide_i64_i32(numerators[n], divisors[d]) == reference_divide(numerators[n], divisors[d]));
            checked++;
        }
    }
    for (unsigned i = 0; i < 4096; i++) {
        int64_t n = (int64_t)(((uint64_t)next_value() << 32) | next_value());
        int32_t d = (int32_t)(next_value() | 1u);
        assert(renderer_divide_i64_i32(n, d) == reference_divide(n, d));
        checked++;
    }
    printf("PASS: %u signed 64/32 divisions, including wide quotients\n", checked);
}

static void check_projection_rounding(void)
{
    uint32_t saved_random = random_state;
    random_state = 0x9826c31b;
    static const int32_t high_words[] = {INT32_MIN, -1, 0, 1, INT32_MAX};
    static const uint32_t low_words[] = {0, 1, 8388607, 8388608, 8388609,
        16777215, 0xff7fffff, 0xff800000, UINT32_MAX};
    unsigned checked = 0;
    for (unsigned i = 0; i < 8192 + 45; i++) {
        uint32_t high, low;
        if (i < 45) {
            high = (uint32_t)high_words[i / 9]; low = low_words[i % 9];
        } else {
            high = next_value(); low = next_value();
        }
        int64_t numerator = (int64_t)(((uint64_t)high << 32) | low);
        if (numerator > INT64_MAX - 8388608) continue;
        int64_t expected = reference_divide(numerator, 16777216);
        int64_t remainder = numerator - expected * 16777216;
        if (remainder >= 8388608) expected++;
        if (remainder <= -8388608) expected--;
        assert(renderer_project_round_24(numerator) == (int32_t)expected);
        checked++;
    }
    random_state = saved_random;
    printf("PASS: %u projection roundings preserve wide products, carries and signed half ties\n", checked);
}

static unsigned check_wall_interpolation_case(int64_t delta, int32_t position, int32_t span)
{
    /* Exercise every defined original product without overflowing the oracle. */
    uint64_t magnitude = delta < 0 ? UINT64_C(0) - (uint64_t)delta : (uint64_t)delta;
    uint64_t distance = position < 0 ? UINT64_C(0) - (uint64_t)(int64_t)position : (uint64_t)position;
    uint64_t bound = ((delta < 0) != (position < 0)) ? UINT64_C(1) << 63 : INT64_MAX;
    if (distance && magnitude > bound / distance) return 0;
    int64_t expected = reference_divide(delta * position, span);
    assert(renderer_wall_interpolate_delta(delta, position, span) == expected);
    return 1;
}

static void check_wall_interpolation(void)
{
    uint32_t saved_random = random_state;
    random_state = 0x73952b17;
    static const int64_t deltas[] = {INT64_MIN, INT64_MIN + 1, -4294967297LL,
        INT32_MIN, -65536, -1, 0, 1, 65536, INT32_MAX, 4294967297LL, INT64_MAX};
    static const int32_t spans[] = {1, 2, 3, 7, 127, 128, 255, 320, 65535, 65536, INT32_MAX};
    unsigned checked = 0;
    for (unsigned d = 0; d < sizeof(deltas) / sizeof(*deltas); d++) {
        for (unsigned s = 0; s < sizeof(spans) / sizeof(*spans); s++) {
            int32_t span = spans[s];
            int32_t positions[] = {0, 1, span / 2, span - 1, span, -1, INT32_MIN, INT32_MAX};
            for (unsigned p = 0; p < sizeof(positions) / sizeof(*positions); p++)
                checked += check_wall_interpolation_case(deltas[d], positions[p], span);
        }
    }
    for (unsigned i = 0; i < 4096; i++) {
        int32_t span = (next_value() & 65535) + 1;
        int32_t position = (int32_t)(next_value() % ((uint32_t)span + 1));
        int64_t delta = (int64_t)(int32_t)next_value() * 65536 + (next_value() & 65535);
        checked += check_wall_interpolation_case(delta, position, span);
    }
    random_state = saved_random;
    printf("PASS: %u wall endpoint interpolations preserve constant, clipped and full-span divisions\n", checked);
}

static void check_wall_texture_walk(void)
{
    uint32_t saved_random = random_state;
    random_state = 0x61438327;
    static const int32_t divisors[] = {1, 2, 3, 17, 127, 128, 255, 16384, 65535, 65536, 131071, 16777216};
    unsigned checked = 0;
    for (unsigned test = 0; test < 128; test++) {
        for (unsigned d = 0; d < sizeof(divisors) / sizeof(*divisors); d++) {
            int32_t divisor = divisors[d];
            int64_t phase = (int64_t)((int32_t)next_value() / 256) * 65536 + (next_value() & 65535);
            int64_t step = (int64_t)((int32_t)next_value() % 4096) * 65536 + (next_value() & 65535);
            RendererWallTextureWalk walk;
            assert(renderer_wall_texture_walk_init(&walk, phase, step, divisor));
            for (unsigned i = 0; i < 320; i++, phase += step) {
                /* Original expanded tex/z expression, not the reduced walk. */
                int32_t expected = (int32_t)(reference_divide((phase >> 16) * INT64_C(16777216), divisor) >> 24);
                assert((int32_t)walk.quotient == expected);
                assert(renderer_wall_tex_column(phase >> 16, divisor) == expected);
                renderer_wall_texture_walk_advance(&walk);
                checked++;
            }
        }
    }
    static const int32_t starts[] = {INT32_MIN + 4, -1, 0, INT32_MAX - 4};
    for (unsigned q = 0; q < sizeof(starts) / sizeof(*starts); q++) {
        for (unsigned d = 0; d < 9; d++) {
            for (int sign = -1; sign <= 1; sign += 2) {
                int32_t divisor = divisors[d];
                int64_t phase = (int64_t)starts[q] * divisor * 65536 + 65535;
                int64_t step = sign * ((int64_t)divisor * 65536 - 1);
                RendererWallTextureWalk walk;
                assert(renderer_wall_texture_walk_init(&walk, phase, step, divisor));
                for (unsigned i = 0; i < 3; i++, phase += step) {
                    int64_t value = phase >> 16;
                    int64_t expected = reference_divide(value, divisor) - (value % divisor < 0);
                    assert((int32_t)walk.quotient == expected);
                    renderer_wall_texture_walk_advance(&walk);
                    checked++;
                }
            }
        }
    }
    RendererWallTextureWalk walk;
    assert(!renderer_wall_texture_walk_init(&walk, 0, ((int64_t)INT32_MAX + 1) * 65536, 1));
    assert(!renderer_wall_texture_walk_init(&walk, 0, ((int64_t)INT32_MIN - 1) * 65536, 1));
    assert(!renderer_wall_texture_walk_init(&walk, 0, 0, 0));
    random_state = saved_random;
    printf("PASS: %u wall texture walk samples retain original division, signed carries and wide-step guards\n", checked);
}

static void check_palette(void)
{
    uint8_t palette[8192];
    for (unsigned c = 0; c < 4096; c++) { palette[c * 2] = c >> 8; palette[c * 2 + 1] = c; }
    renderer_pixel_prepare_palette(palette, sizeof(palette));
    for (unsigned c = 0; c < 4096; c++)
        assert(renderer_pixel_to_c12(renderer_pixel_from_palette(palette + c * 2)) == c);
    uint8_t packaged[8192];
    FILE *file=fopen("COLORS.BIN","rb");
    assert(file);
    size_t bytes=fread(packaged,1,sizeof(packaged),file);
    int closed=fclose(file);
    assert(bytes==sizeof(packaged) && !closed && !memcmp(packaged,palette,sizeof(palette)));
    puts("PASS: all 4096 prepared palette colors retain their source values");
}

static void check_water_projection(void)
{
    static const int32_t depths[] = {1, 2, 3, 127, 255, 256, 257, 65535, 65536, INT32_MAX};
    static const int32_t heights[] = {INT32_MIN, -65537, -257, -1, 0, 1, 257, 65537, INT32_MAX};
    unsigned checked = 0;
    RendererRoundCache cache = {0};
    for (unsigned y = 0; y < sizeof(heights) / sizeof(*heights); y++) {
        for (unsigned z = 0; z < sizeof(depths) / sizeof(*depths); z++) {
            for (int scale = 1; scale <= 2048; scale *= 2) {
                int64_t num = (int64_t)heights[y] * scale * 8;
                int64_t expanded = num * 256, den = (int64_t)depths[z] * 256;
                int64_t expected = reference_divide(expanded >= 0 ? expanded + den / 2 : expanded - den / 2, den);
                assert(renderer_divide_round_i64_i32(num, depths[z]) == expected);
                assert(renderer_divide_round_cached(&cache, num, depths[z]) == (int32_t)expected);
                assert(renderer_divide_round_cached(&cache, num, depths[z]) == (int32_t)expected);
                int64_t changed = num + (int64_t)depths[z] * 3;
                int64_t changed_expected = reference_divide(changed >= 0 ? changed + depths[z] / 2 : changed - depths[z] / 2, depths[z]);
                assert(renderer_divide_round_cached(&cache, changed, depths[z]) == (int32_t)changed_expected);
                assert(renderer_divide_round_cached(&cache, num, depths[z]) == (int32_t)expected);
                if (depths[z] < INT32_MAX) {
                    int32_t new_depth = depths[z] + 1;
                    int64_t new_expected = reference_divide(num >= 0 ? num + new_depth / 2 : num - new_depth / 2, new_depth);
                    assert(renderer_divide_round_cached(&cache, num, new_depth) == (int32_t)new_expected);
                }
                checked++;
            }
        }
    }
    for (int sine = -65536; sine <= 65536; sine += 127) {
        for (int distance = 16; distance <= 32000; distance += 997) {
            int32_t expected = reference_divide((int64_t)sine * 256, (int64_t)(distance + 300) * 64);
            assert(renderer_water_vertical_refraction(sine, distance) == expected);
            checked++;
        }
    }
    printf("PASS: %u water projection and refraction cases match expanded fixed-point expressions\n", checked);
}

static void check_floors(void)
{
    uint8_t *texture = malloc(65536);
    assert(texture);
    for (unsigned i = 0; i < 65536; i++) texture[i] = next_value() >> 24;
    RendererPixel palettes[15][256], actual[322], expected[322];
    for (unsigned l = 0; l < 15; l++) {
        for (unsigned t = 0; t < 256; t++) palettes[l][t] = renderer_pixel_from_c12(l * 256 + t);
    }
    for (unsigned c = 0; c < 4096; c++) {
        unsigned count = c % 321;
        uint32_t u = next_value(), v = next_value(), du = next_value(), dv = next_value();
        if (c % 3 == 0) du = 0;
        if (c % 3 == 1) dv = 0;
        /* Dense sub-texel steps exercise exact positive carries and negative
         * borrows, zero steps, both axes and all unrolled tails. */
        static const uint32_t small_steps[]={0,1,8192,16384,32767,32768,32769,65535,
            65536,0u-1,0u-8192,0u-16384,0u-32767,0u-32768,0u-32769,0u-65535,
            123,4095,4096,4097,8191,8193,0u-123,0u-4095,
            0u-4096,0u-4097,0u-8191,0u-8193,17,0u-17,7777,0u-7777};
        if(c%6==0)dv=small_steps[(c/6)%32];
        if(c%6==1)du=small_steps[(c/6)%32];
        int32_t left = c % 15, right = (c / 15) % 15;
        int32_t level = left * 65536;
        int32_t dlevel = count > 1 ? (right - left) * 65536 / (int32_t)(count - 1) : 0;
        memset(actual, 0x91, sizeof(actual)); memcpy(expected, actual, sizeof(actual));
        uint32_t a = u, b = v;
        int32_t shade = level;
        for (unsigned x = 0; x < count; x++) {
            /* AB3DI d5.w = (V&63)*256 + (U&63), then sample d5.w*4. */
            unsigned offset = (((b >> 16) & 63) * 256 + ((a >> 16) & 63)) * 4;
            expected[x + 1] = palettes[shade >> 16][texture[offset]];
            a += du; b += dv; shade += dlevel;
        }
        dos_raster_floor_gouraud(actual + 1, count, texture, palettes, u, v, du, dv, level, dlevel);
        assert(!memcmp(actual, expected, sizeof(actual)));
        for (unsigned x = 0; x < count; x++) {
            unsigned offset = (((v >> 16) & 63) * 256 + ((u >> 16) & 63)) * 4;
            expected[x + 1] = palettes[left][texture[offset]];
            u += du; v += dv;
        }
        /* Reconstruct the original accumulators after the reference walk. */
        u -= du * count; v -= dv * count;
        dos_raster_floor(actual + 1, count, texture, palettes[left], u, v, du, dv);
        assert(!memcmp(actual, expected, sizeof(actual)));
    }
    free(texture);
    puts("PASS: 8192 floor spans retain UV wrapping, shade gradients and bounds");
}

static void check_sprite_palettes(void)
{
    unsigned checked=0;
    for(unsigned p=0;p<MAX_SPRITE_TYPES;p++) {
        assert(sprite_pal_embedded_size[p]==sprite_pal_source_size[p]);
        assert((sprite_pal_embedded[p]!=NULL)==(sprite_pal_source[p]!=NULL));
        for(size_t i=0;i+1<sprite_pal_source_size[p];i+=2) {
            unsigned source=(sprite_pal_source[p][i]<<8)|sprite_pal_source[p][i+1];
            uint16_t pixel=renderer_pixel_from_palette(sprite_pal_embedded[p]+i);
            assert(pixel==renderer_pixel_from_c12(source));
            assert(renderer_pixel_to_c12(pixel)==(source&4095));
            checked++;
        }
    }
    printf("PASS: %u embedded sprite shade colors match the original palettes\n",checked);
}

static void check_solid_floor_span(const uint8_t *texture,const RendererPixel (*palettes)[256],
                                   unsigned shade,unsigned count,uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    RendererPixel actual[322],expected[322];
    memset(actual,0x91,sizeof(actual));memcpy(expected,actual,sizeof(actual));
    uint32_t eu=u,ev=v;
    for(unsigned x=0;x<count;x++,eu+=du,ev+=dv)
        expected[x+1]=palettes[shade][texture[((eu>>14)&0xfc)|((ev>>6)&0xfc00)]];
    dos_raster_floor_shaded(actual+1,count,texture,palettes,shade,u,v,du,dv);
    assert(!memcmp(actual,expected,sizeof(actual)));
}

static void check_solid_floor_rows(void)
{
    uint32_t saved_random=random_state;random_state=0x16375829;
    static uint8_t texture[65536],other[65536];
    static RendererPixel palettes[3][15][256];
    for(unsigned y=0;y<256;y++)for(unsigned x=0;x<256;x++) {
        texture[y*256+x]=(y%8==0 || x%7==0)?0:(uint8_t)((x*37+y*13)&3);
        other[y*256+x]=(uint8_t)(next_value()>>24);
    }
    for(unsigned bank=0;bank<3;bank++)for(unsigned shade=0;shade<15;shade++)
        for(unsigned c=0;c<256;c++)palettes[bank][shade][c]=(RendererPixel)(c*37+shade*173+bank*977);
    for(unsigned c=0;c<256;c++)palettes[1][14][c]=0x7619;
    for(unsigned i=0;i<3;i++)dos_raster_prepare_floor(i,i==1?other:texture,(const uint8_t *)palettes[i]);
    static const unsigned counts[]={0,1,7,8,15,16,17,31,64,127,239,320};
    unsigned checked=0;
    for(unsigned bank=0;bank<3;bank++)for(unsigned tile=0;tile<16;tile++)
    for(unsigned axis=0;axis<2;axis++)for(int sign=-1;sign<=1;sign+=2)
    for(unsigned mode=0;mode<4;mode++)for(unsigned n=0;n<sizeof(counts)/sizeof(*counts);n++) {
        unsigned offset=(tile/4)*256+tile%4;
        uint32_t u=next_value(),v=next_value(),step=(uint32_t)(sign*(mode?98304:16384));
        if(!mode) {
            unsigned column=0;while((column*4+tile%4)%7)column++;
            u=(column<<16)|65535;v=(2u<<16)|65535;
        }
        uint32_t du=axis?0:step,dv=axis?step:0;
        if(mode==2)du=dv=0;
        if(mode==3){du=step;dv=0u-step;}
        unsigned shade=mode==1?13:14;
        check_solid_floor_span((bank==1?other:texture)+offset,palettes[bank],shade,counts[n],u,v,du,dv);
        checked++;
    }
    /* Rebuild at the same addresses after a palette/source change, and clear
     * before replacing data so no stale uniform classification can survive. */
    for(unsigned c=0;c<256;c++)palettes[0][14][c]=0x5678;
    dos_raster_prepare_floor(0,texture,(const uint8_t *)palettes[0]);
    check_solid_floor_span(texture,palettes[0],14,320,0,0,98304,65535);checked++;
    palettes[0][14][0]=0x1234;palettes[0][14][1]=0x7bef;
    dos_raster_prepare_floor(0,texture,(const uint8_t *)palettes[0]);
    texture[0]=1;
    dos_raster_prepare_floor(0,NULL,NULL);
    check_solid_floor_span(texture,palettes[0],14,320,0,0,65536,0);checked++;
    dos_raster_prepare_floor(0,texture,(const uint8_t *)palettes[0]);
    check_solid_floor_span(texture,palettes[0],14,320,0,0,65536,0);checked++;
    for(unsigned i=0;i<3;i++)dos_raster_prepare_floor(i,NULL,NULL);
    random_state=saved_random;
    printf("PASS: %u classified floor spans preserve both axes, tile channels, shade, UV wraps, bounds and reloads\n",checked);
}

static void check_edges(void)
{
    int16_t actual[4][320], expected[4][320];
    for (unsigned count = 0; count <= 311; count++) {
        for (unsigned offset = 0; offset < 4; offset++) {
            memset(actual, 0x91, sizeof(actual)); memcpy(expected, actual, sizeof(actual));
            for (unsigned i = offset; i < offset + count; i++) {
                expected[0][i] = 320; expected[1][i] = -1;
            }
            dos_raster_floor_edges_init(actual[0] + offset, actual[1] + offset,
                                         count, 320);
            assert(!memcmp(actual, expected, sizeof(actual)));
        }
    }
    for (unsigned c = 0; c < 4096; c++) {
        unsigned count = c % 321;
        int64_t x = (int32_t)(next_value() & 0x3fffffff) - 0x20000000;
        int64_t dx = (int32_t)(next_value() % 200001) - 100000;
        if (c % 8 == 0) { x = INT32_MAX; dx = 0; }
        if (c % 8 == 1) { x = INT32_MIN; dx = 0; }
        int64_t bright = (int32_t)next_value(), db = (int32_t)next_value();
        for (unsigned i = 0; i < 320; i++) {
            actual[0][i] = 320; actual[1][i] = -1;
            actual[2][i] = actual[3][i] = -99;
        }
        memcpy(expected, actual, sizeof(actual));
        int64_t a = x, b = bright;
        for (unsigned i = 0; i < count; i++) {
            int64_t left = a >> 16, right = a <= 0 ? left : (a + 65535) >> 16;
            if (left < expected[0][i]) { expected[0][i] = left; expected[2][i] = b >> 16; }
            if (right > expected[1][i]) { expected[1][i] = right > INT16_MAX ? INT16_MAX : right; expected[3][i] = b >> 16; }
            a += dx; b += db;
        }
        dos_raster_floor_edge(actual[0], actual[1], actual[2], actual[3], count, x, dx, bright, db);
        assert(!memcmp(actual, expected, sizeof(actual)));
    }
    puts("PASS: 4096 floor edges match the wide fixed-point walk");
}

static void check_walls(void)
{
    FILE *file=fopen("WALL.BIN","rb");
    assert(file && !fseek(file,0,SEEK_END));
    long bytes=ftell(file);
    assert(bytes>0 && !fseek(file,0,SEEK_SET));
    uint8_t *data = malloc((size_t)bytes);
    assert(data);
    assert(fread(data,1,(size_t)bytes,file)==(size_t)bytes && !fclose(file));
    size_t source_bytes;
    uint8_t *native_palette=dos_load_wall(data,(size_t)bytes,&source_bytes);
    assert(source_bytes==2048+65536*2);
    for(unsigned i=0;i<1024;i++)
        assert(renderer_pixel_from_palette(native_palette+i*2)==renderer_pixel_from_c12((i*37)&4095));
    const uint8_t *texels=native_palette+DOS_WALL_HEADER_BYTES;
    for (unsigned w = 0; w < 65536; w++)
        for (unsigned c = 0; c < 3; c++) assert(texels[c * 65536 + w] == ((w >> (c * 5)) & 31));
    RendererPixel palettes[4][32];
    static RendererPixel actual[320 * 240], expected[320 * 240];
    for (unsigned l = 0; l < 4; l++)
        for (unsigned t = 0; t < 32; t++) palettes[l][t] = renderer_pixel_from_c12((t * 97 + l * 73) & 4095);
    const RendererPixel *pal = palettes[0];
    for (unsigned c = 0; c < 96; c++) {
        unsigned rows = 1u << (c % 5 + 4), mask = rows - 1;
        uint32_t y = next_value(), dy = c % 2 ? next_value() : 8192;
        if(c%8==0)dy=0;
        if(c%8==2)dy=32768;
        if(c%8==4)dy=32769;
        if(c%8==6)dy=65535;
        if(c%9==0)mask=0;
        if(c%11==0)mask &= 0x55;
        unsigned width = 1 + c % 33;
        DosWallColumn columns[33];
        for (unsigned col = 0; col < 33; col++) {
            columns[col].texels = dos_wall_column(texels,col,rows);
            columns[col].palette = palettes[c % 3 ? 0 : col % 4];
        }
        memset(actual, 0x91, sizeof(actual)); memcpy(expected, actual, sizeof(actual));
        uint32_t a = y;
        for (unsigned row = 0; row < 150; row++) {
            unsigned r = (a >> 16) & mask;
            for (unsigned col = 0; col < width; col++) {
                unsigned word = ((col / 3) * rows + r) & 65535;
                expected[row * 40 + col] = columns[col].palette[(word >> ((col % 3) * 5)) & 31];
            }
            a += dy;
        }
        dos_raster_wall_rows(actual, width, 150, 40, columns, NULL, y, dy, mask);
        assert(!memcmp(actual, expected, sizeof(actual)));
        if(c%3) {
            /* The original packed-word oracle above supplies the expected
             * pixels. An explicit palette must avoid every per-column field. */
            for(unsigned col=0;col<33;col++)columns[col].palette=NULL;
            memset(actual,0x91,sizeof(actual));
            dos_raster_wall_rows(actual,width,150,40,columns,palettes[0],y,dy,mask);
            assert(!memcmp(actual,expected,sizeof(actual)));
        }
        memset(actual, 0x91, sizeof(actual)); memset(expected, 0x91, sizeof(expected));
        a = y;
        unsigned count = c * 23 % 160, stride = c % 3 == 2 ? 40 : 320, base_word=c*17;
        for (unsigned row = 0; row < count; row++) {
            unsigned word = ((a >> 16) & mask)+base_word;
            expected[row * stride] = pal[word & 31];
            a += dy;
        }
        dos_raster_wall(actual, count, stride, texels+base_word, pal, y, dy, mask);
        assert(!memcmp(actual, expected, sizeof(actual)));
        unsigned edge_count = !c ? 0 : c % 13 ? c * 23 % 240 : 240;
        for (unsigned copies = 2; copies <= 3; copies++) {
            memset(actual, 0x91, sizeof(actual)); memset(expected, 0x91, sizeof(expected));
            a = y;
            for (unsigned row = 0; row < edge_count; row++) {
                unsigned word = ((a >> 16) & mask) + base_word;
                for (unsigned x = 0; x < copies; x++) expected[row * stride + 3 + x] = pal[word & 31];
                a += dy;
            }
            dos_raster_wall_edges(actual + 3, edge_count, stride, texels + base_word, pal, y, dy, mask, copies);
            assert(!memcmp(actual, expected, sizeof(actual)));
        }
    }
    free(data);
    puts("PASS: all packed wall words, 192 row/column rasters and 384 edge rasters preserve samples and bounds");
}

static void put_word(uint8_t *p, int value) { p[0] = (unsigned)value >> 8; p[1] = value; }
static int get_word(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }

static void check_objects(void)
{
    static LevelState level;
    static uint8_t objects[256 * OBJECT_SIZE], point_data[256 * 8];
    static uint8_t nasty[NASTY_SHOT_SLOT_COUNT * OBJECT_SIZE], player[PLAYER_SHOT_SLOT_COUNT * OBJECT_SIZE];
    static ObjRotatedPoint points[256];
    const int near_z = ROT_Z_FROM_INT(50);
    level.object_data = objects; level.object_points = point_data; level.num_object_points = 256;
    level.nasty_shot_data = nasty; level.player_shot_data = player; level.draw_group_count = 64;
    for (unsigned z = 0; z < 64; z++) level.draw_groups[z] = z / 2;
    for (unsigned frame = 0; frame < 32; frame++) {
        unsigned count = frame == 31 ? 0 : (frame * 43) % 257;
        memset(objects, 0, sizeof(objects));
        for (unsigned i = 0; i < 256; i++) {
            uint8_t *obj = objects + i * OBJECT_SIZE;
            put_word(obj, i % 27 ? i : 300);
            put_word(obj + 12, i % 29 ? (i + frame) % 64 : -1);
            if (frame % 8 == 7 && i == 5) put_word(obj + 12, 300);
            obj[6] = i % 11 ? 32 : OBJ_3D_SPRITE;
            points[i].z = ROT_Z_FROM_INT((int)((i * 7 + frame) % 151) - 25);
        }
        if (count < 256) put_word(objects + count * OBJECT_SIZE, -1);
        for (unsigned pool = 0; pool < 2; pool++) {
            uint8_t *data = pool ? player : nasty;
            unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
            for (unsigned i = 0; i < slots; i++) {
                uint8_t *shot = data + i * OBJECT_SIZE;
                put_word(shot, (i * 3) % 256);
                put_word(shot + 12, i % 7 ? i % 64 : -1);
                shot[16] = i % 5 ? OBJ_NBR_BULLET : OBJ_NBR_ALIEN;
            }
        }
        dos_objects_prepare(&level, points, near_z);
        int indexed = 1;
        for (unsigned i = 0; i < count; i++) {
            const uint8_t *obj = objects + i * OBJECT_SIZE;
            int p = get_word(obj), z = get_word(obj + 12);
            if ((unsigned)p >= 256 || z < 0 || points[p].z <= (obj[6] == OBJ_3D_SPRITE ? 0 : near_z)) continue;
            if (z >= 256) indexed = 0;
        }
        for (int zone = 0; zone < 64; zone++) {
            int16_t adjacent[] = {zone - 1, zone + 1, zone + 2, -1};
            uint8_t actual[256], expected[256];
            unsigned total = 0;
            for (unsigned i = 0; i < count; i++) {
                const uint8_t *obj = objects + i * OBJECT_SIZE;
                int p = get_word(obj), z = get_word(obj + 12);
                int polygon = obj[6] == OBJ_3D_SPRITE;
                if ((unsigned)p >= 256 || z < 0 || points[p].z <= (polygon ? 0 : near_z)) continue;
                int connected = 0;
                for (unsigned a = 0; a < 4; a++) if (adjacent[a] == z) connected = 1;
                if (indexed && !level_same_draw_room(&level, z, zone) && (polygon || !connected)) continue;
                expected[total++] = i;
            }
            unsigned found = dos_objects_for_room(&level, zone, adjacent, 4, actual);
            assert(found == total && !memcmp(actual, expected, total));
        }
        for (unsigned pool = 0; pool < 2; pool++) {
            const uint8_t *data = pool ? player : nasty;
            unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
            uint8_t expected[NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT];
            unsigned total = 0, found;
            for (unsigned i = 0; i < slots; i++) {
                const uint8_t *obj = data + i * OBJECT_SIZE;
                int point = get_word(obj);
                if (get_word(obj + 12) >= 0 && obj[16] == OBJ_NBR_BULLET &&
                    point >= 0 && point < 256 && points[point].z > near_z) expected[total++] = i;
            }
            const uint8_t *actual = dos_objects_shots(pool, &found);
            assert(found == total && !memcmp(actual, expected, total));
        }
    }
    level.num_object_points = -1;
    dos_objects_prepare(&level, points, near_z);
    uint8_t empty[256];
    assert(dos_objects_for_room(&level, 0, NULL, 0, empty) == 0);
    for (unsigned pool = 0; pool < 2; pool++) {
        unsigned total;
        dos_objects_shots(pool, &total);
        assert(total == 0);
    }
    puts("PASS: 2048 room candidate lists and both shot pools retain ObjDraw slot ordering and rejects");
}

void check_movement_touch(void);

int main(void)
{
    freopen("RASTER.ERR","w",stderr);
    setvbuf(stderr,NULL,_IONBF,0);
    setvbuf(stdout,NULL,_IOLBF,0);
    check_division();
    check_floor_step_rounding();
    check_billboard_projection();
    check_sky_step_cache();
    check_word_offset_bounds(); check_projection_rounding(); check_wall_interpolation(); check_wall_texture_walk(); check_palette(); check_water_projection(); check_sprite_palettes(); check_floors(); check_solid_floor_rows(); check_edges(); check_walls();
    check_sprite_words();check_sprite_rows();
    check_sprite_columns();
    check_sprite_batches();
    check_sprite_batch_repeated_rows();
    check_sprite_narrow_coverage();
    check_sprite_history();
    check_water_transport();check_water();
    check_movement_touch();
    check_objects();
    puts("DOS raster regression passed");
    return 0;
}
