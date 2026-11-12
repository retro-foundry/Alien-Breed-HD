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

#include "dos_sprite_raster_cases.h"

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
    unsigned shot_queries = 0;
    for (unsigned frame = 0; frame < 32; frame++) {
        for (unsigned z = 0; z < 64; z++) level.draw_groups[z] = z / 2;
        if (frame % 8 == 1) level.draw_groups[5] = 300;
        if (frame % 8 == 2) level.draw_groups[63] = -1;
        if (frame % 8 == 5) level.draw_groups[63] = 255;
        if (frame % 8 == 6) level.draw_groups[0] = 511;
        level.nasty_shot_data = frame == 30 || frame == 31 ? NULL : nasty;
        level.player_shot_data = frame == 29 || frame == 31 ? NULL : player;
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
        if (frame == 0 || frame == 16)
            for (unsigned i = 0; i < 256; i++) points[i].z = near_z + 1;
        for (unsigned pool = 0; pool < 2; pool++) {
            uint8_t *data = pool ? player : nasty;
            unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
            for (unsigned i = 0; i < slots; i++) {
                uint8_t *shot = data + i * OBJECT_SIZE;
                put_word(shot, (i * 3) % 256);
                put_word(shot + 12, frame == 16 ? 0 : frame == 0 ? i % 64 : i % 7 ? i % 64 : -1);
                shot[16] = frame == 0 || frame == 16 || i % 5 ? OBJ_NBR_BULLET : OBJ_NBR_ALIEN;
            }
            if (frame != 0 && frame != 16) {
                static const int16_t zones[] = {0,63,255,256,32767,-1,64,5};
                put_word(data, 255); put_word(data + 12, zones[frame % 8]);
                data[16] = OBJ_NBR_BULLET; points[255].z = near_z + 1;
                uint8_t *last = data + (slots - 1) * OBJECT_SIZE;
                put_word(last, 254); put_word(last + 12, 31 + pool);
                last[16] = OBJ_NBR_BULLET; points[254].z = near_z + 1;
                if (frame % 5 == 0) put_word(data + OBJECT_SIZE, 300);
            }
        }
        dos_objects_prepare(&level, points, near_z);
        int indexed = 1;
        for (unsigned i = 0; i < count; i++) {
            const uint8_t *obj = objects + i * OBJECT_SIZE;
            int p = get_word(obj), z = get_word(obj + 12);
            if ((unsigned)p >= 256 || z < 0 || points[p].z <= (obj[6] == OBJ_3D_SPRITE ? 0 : near_z)) continue;
            int group = z < level.draw_group_count ? level.draw_groups[z] : z;
            if (z >= 256 || (unsigned)group >= 256) indexed = 0;
        }
        for (int zone = 0; zone < 64; zone++) {
            int16_t adjacent[] = {zone - 1, zone + 1, zone + 2, -1};
            uint8_t actual[256], expected[256];
            unsigned total = 0;
            int query_group = level.draw_groups[zone];
            for (unsigned i = 0; i < count; i++) {
                const uint8_t *obj = objects + i * OBJECT_SIZE;
                int p = get_word(obj), z = get_word(obj + 12);
                int polygon = obj[6] == OBJ_3D_SPRITE;
                if ((unsigned)p >= 256 || z < 0 || points[p].z <= (polygon ? 0 : near_z)) continue;
                int connected = 0;
                for (unsigned a = 0; a < 4; a++) if (adjacent[a] == z) connected = 1;
                if (indexed && (unsigned)query_group < 256 &&
                    !level_same_draw_room(&level, z, zone) && (polygon || !connected)) continue;
                expected[total++] = i;
            }
            unsigned found = dos_objects_for_room(&level, zone, adjacent, 4, actual);
            assert(found == total && !memcmp(actual, expected, total));
        }
        for (unsigned pool = 0; pool < 2; pool++) {
            const uint8_t *data = pool ? level.player_shot_data : level.nasty_shot_data;
            unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
            uint8_t expected[NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT];
            unsigned total = 0, found;
            for (unsigned i = 0; data && i < slots; i++) {
                const uint8_t *obj = data + i * OBJECT_SIZE;
                int point = get_word(obj);
                if (get_word(obj + 12) >= 0 && obj[16] == OBJ_NBR_BULLET &&
                    point >= 0 && point < 256 && points[point].z > near_z) expected[total++] = i;
            }
            const uint8_t *actual = dos_objects_shots(pool, &found);
            assert(found == total && !memcmp(actual, expected, total));
        }
        /* Compare room preselection against the original per-slot ownership
         * and raw-adjacency tests, including conservative wide-zone scans. */
        int shots_indexed = 1;
        for (unsigned pool = 0; pool < 2; pool++) {
            const uint8_t *data = pool ? level.player_shot_data : level.nasty_shot_data;
            unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
            for (unsigned i = 0; data && i < slots; i++) {
                const uint8_t *obj = data + i * OBJECT_SIZE;
                int point = get_word(obj), z = get_word(obj + 12);
                if (z < 0 || obj[16] != OBJ_NBR_BULLET || point < 0 || point >= 256 || points[point].z <= near_z) continue;
                int group = z < level.draw_group_count ? level.draw_groups[z] : z;
                if ((unsigned)z >= 256 || (unsigned)group >= 256) shots_indexed = 0;
            }
        }
        static const int16_t extra_zones[] = {-1,64,255,256,300,32767,-32768};
        for (unsigned query = 0; query < 64 + sizeof(extra_zones)/sizeof(*extra_zones); query++) {
            int zone = query < 64 ? (int)query : extra_zones[query - 64];
            int group = zone >= 0 && zone < level.draw_group_count ? level.draw_groups[zone] : zone;
            int16_t adjacent[6] = {(int16_t)(zone-1),(int16_t)(zone+1),(int16_t)(zone+1),
                                   (int16_t)zone,255,32767};
            if (frame & 1) { adjacent[0] = -1; adjacent[2] = 256; adjacent[3] = (int16_t)(zone ^ 1); }
            unsigned adjacent_count = frame % 4 ? 6 : 0;
            const int16_t *neighbors = adjacent_count ? adjacent : NULL;
            uint8_t actual[NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT + 16];
            uint8_t expected[NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT];
            unsigned total = 0, expected_counts[2] = {0,0};
            for (unsigned pool = 0; pool < 2; pool++) {
                const uint8_t *data = pool ? level.player_shot_data : level.nasty_shot_data;
                unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
                for (unsigned i = 0; data && i < slots; i++) {
                    const uint8_t *obj = data + i * OBJECT_SIZE;
                    int point = get_word(obj), z = get_word(obj + 12);
                    if (z < 0 || obj[16] != OBJ_NBR_BULLET || point < 0 || point >= 256 || points[point].z <= near_z) continue;
                    int connected = 0;
                    for (unsigned a = 0; a < adjacent_count; a++) if (neighbors[a] == z) connected = 1;
                    if (shots_indexed && (unsigned)group < 256 &&
                        !level_same_draw_room(&level,z,zone) && !connected) continue;
                    expected[total++] = (uint8_t)i; expected_counts[pool]++;
                }
            }
            memset(actual,0x91,sizeof(actual));
            unsigned counts[4] = {0x13579bdu,~0u,~0u,0x2468aceu};
            dos_objects_shots_for_room(&level,zone,neighbors,adjacent_count,counts+1,actual+8);
            assert(counts[0] == 0x13579bdu && counts[3] == 0x2468aceu);
            assert(counts[1] == expected_counts[0] && counts[2] == expected_counts[1]);
            assert(!memcmp(actual+8,expected,total));
            for (unsigned i = 0; i < 8; i++) assert(actual[i] == 0x91);
            for (unsigned i = total+8; i < sizeof(actual); i++) assert(actual[i] == 0x91);
            shot_queries++;
        }
    }
    level.num_object_points = -1;
    level.nasty_shot_data = nasty; level.player_shot_data = player;
    dos_objects_prepare(&level, points, near_z);
    uint8_t empty[256];
    assert(dos_objects_for_room(&level, 0, NULL, 0, empty) == 0);
    for (unsigned pool = 0; pool < 2; pool++) {
        unsigned total;
        dos_objects_shots(pool, &total);
        assert(total == 0);
    }
    unsigned no_shots[2] = {~0u,~0u};
    dos_objects_shots_for_room(&level,0,NULL,0,no_shots,empty);
    assert(no_shots[0] == 0 && no_shots[1] == 0);
    puts("PASS: 2048 room candidate lists and both shot pools retain ObjDraw slot ordering and rejects");
    printf("PASS: %u room shot queries preserve both pools, slot order, aliases, duplicate neighbors, wide IDs/groups, invalid slots, empty pools, frame resets and guards\n",shot_queries);
}

void check_movement_touch(void);
void check_movement_paths(void);

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
    check_sprite_dda_rows();
    check_water_transport();check_water();
    check_movement_touch();
    check_movement_paths();
    check_objects();
    puts("DOS raster regression passed");
    return 0;
}
