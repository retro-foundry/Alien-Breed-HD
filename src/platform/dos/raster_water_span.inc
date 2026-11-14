#include "raster_water_color.inc"

/* Shared byte/prepared phase kernels. Instantiated only by raster_dos.c. */
/* A constant one_row selects an already resolved 64-texel phase row.
 * Both samplers share the original native Brighten/blend operations. */
__attribute__((always_inline)) static inline
void WATER_FUNCTION(water_span)(RendererPixel *out,unsigned count,
                const RendererPixel *background0,const RendererPixel *background1,
                const WATER_PHASE_TYPE *phase,const WATER_BRIGHTEN_TYPE (*brighten)[WATER_BRIGHTEN_SAMPLES],const int8_t *blend,
                unsigned offset,uint32_t u,uint32_t v,uint32_t du,uint32_t dv,int one_row)
{
    /* AB3DI.s texturedwater wraps this selected phase row every 64
     * texels. Keep its six integer bits at the top of the accumulator;
     * unsigned overflow then supplies the exact modulo-64 wrap. */
    if(one_row){u=(u+((uint32_t)offset<<16))<<10;du<<=10;}
    if(!background1) {
        for(unsigned x=0;x<count;x++,u+=du,v+=dv) {
            unsigned level;
            if(one_row)level=WATER_LEVEL(phase[u>>26]);
            else {
                uint16_t uv=(uint16_t)(((v>>8)&0xff00)|((u>>16)&255));
                uv=(uint16_t)(uv+offset);
                level=WATER_LEVEL(phase[((uv>>2)&0xfc0)|(uv&63)]);
            }
#ifdef AB3D_DOS_INDEXED
            unsigned row=(unsigned)(brighten-water_brighten)+level;
            out[x]=water_unblended[row][water_gb_index[background0[x]]];
#else
            out[x]=(RendererPixel)WATER_SAMPLE(brighten,level,background0[x]);
#endif
        }
        return;
    }
    for(unsigned x=0;x<count;x++,u+=du,v+=dv) {
        unsigned level;
        if(one_row)level=WATER_LEVEL(phase[u>>26]);
        else {
            uint16_t uv=(uint16_t)(((v>>8)&0xff00)|((u>>16)&255));
            uv=(uint16_t)(uv+offset);
            level=WATER_LEVEL(phase[((uv>>2)&0xfc0)|(uv&63)]);
        }
        uint32_t c0=WATER_SAMPLE(brighten,level,background0[x]);
        uint32_t c1=WATER_SAMPLE(brighten,level,background1[x]);
        if(c0==c1) {
            out[x]=(RendererPixel)c0;
            continue;
        }
        /* The three guard bits prevent a negative channel difference from
         * borrowing into the next lane. Each lookup still uses delta+16. */
        uint32_t difference=((WATER_COLOR_LANES(c1))|0x4210u)-(WATER_COLOR_LANES(c0));
        /* AB3DI Brighten channels and the existing rounded interpolation:
         * add signed deltas at their original weights. Each resulting channel
         * is 0..15, so the complete sum leaves both lane guard bits clear. */
        unsigned lanes=(WATER_COLOR_LANES(c0))+blend[(difference>>10)&31]*1024+
            blend[(difference>>5)&31]*32+blend[difference&31];
#ifdef AB3D_NATIVE_RGB565
        out[x]=water_rg[lanes>>5]|water_blue[lanes&15];
#else
        out[x]=water_lanes[lanes];
#endif
    }
}


/* EDX starts with the AB3DI.s WaterFile level. Read both background
 * pixels before writing, preserving forward order even for overlapping spans.
 * Keep the exact signed delta arithmetic and native lane conversion shared by
 * the horizontal and general UV samplers. EAX/ECX/EDX are scratch. */
#define WATER_BLEND_APPLY(LOAD0, LOAD1, BRIGHTEN, DIFFERENCE, BLEND_CX, BLEND_DX, LANES, STORE) \
    WATER_LEVEL_SHIFT \
    BRIGHTEN \
    LOAD0 \
    WATER_MASK0 \
    WATER_GB0 \
    WATER_BRIGHTEN0 \
    LOAD1 \
    WATER_MASK1 \
    WATER_GB1 \
    WATER_BRIGHTEN1 \
    "cmpl %%eax,%%edx\n\t" \
    "je 2f\n\t" \
    DIFFERENCE \
    "movl %%edx,%%ecx\n\t" \
    WATER_RED_DIFFERENCE \
    "andl $31,%%ecx\n\t" \
    BLEND_CX \
    "sall $10,%%ecx\n\t" \
    "addl %%ecx,%%eax\n\t" \
    "movl %%edx,%%ecx\n\t" \
    WATER_GREEN_DIFFERENCE \
    "andl $31,%%ecx\n\t" \
    BLEND_CX \
    "sall $5,%%ecx\n\t" \
    "addl %%ecx,%%eax\n\t" \
    WATER_BLUE_DIFFERENCE \
    BLEND_DX \
    "addl %%edx,%%eax\n\t" \
    LANES \
    "2:\n\t" \
    STORE
#define WATER_BLEND_CORE(LOAD0, LOAD1, BRIGHTEN, LANES, STORE) \
    WATER_BLEND_APPLY(LOAD0, LOAD1, BRIGHTEN, WATER_DIFFERENCE, \
        "movsbl (%[blend],%%ecx),%%ecx\n\t", \
        "movsbl (%[blend],%%edx),%%edx\n\t", LANES, STORE)
#define WATER_BLEND_PIXEL WATER_BLEND_CORE(WATER_LOAD0, WATER_LOAD1, \
    "addl %[brighten],%%edx\n\t", \
    WATER_LANE_LOAD, "movl %[delta],%%edx\n\t" WATER_STORE)


#define WATER_ROW_SAMPLE \
    "movl %[phase],%%ecx\n\t" \
    "movl %[u],%%eax\n\t" \
    "shrl $26,%%eax\n\t" \
    WATER_ROW_FETCH
#ifdef AB3D_DOS_INDEXED
#define WATER_UV_WORD_MASK "andl $0x3f3f,%%eax\n\t"
#else
#define WATER_UV_WORD_MASK ""
#endif
#ifdef AB3D_NATIVE_RGB565
/* AB3DI texturedwater packs the integer U/V bytes before adding wateroff.
 * On the little-endian 486, reading U's upper word then replacing AH with
 * V's integer byte builds that exact word in two instructions. Keep phase
 * in the freed register and advance the full 32-bit U accumulator in RAM. */
#define WATER_UV_WORD \
    "movzwl %[ui],%%eax\n\t" \
    "movb %[vi],%%ah\n\t"
#define WATER_UV_PHASE WATER_NATIVE_UV_FETCH
#define WATER_UV_ADVANCE_U "movl %[du],%%eax\n\t" "addl %%eax,%[u]\n\t"
#define WATER_UV_OUTPUTS [u] "+m"(u),[v] "+m"(v)
#define WATER_UV_INPUTS [phase] "r"(phase), \
    [ui] "m"(((const uint16_t *)&u)[1]),[vi] "m"(((const uint8_t *)&v)[2])
#else
#define WATER_UV_WORD \
    "movl %[u],%%eax\n\t" \
    "shrl $16,%%eax\n\t" \
    "andl $255,%%eax\n\t" \
    "movl %[v],%%ecx\n\t" \
    "shrl $8,%%ecx\n\t" \
    "andl $0xff00,%%ecx\n\t" \
    "orl %%ecx,%%eax\n\t"
#define WATER_UV_PHASE \
    "movl %[phase],%%ecx\n\t" \
    "movzbl (%%ecx,%%edx),%%edx\n\t"
#define WATER_UV_ADVANCE_U "addl %[du],%[u]\n\t"
#define WATER_UV_OUTPUTS [u] "+&r"(u),[v] "+m"(v)
#define WATER_UV_INPUTS [phase] "m"(phase)
#endif
#define WATER_UV_SAMPLE \
    WATER_UV_WORD \
    "addl %[offset],%%eax\n\t" \
    WATER_UV_WORD_MASK \
    "movl %%eax,%%edx\n\t" \
    "shrl $2,%%edx\n\t" \
    "andl $0xfc0,%%edx\n\t" \
    "andl $63,%%eax\n\t" \
    "orl %%eax,%%edx\n\t" \
    WATER_UV_PHASE

/* du/u already place the modulo-64 coordinate in the high six bits. */
__attribute__((always_inline)) static inline
void WATER_FUNCTION(water_span_blended_row)(RendererPixel *out,unsigned count,const RendererPixel *source0,
                       const RendererPixel *source1,const WATER_PHASE_TYPE *phase,
                       const WATER_BRIGHTEN_TYPE (*brighten)[WATER_BRIGHTEN_SAMPLES],const int8_t *blend,uint32_t u,uint32_t du)
{
    if(!count)return;
    uintptr_t delta=(uintptr_t)out-(uintptr_t)source0;
#ifdef AB3D_NATIVE_RGB565
    uintptr_t brighten_half=(uintptr_t)brighten>>1;
#endif
#ifdef AB3D_NATIVE_RGB565
#ifdef WATER_RUN_HINTS
    /* U keeps six integer bits above the fractional 26-bit field. Requiring
     * ceil(3*du/one_texel) extra equal texels covers even the largest starting
     * fraction, so a group needs only one byte comparison. Preserve per-pixel
     * phase reads if output can overwrite this row of the source bank. */
    uint8_t minimum=WATER_PHASE_RUN_MAX<<WATER_PHASE_RUN_SHIFT;
    uintptr_t output=(uintptr_t)out,table=(uintptr_t)phase;
    if(du && du<=0x04000000u && output-table>=64*sizeof(*phase) && table-output>=count*sizeof(*out))
        minimum=(uint8_t)(((du*3+0x03ffffffu)>>26)<<WATER_PHASE_RUN_SHIFT);
    uint32_t quad_du=du<<2;
    uintptr_t chosen;
#endif
#define WATER_ROW_PIXEL_AT(OFFSET,BRIGHTEN) \
        WATER_BLEND_CORE( \
            "movzwl " #OFFSET "(%[source0]),%%eax\n\t", \
            "movzwl " #OFFSET "(%[source1]),%%ecx\n\t", \
            BRIGHTEN, \
            WATER_LANE_LOAD, \
            "movl %[delta],%%edx\n\t" "movw %%ax," #OFFSET "(%[source0],%%edx)\n\t")
#define WATER_ROW_AT(OFFSET) \
        WATER_ROW_SAMPLE \
        WATER_ROW_PIXEL_AT(OFFSET,"addl %[brighten],%%edx\n\t") \
        "addl %[du],%[u]\n\t"
    /* AB3DI texturedwater still reads and writes each pixel in forward order.
     * Four pixels share source-pointer advances and the loop test. The fixed
     * offsets also preserve overlapping-source behavior within each group. */
    unsigned full=count&~3u;
    if(full) {
        const RendererPixel *end=source0+full;
        __asm__ volatile(
            "1:\n\t"
#ifdef WATER_RUN_HINTS
            WATER_ROW_SAMPLE
            "cmpb %[minimum],%%dl\n\t"
            "jb 8f\n\t"
            "addl %[brighten],%%edx\n\t"
            "movl %%edx,%[chosen]\n\t"
            WATER_ROW_PIXEL_AT(0,"")
            WATER_ROW_PIXEL_AT(2,"movl %[chosen],%%edx\n\t")
            WATER_ROW_PIXEL_AT(4,"movl %[chosen],%%edx\n\t")
            WATER_ROW_PIXEL_AT(6,"movl %[chosen],%%edx\n\t")
            "addl %[quad_du],%[u]\n\t"
            "jmp 9f\n\t"
            "8:\n\t"
            WATER_ROW_PIXEL_AT(0,"addl %[brighten],%%edx\n\t")
            "addl %[du],%[u]\n\t"
            WATER_ROW_AT(2) WATER_ROW_AT(4) WATER_ROW_AT(6)
            "9:\n\t"
#else
            WATER_ROW_AT(0)
            WATER_ROW_AT(2)
            WATER_ROW_AT(4)
            WATER_ROW_AT(6)
#endif
            "addl $8,%[source0]\n\t"
            "addl $8,%[source1]\n\t"
            "cmpl %[end],%[source0]\n\t"
            "jne 1b"
            : [source0] "+&S"(source0),[source1] "+&D"(source1),[u] "+&r"(u)
#ifdef WATER_RUN_HINTS
              ,[chosen] "=&m"(chosen)
#endif
            : [blend] "b"(blend),[phase] "m"(phase),[brighten] "m"(WATER_BRIGHTEN_ADDRESS),[du] "m"(du),
#ifdef WATER_RUN_HINTS
              [minimum] "m"(minimum),[quad_du] "m"(quad_du),
#endif
              [end] "m"(end),[delta] "m"(delta),WATER_GB_INPUT WATER_TABLE_INPUTS
            : "eax","ecx","edx","memory","cc");
    }
    count&=3;
    if(!count)return;
#undef WATER_ROW_AT
#undef WATER_ROW_PIXEL_AT
#endif
    const RendererPixel *end=source0+count;
    __asm__ volatile(
        "1:\n\t"
        WATER_ROW_SAMPLE
        WATER_BLEND_PIXEL
        WATER_ADVANCE
        "addl %[du],%[u]\n\t"
        "cmpl %[end],%[source0]\n\t"
        "jne 1b"
        : [source0] "+&S"(source0),[source1] "+&D"(source1),[u] "+&r"(u)
        : [blend] "b"(blend),[phase] "m"(phase),[brighten] "m"(WATER_BRIGHTEN_ADDRESS),[du] "m"(du),
          [end] "m"(end),[delta] "m"(delta),WATER_GB_INPUT WATER_TABLE_INPUTS
        : "eax","ecx","edx","memory","cc");
}

#ifdef WATER_PACKED_UV
/* Adjacent RGB565 rows free EDI for the phase pointer and EBX for V.
 * Retain the shared Brighten/difference/blend operations in both directions. */
#define WATER_UV_PACKED_SAMPLE \
    "movl %[u],%%eax\n\t" \
    "shrdl $24,%[v],%%eax\n\t" \
    "addl %[offset],%%eax\n\t" \
    "movl %%eax,%%edx\n\t" \
    "shrl $2,%%edx\n\t" \
    "andl $0xfc0,%%edx\n\t" \
    "andl $63,%%eax\n\t" \
    "orl %%eax,%%edx\n\t" \
    WATER_NATIVE_UV_FETCH
#define WATER_UV_PACKED_DIFFERENCE WATER_NATIVE_DIFFERENCE( \
    "cmpb %[alpha],%%cl\n\t", \
    "movl %[v],%[saved_v]\n\t" "movl %[blend_ptr],%%ebx\n\t")
#define WATER_UV_FUNCTION WATER_FUNCTION(water_uv_forward)
#define WATER_UV_SOURCE_OFFSET "+640"
#include "raster_water_uv.inc"
#undef WATER_UV_FUNCTION
#undef WATER_UV_SOURCE_OFFSET
#define WATER_UV_FUNCTION WATER_FUNCTION(water_uv_backward)
#define WATER_UV_SOURCE_OFFSET "-640"
#include "raster_water_uv.inc"
#undef WATER_UV_FUNCTION
#undef WATER_UV_SOURCE_OFFSET
#undef WATER_UV_PACKED_SAMPLE
#undef WATER_UV_PACKED_DIFFERENCE
#endif

/* AB3DI.s texturedwater forms the packed 16-bit UV word before adding
 * wateroff. Masking the 32-bit sum to 0x3f3f preserves both byte and word
 * carries; the two fixed-point accumulators still wrap independently. */
__attribute__((always_inline)) static inline
void WATER_FUNCTION(water_span_blended_uv)(RendererPixel *out,unsigned count,const RendererPixel *source0,
                       const RendererPixel *source1,const WATER_PHASE_TYPE *phase,
                       const WATER_BRIGHTEN_TYPE (*brighten)[WATER_BRIGHTEN_SAMPLES],const int8_t *blend,unsigned offset,uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    if(!count)return;
#ifdef WATER_PACKED_UV
    /* The renderer refracts between adjacent 320-pixel rows. Other callers
     * retain the general two-pointer kernel and its full stride contract. */
    uintptr_t source_delta=(uintptr_t)source1-(uintptr_t)source0;
    if(source_delta==640u){WATER_FUNCTION(water_uv_forward)(out,count,source0,phase,brighten,blend,offset,u,v,du,dv);return;}
    if(source_delta==0u-640u){WATER_FUNCTION(water_uv_backward)(out,count,source0,phase,brighten,blend,offset,u,v,du,dv);return;}
#endif
    uintptr_t delta=(uintptr_t)out-(uintptr_t)source0;
#ifdef AB3D_NATIVE_RGB565
    uintptr_t brighten_half=(uintptr_t)brighten>>1;
#endif
#ifdef AB3D_NATIVE_RGB565
#define WATER_UV_AT(OFFSET) \
        WATER_UV_SAMPLE \
        WATER_BLEND_CORE( \
            "movzwl " #OFFSET "(%[source0]),%%eax\n\t", \
            "movzwl " #OFFSET "(%[source1]),%%ecx\n\t", \
            "addl %[brighten],%%edx\n\t", \
            WATER_LANE_LOAD, \
            "movl %[delta],%%edx\n\t" "movw %%ax," #OFFSET "(%[source0],%%edx)\n\t") \
        WATER_UV_ADVANCE_U \
        "movl %[dv],%%eax\n\t" \
        "addl %%eax,%[v]\n\t"
    /* AB3DI texturedwater still reads and writes each pixel in forward order.
     * Four pixels share source-pointer advances and the loop test. The fixed
     * offsets also preserve overlapping-source behavior within each group. */
    unsigned full=count&~3u;
    if(full) {
        const RendererPixel *end=source0+full;
        __asm__ volatile(
            "1:\n\t"
            WATER_UV_AT(0)
            WATER_UV_AT(2)
            WATER_UV_AT(4)
            WATER_UV_AT(6)
            "addl $8,%[source0]\n\t"
            "addl $8,%[source1]\n\t"
            "cmpl %[end],%[source0]\n\t"
            "jne 1b"
            : [source0] "+&S"(source0),[source1] "+&D"(source1),WATER_UV_OUTPUTS
            : [blend] "b"(blend),WATER_UV_INPUTS,[brighten] "m"(WATER_BRIGHTEN_ADDRESS),[du] "m"(du),
              [dv] "m"(dv),[offset] "m"(offset),[end] "m"(end),[delta] "m"(delta),
              WATER_GB_INPUT WATER_TABLE_INPUTS
            : "eax","ecx","edx","memory","cc");
    }
    count&=3;
    if(!count)return;
#undef WATER_UV_AT
#endif
    const RendererPixel *end=source0+count;
    __asm__ volatile(
        "1:\n\t"
        WATER_UV_SAMPLE
        WATER_BLEND_PIXEL
        WATER_ADVANCE
        WATER_UV_ADVANCE_U
        "movl %[dv],%%eax\n\t"
        "addl %%eax,%[v]\n\t"
        "cmpl %[end],%[source0]\n\t"
        "jne 1b"
        : [source0] "+&S"(source0),[source1] "+&D"(source1),WATER_UV_OUTPUTS
        : [blend] "b"(blend),[brighten] "m"(WATER_BRIGHTEN_ADDRESS),
          WATER_UV_INPUTS,[du] "m"(du),[dv] "m"(dv),[offset] "m"(offset),
          [end] "m"(end),[delta] "m"(delta),WATER_GB_INPUT WATER_TABLE_INPUTS
        : "eax","ecx","edx","memory","cc");
}

#undef WATER_ROW_SAMPLE
#undef WATER_UV_SAMPLE
#undef WATER_UV_WORD
#undef WATER_UV_PHASE
#undef WATER_UV_ADVANCE_U
#undef WATER_UV_OUTPUTS
#undef WATER_UV_INPUTS
#undef WATER_UV_WORD_MASK
#undef WATER_DIFFERENCE
#undef WATER_BLEND_CORE
#undef WATER_BLEND_APPLY
#undef WATER_NATIVE_DIFFERENCE
#undef WATER_BLEND_PIXEL
#undef WATER_LOAD0
#undef WATER_LOAD1
#undef WATER_STORE
#undef WATER_ADVANCE
#undef WATER_TABLE_INPUTS
#undef WATER_LANE_LOAD

/* Keep horizontal sampling state separate from byte-carry segmentation. */
__attribute__((noinline,optimize("O3,unroll-loops")))
static void WATER_FUNCTION(water_span_row)(RendererPixel *out,unsigned count,
                           const RendererPixel *background0,const RendererPixel *background1,
                           const WATER_PHASE_TYPE *phase,const WATER_BRIGHTEN_TYPE (*brighten)[WATER_BRIGHTEN_SAMPLES],const int8_t *blend,
                           unsigned offset,uint32_t u,uint32_t du)
{
    if(!background1)WATER_FUNCTION(water_span)(out,count,background0,NULL,phase,brighten,blend,offset,u,0,du,0,1);
    else WATER_FUNCTION(water_span_blended_row)(out,count,background0,background1,phase,brighten,blend,(u+(offset<<16))<<10,du<<10);
}

__attribute__((noinline,optimize("O3,unroll-loops")))
void WATER_FUNCTION(dos_raster_water)(RendererPixel *out,unsigned count,
                        const RendererPixel *background0,const RendererPixel *background1,
                        const WATER_PHASE_TYPE *phase,unsigned shade,unsigned offset,unsigned alpha,
                        uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    const WATER_BRIGHTEN_TYPE (*brighten)[WATER_BRIGHTEN_SAMPLES]=water_brighten+shade;
    /* Exact rounded interpolation is symmetric under swapping the samples. */
    if(background1 && alpha>127) {
        const RendererPixel *swap=background0;background0=background1;background1=swap;
        alpha=255-alpha;
    }
#ifndef AB3D_DOS_INDEXED
    /* At half-alpha 0..8 every possible 4-bit channel delta rounds to zero.
     * Indexed blending still needs its second palette conversion. */
    if(alpha<=8 || background0==background1)background1=NULL;
#endif
#ifdef AB3D_NATIVE_RGB565
    const int8_t *blend=background1?water_blend_delta[alpha].delta:NULL;
#else
    const int8_t *blend=background1?water_blend_delta[alpha]:NULL;
#endif
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
            WATER_FUNCTION(water_span_row)(out,run,background0,background1,phase+(((base+carry)&63)*64),
                           brighten,blend,bias,u,du);
            out+=run;background0+=run;if(background1)background1+=run;
            u+=run*du;count-=run;
        }
    }
    else if(background1)WATER_FUNCTION(water_span_blended_uv)(out,count,background0,background1,phase,brighten,blend,offset,u,v,du,dv);
    else WATER_FUNCTION(water_span)(out,count,background0,background1,phase,brighten,blend,offset,u,v,du,dv,0);
}


#undef WATER_BRIGHTEN_TYPE
#undef WATER_BRIGHTEN_SAMPLES
#undef WATER_SAMPLE
#undef WATER_COLOR_LANES
#undef WATER_MASK0
#undef WATER_MASK1
#undef WATER_GB0
#undef WATER_GB1
#undef WATER_BRIGHTEN0
#undef WATER_BRIGHTEN1
#undef WATER_RED_DIFFERENCE
#undef WATER_GREEN_DIFFERENCE
#undef WATER_BLUE_DIFFERENCE
#undef WATER_BRIGHTEN_ADDRESS
#undef WATER_GB_INPUT
