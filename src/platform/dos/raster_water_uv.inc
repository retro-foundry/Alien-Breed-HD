/* Native prepared UV sampling for one constant adjacent-row direction.
 * Instantiated only by raster_water_span.inc; colour arithmetic stays there. */
__attribute__((always_inline)) static inline
void WATER_UV_FUNCTION(RendererPixel *out,unsigned count,const RendererPixel *source0,
                      const WATER_PHASE_TYPE *phase,
                      const WATER_BRIGHTEN_TYPE (*brighten)[WATER_BRIGHTEN_SAMPLES],
                      const int8_t *blend,unsigned offset,uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    uintptr_t delta=(uintptr_t)out-(uintptr_t)source0,brighten_half=(uintptr_t)brighten>>1;
    uint8_t alpha=(uint8_t)blend[32];uint32_t saved_v;
    /* AB3DI texturedwater needs the low integer byte and all 16 fraction bits.
     * U puts that byte at the top; V rotates its fraction above its integer.
     * SHRD then forms the original packed UV word, including offset carries.
     * ADD/ADC carries V's fraction into its integer. The unused gap can collect
     * at most eight byte overflows before normalization, never a fraction bit.
     * EBX is borrowed for the blend pointer only when interpolation changes a
     * colour; the common equal/zero-delta paths retain V in that register. */
    u<<=8;du<<=8;v=(v<<16)|((v>>16)&255);dv=(dv<<16)|((dv>>16)&255);
#define WATER_UV_PACKED_AT(OFFSET) \
    WATER_UV_PACKED_SAMPLE \
    WATER_BLEND_APPLY( \
        "movzwl " #OFFSET "(%[source0]),%%eax\n\t", \
        "movzwl " #OFFSET WATER_UV_SOURCE_OFFSET "(%[source0]),%%ecx\n\t", \
        "addl %[brighten],%%edx\n\t", WATER_UV_PACKED_DIFFERENCE, \
        "movsbl (%%ebx,%%ecx),%%ecx\n\t", "movsbl (%%ebx,%%edx),%%edx\n\t", \
        WATER_LANE_LOAD "movl %[saved_v],%[v]\n\t", \
        "movl %[delta],%%edx\n\t" "movw %%ax," #OFFSET "(%[source0],%%edx)\n\t") \
    "addl %[du],%[u]\n\t" \
    "addl %[dv],%[v]\n\t" \
    "adcl $0,%[v]\n\t"
    unsigned full=count&~7u;
    if(full){
        const RendererPixel *end=source0+full;
        __asm__ volatile(
            "1:\n\t"
            WATER_UV_PACKED_AT(0) WATER_UV_PACKED_AT(2) WATER_UV_PACKED_AT(4) WATER_UV_PACKED_AT(6)
            WATER_UV_PACKED_AT(8) WATER_UV_PACKED_AT(10) WATER_UV_PACKED_AT(12) WATER_UV_PACKED_AT(14)
            "andl $0xffff00ff,%[v]\n\t"
            "addl $16,%[source0]\n\t" "cmpl %[end],%[source0]\n\t" "jne 1b"
            : [source0] "+&S"(source0),[u] "+&r"(u),[v] "+&b"(v),[saved_v] "=&m"(saved_v)
            : [phase] "D"(phase),[blend_ptr] "m"(blend),[alpha] "m"(alpha),[offset] "m"(offset),
              [brighten] "m"(brighten_half),[du] "m"(du),[dv] "m"(dv),[end] "m"(end),[delta] "m"(delta),WATER_TABLE_INPUTS
            : "eax","ecx","edx","memory","cc");
    }
    count&=7;
    if(count){
        const RendererPixel *end=source0+count;
        __asm__ volatile(
            "1:\n\t" WATER_UV_PACKED_AT(0)
            "addl $2,%[source0]\n\t" "cmpl %[end],%[source0]\n\t" "jne 1b"
            : [source0] "+&S"(source0),[u] "+&r"(u),[v] "+&b"(v),[saved_v] "=&m"(saved_v)
            : [phase] "D"(phase),[blend_ptr] "m"(blend),[alpha] "m"(alpha),[offset] "m"(offset),
              [brighten] "m"(brighten_half),[du] "m"(du),[dv] "m"(dv),[end] "m"(end),[delta] "m"(delta),WATER_TABLE_INPUTS
            : "eax","ecx","edx","memory","cc");
    }
#undef WATER_UV_PACKED_AT
}
